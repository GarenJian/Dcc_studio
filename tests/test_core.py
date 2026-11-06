"""Unit tests for the numerical core, using synthetic data only.

These tests are fast and fully deterministic: they use small random or hand
built matrices to pin down individual formulas rather than a whole run. They
cover ROI/network aggregation, sliding-window correlation shapes and values,
sign handling, global/ROI thresholding, state encoding and decoding, Shannon
entropy, the per-sample ``connectome_complex`` entry point, atlas validation,
group summaries, ICC, two-sample t tests, Cohen counts, FDR correction, PLS
basics and configuration round-tripping / validation.

Because the inputs are synthetic, a failure here points to a specific helper;
the real-data regression cases in ``test_reference_parity.py`` complement them
by checking the published numerical outputs.
"""
from __future__ import annotations

import unittest
import numpy as np

from dcc.atlas import Atlas
from dcc.complexity import (compute_states, connectome_complex, decode_state,
                            state_weights)
from dcc.config import DCCConfig
from dcc.dfc import apply_sign, binarize_dfc, sliding_window_fc
from dcc.network import (group_state_prob, keep_above, net_analysis,
                         network_state_prob, stack_state_prob, state_table,
                         top_state_coverage)
from dcc.pls import bootstrap_stats, colcorr, normalize, pls_analysis
from dcc.stats import (cohen_network_counts, fdr_bh, group_summary, icc_1_1,
                       node_icc, two_sample_ttest)
from dcc.timeseries import get_ts_roi


class TestTimeSeries(unittest.TestCase):
    def test_get_ts_roi(self):
        data = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        out = get_ts_roi(data, np.array([1, 1, 2, 2]))
        np.testing.assert_allclose(out, [[2., 3.], [6., 7.]])

    def test_get_ts_roi_higher_dim(self):
        data = np.arange(24, dtype=float).reshape(4, 3, 2)
        self.assertEqual(get_ts_roi(data, np.array([1, 1, 2, 2])).shape,
                         (2, 3, 2))

    def test_label_length_mismatch(self):
        with self.assertRaises(ValueError):
            get_ts_roi(np.zeros((4, 5)), np.array([1, 2]))

    def test_zero_labels_are_ignored(self):
        data = np.arange(8, dtype=float).reshape(4, 2)
        out = get_ts_roi(data, np.array([0, 1, 1, 2]))
        self.assertEqual(out.shape, (2, 2))


class TestDFC(unittest.TestCase):
    def test_sliding_window_correlation(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=60)
        ts_roi = np.vstack([base, -base, np.full(60, 3.0)])
        dfc = sliding_window_fc(ts_roi, base[None, :], swl=20, sws=1)
        self.assertEqual(dfc.shape, (3, 1, 41))
        np.testing.assert_allclose(dfc[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(dfc[1, 0], -1.0, atol=1e-10)
        np.testing.assert_allclose(dfc[2, 0], 0.0)

    def test_window_count(self):
        ts = np.random.default_rng(1).normal(size=(5, 100))
        dfc = sliding_window_fc(ts, ts[:2], swl=30, sws=4)
        self.assertEqual(dfc.shape[2], 1 + (100 - 30) // 4)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            sliding_window_fc(np.zeros((2, 10)), np.zeros((1, 10)), 20, 1)

    def test_sign_and_binarize(self):
        data = np.array([[[-1., 0.5, 2.]]])
        np.testing.assert_allclose(apply_sign(data, "abs").ravel(),
                                   [1., 0.5, 2.])
        np.testing.assert_allclose(apply_sign(data, "pos").ravel(),
                                   [0., 0.5, 2.])
        np.testing.assert_allclose(apply_sign(data, "raw").ravel(),
                                   [-1., 0.5, 2.])
        with self.assertRaises(ValueError):
            apply_sign(data, "invalid")
        cube = np.arange(27, dtype=float).reshape(3, 3, 3)
        binary, thr, _, _ = binarize_dfc(cube, 0.5)
        self.assertAlmostEqual(thr, cube.mean() + 0.5 * cube.std(ddof=1))
        self.assertEqual(int(binary.sum()), int((cube > thr).sum()))
        _, thr_roi, _, _ = binarize_dfc(cube, 0.5, mode="roi")
        self.assertEqual(thr_roi.shape, (3,))
        with self.assertRaises(ValueError):
            binarize_dfc(cube, 0.5, mode="unknown")


class TestComplexity(unittest.TestCase):
    def test_weights_and_decode(self):
        np.testing.assert_array_equal(state_weights(3), [4, 2, 1])
        self.assertEqual(decode_state(18), "0010010")
        self.assertEqual(decode_state(3, 3), "011")
        self.assertEqual(decode_state(127), "1111111")

    def test_state_counts_and_entropy(self):
        binary = np.zeros((2, 7, 10), dtype=np.uint8)
        binary[:, :, :5] = 1
        binary[:, 6, 5:] = 1
        codes, counts, prob, entropy = compute_states(binary, 7)
        self.assertEqual(codes.shape, (2, 10))
        self.assertEqual(counts.shape, (2, 127))
        self.assertEqual(counts[0, 126], 5)
        self.assertEqual(counts[0, 0], 5)
        self.assertAlmostEqual(prob[0].sum(), 1.0)
        self.assertAlmostEqual(entropy[0], 1.0)

    def test_uniform_entropy(self):
        binary = np.zeros((1, 7, 127), dtype=np.uint8)
        for state in range(1, 128):
            binary[0, :, state - 1] = [int(ch) for ch in
                                       format(state, "07b")]
        _, _, _, entropy = compute_states(binary, 7)
        self.assertAlmostEqual(entropy[0], np.log2(127), places=8)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            compute_states(np.zeros((2, 3, 4)), 7)

    def test_connectome_complex_end_to_end(self):
        rng = np.random.default_rng(11)
        atlas = Atlas.from_labels(np.array([1] * 4 + [2] * 4 + [3] * 4))
        ts = rng.normal(size=(12, 80))
        cfg = DCCConfig(swl=10, sws=2, k=0.5)
        result = connectome_complex(ts, atlas, cfg, subject_id="s1")
        self.assertEqual(result.entropy.shape, (12,))
        self.assertEqual(result.states_prob.shape, (12, 7))
        self.assertEqual(result.dfc_binary.shape, (12, 3, result.n_windows))
        self.assertEqual(result.n_windows, 1 + (80 - 10) // 2)
        self.assertTrue(np.isfinite(result.entropy).all())
        self.assertTrue(np.all(result.entropy >= 0))
        self.assertAlmostEqual(result.states_prob.sum(axis=1).mean(), 1.0)
        self.assertEqual(result.subject_id, "s1")
        with self.assertRaises(ValueError):
            connectome_complex(ts[:5], atlas, cfg)


class TestAtlas(unittest.TestCase):
    def test_arbitrary_size(self):
        atlas = Atlas.from_labels([1, 1, 2, 2, 3], names=("A", "B"))
        self.assertEqual(atlas.n_roi, 5)
        self.assertEqual(atlas.n_net, 3)
        self.assertEqual(atlas.n_states, 7)
        self.assertEqual(atlas.name(2), "Network 3")
        np.testing.assert_array_equal(atlas.sizes, [2, 2, 1])
        self.assertEqual(atlas.boundaries(), [2, 4])
        np.testing.assert_allclose(atlas.aggregate(np.arange(5, dtype=float)),
                                   [0.5, 2.5, 4.0])

    def test_too_many_networks(self):
        with self.assertRaises(ValueError):
            Atlas.from_labels(np.arange(1, 18))


class TestNetworkAndStats(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.atlas = Atlas.from_labels([1] * 4 + [2] * 4)
        self.results = []
        for i in range(4):
            prob = rng.random((8, 3))
            prob /= prob.sum(axis=1, keepdims=True)
            self.results.append(type("R", (), {
                "entropy": rng.random(8),
                "states_prob": prob,
                "states_counts": np.zeros((8, 3), dtype=np.uint16),
            })())

    def test_net_analysis(self):
        sums, means = net_analysis(np.array([1., 2., 3., 4.]),
                                   np.array([1, 1, 2, 2]))
        np.testing.assert_allclose(sums, [3., 7.])
        np.testing.assert_allclose(means, [1.5, 3.5])

    def test_state_probability_helpers(self):
        group_prob = group_state_prob(self.results)
        self.assertEqual(group_prob.shape, (8, 3))
        self.assertEqual(stack_state_prob(self.results).shape, (8, 3, 4))
        self.assertEqual(network_state_prob(group_prob, self.atlas).shape, (2, 3))
        np.testing.assert_allclose(keep_above(np.array([0.1, 0.9]), 0.5),
                                   [0.0, 0.9])

    def test_state_table_and_coverage(self):
        prob = np.zeros((10, 127))
        prob[:, 0] = 0.5
        prob[:, 1] = 0.2
        table = state_table(prob.mean(axis=0), 0.1)
        self.assertEqual(len(table), 2)
        self.assertEqual(table.iloc[0]["binary_pattern"], "0000001")
        self.assertIn("state_index", table.columns)
        coverage = top_state_coverage(prob, 0.1)
        self.assertEqual(coverage["top_k"], 13)
        self.assertAlmostEqual(coverage["top_coverage"], 1.0)

    def test_group_summary(self):
        summary = group_summary(self.results, self.atlas)
        self.assertEqual(summary["node_temporal"].shape, (8, 4))
        self.assertEqual(summary["node_prob"].shape, (8, 3, 4))
        self.assertEqual(summary["sample_correlation"].shape, (4, 4))
        self.assertEqual(len(summary["net_flexibility"]), 2)
        self.assertEqual(summary["n_samples"], 4)

    def test_icc(self):
        data = np.repeat(np.arange(5., dtype=float)[:, None], 4, axis=1)
        self.assertAlmostEqual(icc_1_1(data), 1.0)
        flex = np.tile(np.arange(8, dtype=float), (6, 1))
        values = node_icc(flex, 3, 2)
        self.assertEqual(values.shape, (8,))

    def test_ttest_and_cohen(self):
        rng = np.random.default_rng(3)
        a = rng.normal(loc=0.5, size=(8, 12))
        b = rng.normal(size=(8, 15))
        stat = two_sample_ttest(a, b)
        self.assertEqual(list(stat.columns), ["t", "p", "cohens_d"])
        self.assertEqual(len(stat), 8)
        self.assertTrue(np.isfinite(stat.to_numpy()).all())
        counts = cohen_network_counts(stat["cohens_d"].to_numpy(),
                                      self.atlas, 0.3)
        self.assertEqual(list(counts["network"]), ["Network 1", "Network 2"])
        self.assertEqual(counts["n_roi"].sum(), 8)

    def test_fdr(self):
        np.testing.assert_array_equal(fdr_bh([0.001, 0.02, 0.5], 0.05),
                                      [0, 1])
        self.assertEqual(len(fdr_bh([0.4, 0.5], 0.05)), 0)


class TestPLS(unittest.TestCase):
    def test_deterministic_and_explained(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 15))
        y = x[:, :4] @ rng.normal(size=(4, 6)) + 0.1 * rng.normal(size=(40, 6))
        first = pls_analysis(x, y)
        second = pls_analysis(x, y)
        np.testing.assert_allclose(first["expl_cov"], second["expl_cov"])
        self.assertAlmostEqual(first["expl_cov"].sum(), 1.0)
        self.assertEqual(first["rsfc_loadings"].shape, (15, 6))
        self.assertEqual(first["behav_loadings"].shape, (6, 6))

    def test_normalise_and_colcorr(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(20, 4))
        np.testing.assert_allclose(normalize(x, 1).mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(normalize(x, 0), x)
        corr = colcorr(x, x)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_bootstrap_stats(self):
        loadings = np.array([[1.0], [2.0]])
        boot = np.ones((2, 1, 20))
        stats = bootstrap_stats(loadings, boot)
        self.assertTrue(np.all(stats["z"] >= 0))
        self.assertEqual(stats["z"].shape, (2, 1))


class TestConfig(unittest.TestCase):
    def test_run_tag(self):
        self.assertEqual(DCCConfig().run_tag(),
                         "swl30_sws1_k0.5_abs_global")
        self.assertEqual(DCCConfig(swl=50, k=1.0, sign_type="raw").run_tag(),
                         "swl50_sws1_k1_raw_global")

    def test_round_trip(self):
        cfg = DCCConfig(swl=42, sign_type="pos", binarize_mode="roi",
                        n_perms=7, variable="ts")
        again = DCCConfig.from_dict(cfg.to_dict())
        self.assertEqual(again.to_dict(), cfg.to_dict())

    def test_validation(self):
        for kwargs in ({"sign_type": "bad"}, {"sws": 0}, {"swl": 1},
                       {"binarize_mode": "oops"}, {"n_perms": -1},
                       {"fdr_q": 1.0}, {"top_state_fraction": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DCCConfig.from_dict(DCCConfig(**kwargs).to_dict())


if __name__ == "__main__":
    unittest.main()
