"""End-to-end tests of the complete workflow with synthetic data.

A helper builds six samples (three networks, group metadata, a balanced
subject/session layout and a behaviour table) in a temporary directory. The
tests then exercise ``run_analysis`` as a whole:

* a full run produces the expected CSV/JSON/PNG files and all six stage
  timings, with group comparison, ICC(1,1) and PLS-C recorded as analyses;
* re-running the same configuration reuses the per-sample cache;
* cooperative cancellation raises ``AnalysisCancelled`` between samples;
* missing inputs, an overlong window, atlas/sample mismatches and a behaviour
  table with missing rows produce clear errors or warnings;
* two different parameter sets write to separate result directories;
* a core-only run (no metadata/behaviour) still produces the complete summary.

These tests complement the reference-parity suite: here the emphasis is on
control flow, file layout and error handling rather than on published numbers.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from dcc.config import DCCConfig, RunPaths
from dcc.pipelines import AnalysisCancelled, Callbacks, STAGES, run_analysis


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.default_rng(42)
        labels = [1] * 4 + [2] * 4 + [3] * 4
        (self.root / "labels.txt").write_text(
            "\n".join(map(str, labels)) + "\n", encoding="utf-8")
        self.ids = [f"s{i + 1}" for i in range(6)]
        for sid in self.ids:
            np.save(self.data / f"{sid}.npy", rng.normal(size=(12, 60)))
        pd.DataFrame({"sample_id": self.ids,
                      "group": ["A", "A", "A", "B", "B", "B"],
                      "subject": ["p1", "p1", "p2", "p2", "p3", "p3"],
                      "session": ["1", "2", "1", "2", "1", "2"]}).to_csv(
                          self.root / "metadata.csv", index=False)
        pd.DataFrame({"sample_id": self.ids,
                      "score1": rng.normal(size=6),
                      "score2": rng.normal(size=6),
                      "score3": rng.normal(size=6)}).to_csv(
                          self.root / "behaviour.csv", index=False)
        self.paths = RunPaths(data_source=self.data,
                              atlas_labels=self.root / "labels.txt",
                              metadata=self.root / "metadata.csv",
                              behaviour=self.root / "behaviour.csv",
                              output_root=self.root / "results")
        self.cfg = DCCConfig(swl=10, sws=2, k=0.5,
                             state_prob_threshold=0.01, top_state_fraction=0.2,
                             n_perms=3, n_bootstraps=3, pls_n_lc=2,
                             random_seed=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run(self):
        logs, stages, progress = [], [], []
        cb = Callbacks(log=logs.append,
                       progress=lambda s, d, t: progress.append((s, d, t)),
                       stage=lambda s, i, t: stages.append((s, i, t)))
        result = run_analysis(self.paths, self.cfg, cb)

        self.assertEqual([entry[0] for entry in stages], list(STAGES))
        self.assertEqual([entry[1] for entry in stages],
                         list(range(1, len(STAGES) + 1)))
        self.assertTrue(progress)
        self.assertTrue(any("DCC finished" in line for line in logs))
        self.assertTrue(result.run_dir.exists())

        info = json.loads((result.run_dir / "summary.json").read_text(
            encoding="utf-8"))
        self.assertEqual(info["n_samples"], 6)
        self.assertEqual(info["n_roi"], 12)
        self.assertEqual(info["n_networks"], 3)
        self.assertEqual(info["n_states"], 7)
        self.assertEqual(info["computed"], 6)
        self.assertIn("group comparison", info["analyses"])
        self.assertIn("ICC(1,1)", info["analyses"])
        self.assertIn("PLS-C", info["analyses"])
        self.assertEqual(info["warnings"], [])
        self.assertEqual(len(info["stages"]), len(STAGES))

        names = {p.name for p in result.run_dir.rglob("*") if p.is_file()}
        for expected in ("dcc_matrix.csv", "dcc_per_sample.csv",
                         "group_summary.csv", "network_summary.csv",
                         "state_probability.csv", "network_state_probability.csv",
                         "high_frequency_states.csv", "top_state_coverage.json",
                         "sample_similarity.csv", "icc.csv",
                         "pls_explained_covariance.csv",
                         "run_manifest.json", "summary.json"):
            self.assertIn(expected, names)
        self.assertTrue(any(n.startswith("comparison_") for n in names))
        self.assertGreaterEqual(len(result.figures), 8)
        self.assertTrue(all(p.exists() for p in result.files))

        long_table = pd.read_csv(result.run_dir / "dcc_per_sample.csv")
        self.assertEqual(len(long_table), 6 * 12)
        self.assertTrue(long_table["dcc"].between(0, np.log2(7) + 1e-9).all())
        matrix = pd.read_csv(result.run_dir / "dcc_matrix.csv")
        self.assertEqual(matrix.shape, (12, 8))

    def test_second_run_uses_cache(self):
        first = run_analysis(self.paths, self.cfg)
        second = run_analysis(self.paths, self.cfg)
        self.assertEqual(first.summary["computed"], 6)
        self.assertEqual(second.summary["computed"], 0)
        self.assertEqual(second.summary["cached"], 6)
        self.assertEqual(first.run_dir, second.run_dir)

    def test_cancellation(self):
        with self.assertRaises(AnalysisCancelled):
            run_analysis(self.paths, self.cfg,
                         Callbacks(cancelled=lambda: True))

    def test_missing_inputs(self):
        paths = RunPaths(data_source=self.root / "missing",
                         atlas_labels=self.root / "labels.txt",
                         output_root=self.root / "results")
        with self.assertRaises(FileNotFoundError):
            run_analysis(paths, self.cfg)

    def test_window_too_long(self):
        with self.assertRaises(ValueError):
            run_analysis(self.paths, DCCConfig(swl=1000))

    def test_atlas_mismatch(self):
        wrong = self.root / "wrong_labels.txt"
        wrong.write_text("1\n2\n", encoding="utf-8")
        paths = RunPaths(data_source=self.data, atlas_labels=wrong,
                         output_root=self.root / "results")
        with self.assertRaises(ValueError):
            run_analysis(paths, self.cfg)

    def test_parameters_create_separate_runs(self):
        first = run_analysis(self.paths, self.cfg)
        other = DCCConfig.from_dict({**self.cfg.to_dict(), "swl": 12})
        second = run_analysis(self.paths, other)
        self.assertNotEqual(first.run_dir, second.run_dir)
        self.assertTrue(first.run_dir.exists() and second.run_dir.exists())

    def test_core_only_without_optional_metadata(self):
        paths = RunPaths(data_source=self.data,
                         atlas_labels=self.root / "labels.txt",
                         output_root=self.root / "plain")
        cfg = DCCConfig(swl=10, n_perms=0, n_bootstraps=0)
        result = run_analysis(paths, cfg)
        self.assertEqual(result.summary["analyses"], [])
        self.assertTrue((result.run_dir / "dcc_matrix.csv").exists())
        self.assertFalse((result.run_dir / "icc.csv").exists())

    def test_behaviour_row_mismatch_is_a_warning(self):
        wrong = self.root / "short_behaviour.csv"
        pd.DataFrame({"sample_id": ["s1", "s2"],
                      "score": [1.0, 2.0]}).to_csv(wrong, index=False)
        paths = RunPaths(data_source=self.data,
                         atlas_labels=self.root / "labels.txt",
                         metadata=self.root / "metadata.csv",
                         behaviour=wrong,
                         output_root=self.root / "mismatch")
        result = run_analysis(paths, DCCConfig(swl=10, n_perms=0,
                                               n_bootstraps=0))
        self.assertTrue(any("PLS skipped" in w
                            for w in result.summary["warnings"]))
        self.assertTrue((result.run_dir / "dcc_matrix.csv").exists())


if __name__ == "__main__":
    unittest.main()
