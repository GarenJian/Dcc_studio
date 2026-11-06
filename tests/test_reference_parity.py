"""Functional regression tests built directly on the copied reference files.

The files under ``tests/fixtures/`` are verbatim copies of the input data and
published results from the reference study folder. Every test opens a real
file, uses it to construct a case, and compares what the current implementation
computes against the stored result.

Reporting
---------
These are the *functional* cases of the suite, so each numerical comparison is
also recorded through ``tests/functional_report.py``. The one-click runner
turns those records into the detailed first chapter of
``reports/test-report.md``: which dataset was used, what the reference result
was, what the current code produced, the largest deviation and the tolerance.
Programmatic unit tests are intentionally reported as a summary only.

Layout (mirrors how the reference folder groups inputs and results)
-------------------------------------------------------------------
``Brain_mask/``            ROI labels and network names
``MSC_process/TS/``        200 x 818 x 90 real time series
``MSC_process/results/``   per-sample DCC results and group statistics
``UCLA_process/TS/``       cell array of real time series
``UCLA_process/results/``  per-sample DCC results and PLS outputs
``TMS_process/``           pre/post group matrices and published t tests

Coverage map
------------
=====================================  ======================================
Test                                   Verified computation
=====================================  ======================================
``test_copied_reference_files``        All 13 files present and correctly shaped
``test_atlas``                         200 ROIs, 7 networks, 127 states
``test_single_sample_dcc``             dFC, thresholding, state coding, entropy
``test_group_statistics``              ROI x sample and state aggregation
``test_group_comparison``              t statistic, p value, Cohen's d
``test_pls``                           SVD components, covariance, loadings
``test_ucla_dcc_matrix``               DCC matrix used as PLS imaging input
``test_ucla_results``                  All 224 stored results match the PLS input
``test_workflow_with_real_data``       Full six-stage workflow and CSV export
=====================================  ======================================

Large ``.mat`` containers are loaded once in ``setUpClass`` and shared by all
cases, so a full suite still runs in seconds.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from dcc.atlas import Atlas
from dcc.complexity import connectome_complex
from dcc.config import DCCConfig, RunPaths
from dcc.pipelines import run_analysis
from dcc.pls import pls_analysis
from dcc.stats import group_summary, two_sample_ttest

try:                                    # package import (unittest discovery)
    from . import functional_report as fr
except ImportError:                     # direct script execution
    import functional_report as fr     # type: ignore[no-redef]

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Paths of the copied reference files. Keeping them in one table makes the
# provenance check below explicit and easy to extend.
ATLAS_LABELS = FIXTURES / "Brain_mask" / "ROI_idx.txt"
ATLAS_NAMES = FIXTURES / "Brain_mask" / "7Net_200.csv"
MSC_TS = FIXTURES / "MSC_process" / "TS" / "TS_data.mat"
MSC_RESULTS = FIXTURES / "MSC_process" / "results" / "node_flexibility.mat"
MSC_GROUP = FIXTURES / "MSC_process" / "results" / "group_statis.mat"
UCLA_TS = FIXTURES / "UCLA_process" / "TS" / "Ts_data.mat"
UCLA_RESULTS = FIXTURES / "UCLA_process" / "results" / "node_flexibility.mat"
PLS_RESULTS = FIXTURES / "UCLA_process" / "results" / "PLSresults.mat"
TMS_DIR = FIXTURES / "TMS_process"

REQUIRED_FILES = (
    ATLAS_LABELS, ATLAS_NAMES, MSC_TS, MSC_RESULTS, MSC_GROUP,
    UCLA_TS, UCLA_RESULTS, PLS_RESULTS,
    TMS_DIR / "Pre_group_statis.mat",
    TMS_DIR / "CTR_group_statis.mat",
    TMS_DIR / "FRO_group_statis.mat",
    TMS_DIR / "OCC_group_statis.mat",
    TMS_DIR / "Post_vs_Pre_res.mat",
)

# Human-readable dataset descriptions reused by the functional report.
ATLAS_DATASET = "Brain_mask/ROI_idx.txt + 7Net_200.csv (200 ROIs, 7 networks)"
MSC_DATASET = ("MSC_process/TS/TS_data.mat - 200 ROIs x 818 time points, "
               "90 samples, SWL=30, SWS=1, k=0.5, abs, global")
MSC_RESULT_SOURCE = "MSC_process/results/node_flexibility.mat"
GROUP_DATASET = "MSC_process/results/group_statis.mat (two-sample slice)"
TMS_DATASET = ("TMS_process/*_group_statis.mat - Pre (39) vs CTR/FRO/OCC "
               "post groups, SWL=50")
PLS_DATASET = ("UCLA_process/results/PLSresults.mat - X (224x200), "
               "Y (224x52), SVD components")
UCLA_DATASET = "UCLA_process/TS/Ts_data.mat - 224 real sample matrices"


def _mat(path: Path):
    """Load one copied ``.mat`` container without squeezing cell dimensions."""
    return loadmat(str(path), squeeze_me=False, struct_as_record=False)


def _compare(case: str, dataset: str, source: str, quantity: str,
             expected, actual, atol: float, note: str = "") -> None:
    """Assert equality within ``atol`` and record the values for the report.

    ``expected`` is always the stored reference result and ``actual`` is what
    the current implementation computed; keeping that order consistent makes
    the generated comparison tables unambiguous. ``atol=0`` means an exact
    element-by-element match.
    """
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
    fr.record(case, dataset, source, quantity, expected, actual, atol, note)


class TestReferenceParity(unittest.TestCase):
    """Compare the current implementation with the copied reference results."""

    @classmethod
    def setUpClass(cls):
        """Load every reference file once and derive the shared run settings."""
        fr.reset()
        # Reference parameters: SWL=30, step 1, k=0.5, absolute correlations,
        # global threshold. Changing any of them invalidates the comparisons.
        cls.cfg = DCCConfig(swl=30, sws=1, k=0.5, sign_type="abs",
                            binarize_mode="global")
        cls.atlas = Atlas.load(ATLAS_LABELS, ATLAS_NAMES)

        cls.ts = _mat(MSC_TS)["Ts_data"]                    # (200, 818, 90)
        cls.cells = _mat(MSC_RESULTS)["result"]              # (90, 4) cells
        cls.group = _mat(MSC_GROUP)
        cls.ucla_ts = _mat(UCLA_TS)["Ts_data"]               # (224, 1) cells
        cls.ucla_results = _mat(UCLA_RESULTS)["result"]      # (224, 4) cells
        cls.pls = _mat(PLS_RESULTS)
        cls.pre = _mat(TMS_DIR / "Pre_group_statis.mat")["node_temporal"]
        cls.posts = {
            short: _mat(TMS_DIR / f"{short}_group_statis.mat")["node_temporal"]
            for short in ("CTR", "FRO", "OCC")
        }
        cls.tms_ref = _mat(TMS_DIR / "Post_vs_Pre_res.mat")

    @classmethod
    def tearDownClass(cls):
        """Persist all recorded functional comparisons for the report."""
        fr.flush()

    def test_copied_reference_files_are_complete(self):
        """Every copied file exists, is non-empty and has the expected shape."""
        case = "test_copied_reference_files_are_complete"
        missing = [str(path) for path in REQUIRED_FILES
                   if not path.exists() or path.stat().st_size == 0]
        self.assertEqual(missing, [])
        _compare(case, "Reference file inventory", "tests/fixtures",
                 "number of copied reference files", [13],
                 [len(REQUIRED_FILES)], 0, "verbatim copies, no generation")
        _compare(case, MSC_DATASET, "MSC_process/TS/TS_data.mat",
                 "time-series tensor shape", [200, 818, 90],
                 list(self.ts.shape), 0)
        _compare(case, MSC_RESULT_SOURCE,
                 "MSC_process/results/node_flexibility.mat",
                 "per-sample result cell shape", [90, 4],
                 list(self.cells.shape), 0)
        _compare(case, PLS_DATASET, "UCLA_process/results/PLSresults.mat",
                 "PLS imaging matrix shape", [224, 200],
                 list(self.pls["X"].shape), 0)
        self.assertEqual(self.pls["Y"].shape[0], 224)

    def test_atlas_matches_reference(self):
        """The copied 200-ROI / 7-network atlas defines 2**7 - 1 states."""
        _compare("test_atlas_matches_reference", ATLAS_DATASET,
                 "Brain_mask/ROI_idx.txt + 7Net_200.csv",
                 "atlas dimensions (ROIs, networks, states)",
                 [200, 7, 127],
                 [self.atlas.n_roi, self.atlas.n_net, self.atlas.n_states],
                 0, "states = 2**networks - 1")

    def test_single_sample_dcc(self):
        """Entropy, probabilities, counts and binary dFC match exactly.

        ``result[0, 3]`` compares all 200 x 7 x 789 threshold decisions, which
        would expose any change in sign handling, correlation normalisation or
        the ``mean + k*std`` threshold.
        """
        case = "test_single_sample_dcc"
        for index in (0, 1):
            dataset = f"{MSC_DATASET} - sample {index + 1}"
            result = connectome_complex(
                self.ts[:, :, index], self.atlas, self.cfg,
                swl=self.cfg.swl, subject_id=f"sample_{index + 1}")
            _compare(case, dataset, MSC_RESULT_SOURCE,
                     f"sample {index + 1}: ROI entropy (bits)",
                     np.asarray(self.cells[index, 0]).ravel(), result.entropy,
                     1e-5)
            _compare(case, dataset, MSC_RESULT_SOURCE,
                     f"sample {index + 1}: state probabilities (200 x 127)",
                     np.asarray(self.cells[index, 1]), result.states_prob,
                     1e-6)
            _compare(case, dataset, MSC_RESULT_SOURCE,
                     f"sample {index + 1}: state occurrence counts",
                     np.asarray(self.cells[index, 2]), result.states_counts,
                     0, "exact integer equality")
            _compare(case, dataset, MSC_RESULT_SOURCE,
                     f"sample {index + 1}: sliding-window count",
                     [789], [result.n_windows], 0)
        first = connectome_complex(self.ts[:, :, 0], self.atlas, self.cfg,
                                   swl=self.cfg.swl)
        _compare(case, f"{MSC_DATASET} - sample 1", MSC_RESULT_SOURCE,
                 "binary dFC decisions (200 x 7 x 789)",
                 np.asarray(self.cells[0, 3]), first.dfc_binary, 0,
                 "every threshold decision compared")
        _compare(case, f"{MSC_DATASET} - sample 1", MSC_RESULT_SOURCE,
                 "binarisation threshold", [0.4833296374],
                 [first.threshold], 1e-6, "mean + 0.5 * std, ddof=1")

    def test_group_statistics(self):
        """Two-sample aggregation reproduces the published matrices.

        ``group_temporal_flexibility`` in the reference file is the mean over
        all 90 samples, so only its shape and finiteness are checked here; the
        two-sample quantities are compared element-by-element.
        """
        case = "test_group_statistics"
        results = [connectome_complex(self.ts[:, :, index], self.atlas,
                                      self.cfg, swl=self.cfg.swl)
                   for index in (0, 1)]
        node_temporal = np.column_stack([r.entropy for r in results])
        node_prob = np.stack([r.states_prob for r in results], axis=2)
        _compare(case, GROUP_DATASET, "MSC_process/results/group_statis.mat",
                 "ROI x sample DCC matrix (200 x 2)",
                 np.asarray(self.group["node_temporal"])[:, :2],
                 node_temporal, 1e-5)
        _compare(case, GROUP_DATASET, "MSC_process/results/group_statis.mat",
                 "ROI x state x sample probabilities (200 x 127 x 2)",
                 np.asarray(self.group["node_prob"])[:, :, :2],
                 node_prob, 1e-6)
        full_mean = np.asarray(
            self.group["group_temporal_flexibility"]).ravel()
        self.assertEqual(full_mean.shape, (200,))
        self.assertTrue(np.isfinite(full_mean).all())
        _compare(case, GROUP_DATASET, "MSC_process/results/group_statis.mat",
                 "sample count used by group_summary",
                 [2], [group_summary(results, self.atlas)["n_samples"]], 0)

    def test_group_comparison_statistics(self):
        """Recomputed t / p / Cohen's d match the published post-vs-pre tests.

        The three post groups have different sample counts while the common Pre
        group has 39, so this also exercises unequal group sizes and the
        pooled-variance formula.
        """
        case = "test_group_comparison_statistics"
        for short in ("CTR", "FRO", "OCC"):
            post = self.posts[short]
            dataset = (f"{TMS_DATASET} - {short} (n={post.shape[1]}) "
                       f"vs Pre (n={self.pre.shape[1]})")
            stat = two_sample_ttest(post, self.pre)
            source = "TMS_process/Post_vs_Pre_res.mat"
            _compare(case, dataset, source,
                     f"{short}: t / p / Cohen's d (200 x 3)",
                     np.asarray(self.tms_ref[f"stat_{short}_Pre"]),
                     stat.to_numpy(), 1e-8)
            _compare(case, dataset, source,
                     f"{short}: published Cohen's d (200 x 1)",
                     np.asarray(self.tms_ref[f"{short}_Pre_cohen"]),
                     stat["cohens_d"].to_numpy().reshape(-1, 1), 1e-8)

    def test_pls_main_analysis(self):
        """SVD components, explained covariance and loadings are reproduced.

        Signs of singular vectors are arbitrary, hence the absolute-value
        comparison. ``explCovLC`` is a variance fraction and is compared as is.
        """
        case = "test_pls_main_analysis"
        source = "UCLA_process/results/PLSresults.mat"
        result = pls_analysis(self.pls["X"], self.pls["Y"])
        _compare(case, PLS_DATASET, source,
                 "explained covariance per latent component",
                 np.asarray(self.pls["explCovLC"]).ravel(),
                 result["expl_cov"], 1e-10)
        for key in ("U", "V", "Lx", "Ly"):
            _compare(case, PLS_DATASET, source,
                     f"SVD component {key}", np.abs(self.pls[key]),
                     np.abs(result[key]), 1e-8, "sign-invariant (absolute values)")
        _compare(case, PLS_DATASET, source, "ROI loadings",
                 np.abs(self.pls["LC_RSFC_loadings"]),
                 np.abs(result["rsfc_loadings"]), 1e-8,
                 "sign-invariant (absolute values)")
        _compare(case, PLS_DATASET, source, "behaviour loadings",
                 np.abs(self.pls["LC_behav_loadings"]),
                 np.abs(result["behav_loadings"]), 1e-8,
                 "sign-invariant (absolute values)")

    def test_ucla_dcc_matrix(self):
        """The DCC matrix used as PLS imaging input is reproduced exactly."""
        values = np.column_stack([
            connectome_complex(np.asarray(self.ucla_ts[index, 0]), self.atlas,
                               self.cfg, swl=self.cfg.swl).entropy
            for index in range(4)]).T
        _compare("test_ucla_dcc_matrix", UCLA_DATASET,
                 "UCLA_process/results/PLSresults.mat",
                 "recomputed DCC matrix, first 4 samples (4 x 200)",
                 np.asarray(self.pls["X"])[:4], values, 1e-5)

    def test_ucla_results_match_pls_input(self):
        """All 224 stored per-sample results agree with the PLS imaging matrix.

        This cross-checks two independent result files from the reference run,
        so a corrupted or mismatched copy is detected even before any
        computation is compared.
        """
        stored = np.vstack([
            np.asarray(self.ucla_results[index, 0]).ravel()
            for index in range(self.ucla_results.shape[0])])
        _compare("test_ucla_results_match_pls_input",
                 "UCLA_process/results/node_flexibility.mat - all 224 samples",
                 "UCLA_process/results/PLSresults.mat",
                 "stored DCC profiles vs PLS imaging matrix (224 x 200)",
                 np.asarray(self.pls["X"]), stored, 1e-8,
                 "cross-check of two reference result files")

    def test_workflow_with_real_data(self):
        """The full six-stage workflow reproduces the reference DCC matrix.

        This is the integration counterpart of the arithmetic tests above: it
        also verifies sample discovery, per-sample caching, CSV export and the
        metadata written to ``summary.json`` on real-shaped input data.
        """
        case = "test_workflow_with_real_data"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            for index in (0, 1):
                np.save(data / f"sample_{index + 1:03d}.npy",
                        self.ts[:, :, index])
            paths = RunPaths(data_source=data, atlas_labels=ATLAS_LABELS,
                             atlas_names=ATLAS_NAMES,
                             output_root=root / "results")
            cfg = DCCConfig(swl=30, n_perms=0, n_bootstraps=0)
            result = run_analysis(paths, cfg)
            info = result.summary
            self.assertEqual(info["method"], "dcc")
            self.assertEqual(info["warnings"], [])
            _compare(case, f"{MSC_DATASET} - first 2 samples",
                     "MSC_process/results/group_statis.mat",
                     "workflow dcc_matrix.csv (200 x 2)",
                     np.asarray(self.group["node_temporal"])[:, :2],
                     np.loadtxt(result.run_dir / "dcc_matrix.csv",
                                delimiter=",", skiprows=1, usecols=(2, 3)),
                     1e-5, "complete six-stage workflow output")
            _compare(case, f"{MSC_DATASET} - first 2 samples",
                     "dcc/pipelines.py summary contract",
                     "workflow summary (samples, ROIs, networks, states, windows, stages)",
                     [2, 200, 7, 127, 789, 6],
                     [info["n_samples"], info["n_roi"], info["n_networks"],
                      info["n_states"], info["n_windows"], len(info["stages"])],
                     0)


if __name__ == "__main__":
    unittest.main()
