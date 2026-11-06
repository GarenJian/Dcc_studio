"""Tests for generic input discovery, atlas loading and result files.

The IO layer is where real-world data is most likely to break: a container may
be a 3-D array or a cell array, a directory may contain unrelated text files,
a metadata table may list samples in a different order. These tests write tiny
temporary files in every supported format (``.mat``, ``.npy``, ``.npz``,
``.csv``) and verify that:

* atlas labels and optional network names produce the expected ROI/network
  counts and boundaries;
* 2-D, 3-D and object-array containers are all normalised to float matrices;
* directory discovery ignores unsupported extensions and aligns metadata;
* mismatched ROI counts are rejected with a clear error;
* ``ResultStore`` writes and reloads per-sample caches, CSV, JSON and PNG;
* settings survive a ``save_settings`` / ``load_settings`` round trip.

All temporary files live in ``TemporaryDirectory`` contexts and are removed
even when an assertion fails.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcc.atlas import Atlas
from dcc.complexity import SubjectResult
from dcc.config import DCCConfig, RunPaths, load_settings, save_settings
from dcc.results import ResultStore
from dcc.timeseries import discover_samples, load_timeseries


class TestAtlasIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_labels_and_names(self):
        labels = self.root / "labels.txt"
        labels.write_text("1\n1\n2\n2\n3\n", encoding="utf-8")
        names = self.root / "names.csv"
        names.write_text("net_idx,net_name\n1,Vision\n2,Motor\n3,Limbic\n",
                         encoding="utf-8")
        atlas = Atlas.load(labels, names)
        self.assertEqual(atlas.n_roi, 5)
        self.assertEqual(atlas.n_net, 3)
        self.assertEqual(atlas.name(0), "Vision")
        self.assertEqual(atlas.name(2), "Limbic")
        self.assertEqual(atlas.n_states, 7)

    def test_labels_with_header(self):
        path = self.root / "labels.csv"
        path.write_text("label\n1\n1\n2\n2\n", encoding="utf-8")
        atlas = Atlas.load(path)
        np.testing.assert_array_equal(atlas.roi_idx, [1, 1, 2, 2])
        self.assertEqual(atlas.n_net, 2)

    def test_names_plain_text(self):
        path = self.root / "names.txt"
        path.write_text("A\nB\nC\n", encoding="utf-8")
        atlas = Atlas.from_labels([1, 2, 3], names=path.read_text().splitlines())
        self.assertEqual(atlas.name(1), "B")

    def test_empty_labels(self):
        path = self.root / "empty.txt"
        path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            Atlas.load(path)


class TestTimeSeriesIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.matrix = np.arange(24, dtype=float).reshape(4, 6)

    def tearDown(self):
        self.tmp.cleanup()

    def test_npy_2d_3d_object(self):
        np.save(self.root / "a.npy", self.matrix)
        np.save(self.root / "b.npy",
                np.stack([self.matrix, self.matrix], axis=2))
        cells = np.empty(2, dtype=object)
        cells[0] = self.matrix
        cells[1] = self.matrix
        np.save(self.root / "c.npy", cells)
        self.assertEqual(load_timeseries(self.root / "a.npy")[0][1].shape, (4, 6))
        self.assertEqual(len(load_timeseries(self.root / "b.npy")), 2)
        self.assertEqual(len(load_timeseries(self.root / "c.npy")), 2)

    def test_npz(self):
        np.savez(self.root / "a.npz", data=self.matrix)
        self.assertEqual(load_timeseries(self.root / "a.npz")[0][1].shape, (4, 6))

    def test_csv_transposed(self):
        np.savetxt(self.root / "a.csv", self.matrix.T, delimiter=",")
        self.assertEqual(load_timeseries(self.root / "a.csv")[0][1].shape, (4, 6))

    def test_mat_3d_and_cell(self):
        savemat(self.root / "a.mat", {"ts": np.stack([self.matrix, self.matrix],
                                                     axis=2)})
        self.assertEqual(len(load_timeseries(self.root / "a.mat", "ts")), 2)
        cell = np.empty((2, 1), dtype=object)
        cell[0, 0] = self.matrix
        cell[1, 0] = self.matrix
        savemat(self.root / "b.mat", {"ts": cell})
        self.assertEqual(len(load_timeseries(self.root / "b.mat", "ts")), 2)
        with self.assertRaises(KeyError):
            load_timeseries(self.root / "a.mat", "missing")

    def test_unsupported_format(self):
        path = self.root / "a.bin"
        path.write_bytes(b"x")
        with self.assertRaises(ValueError):
            load_timeseries(path)


class TestSampleDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(5)
        for name in ("s1", "s2", "s3"):
            np.save(self.root / f"{name}.npy",
                    self.rng.normal(size=(6, 40)))
        (self.root / "labels.txt").write_text("1\n1\n2\n2\n3\n3\n",
                                              encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_ignores_other_extensions(self):
        data_dir = self.root / "all"
        data_dir.mkdir()
        for name in ("a", "b"):
            np.save(data_dir / f"{name}.npy", self.rng.normal(size=(6, 40)))
        (data_dir / "notes.md").write_text("ignored", encoding="utf-8")
        samples = discover_samples(data_dir)
        self.assertEqual([s.sample_id for s in samples], ["a", "b"])
        self.assertTrue(all(s.data.shape == (6, 40) for s in samples))

    def test_directory_of_npy_only(self):
        data_dir = self.root / "data"
        data_dir.mkdir()
        for name in ("s1", "s2"):
            np.save(data_dir / f"{name}.npy", self.rng.normal(size=(6, 40)))
        samples = discover_samples(data_dir)
        self.assertEqual([s.sample_id for s in samples], ["s1", "s2"])

    def test_metadata_by_sample_id(self):
        data_dir = self.root / "data"
        data_dir.mkdir()
        for name in ("s1", "s2"):
            np.save(data_dir / f"{name}.npy", self.rng.normal(size=(6, 40)))
        meta = self.root / "meta.csv"
        pd.DataFrame({"sample_id": ["s2", "s1"],
                      "group": ["B", "A"],
                      "subject": ["p2", "p1"],
                      "session": ["2", "1"]}).to_csv(meta, index=False)
        samples = discover_samples(data_dir, metadata_path=meta)
        self.assertEqual(samples[0].group, "A")
        self.assertEqual(samples[1].group, "B")
        self.assertEqual(samples[0].subject, "p1")

    def test_metadata_by_row_order(self):
        data_dir = self.root / "data"
        data_dir.mkdir()
        for name in ("s1", "s2"):
            np.save(data_dir / f"{name}.npy", self.rng.normal(size=(6, 40)))
        meta = self.root / "meta.csv"
        pd.DataFrame({"group": ["A", "B"]}).to_csv(meta, index=False)
        samples = discover_samples(data_dir, metadata_path=meta)
        self.assertEqual([s.group for s in samples], ["A", "B"])

    def test_inconsistent_roi_count(self):
        data_dir = self.root / "bad"
        data_dir.mkdir()
        np.save(data_dir / "s1.npy", self.rng.normal(size=(6, 40)))
        np.save(data_dir / "s2.npy", self.rng.normal(size=(7, 40)))
        with self.assertRaises(ValueError):
            discover_samples(data_dir)

    def test_missing_source(self):
        with self.assertRaises(FileNotFoundError):
            discover_samples(self.root / "missing")


class TestResultStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ResultStore(Path(self.tmp.name) / "run")

    def tearDown(self):
        self.tmp.cleanup()

    def test_sample_cache_round_trip(self):
        result = SubjectResult(entropy=np.arange(4.0),
                               states_prob=np.full((4, 3), 1 / 3),
                               states_counts=np.ones((4, 3), dtype=np.uint16),
                               threshold=0.5, n_windows=10,
                               subject_id="s1")
        self.store.save_sample("s1", result)
        self.assertTrue(self.store.has_sample("s1"))
        loaded = self.store.load_sample("s1")
        np.testing.assert_allclose(loaded.entropy, result.entropy)
        self.assertEqual(loaded.subject_id, "s1")
        self.assertEqual(loaded.n_windows, 10)

    def test_files(self):
        self.store.save_csv("table.csv", pd.DataFrame({"a": [1, 2]}))
        self.store.save_json("info.json", {"x": np.float32(1.5)})
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        self.store.save_figure("plot.png", fig)
        plt.close(fig)
        names = {p.name for p in self.store.list_files()}
        self.assertIn("table.csv", names)
        self.assertIn("info.json", names)
        self.assertIn("plot.png", names)


class TestSettings(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            paths = RunPaths(data_source=path.parent / "data",
                             atlas_labels=path.parent / "labels.txt",
                             output_root=path.parent / "out")
            cfg = DCCConfig(swl=17, k=0.75, variable="ts")
            save_settings(paths, cfg, path)
            again_paths, again_cfg = load_settings(path)
            self.assertEqual(again_cfg.run_tag(), cfg.run_tag())
            self.assertEqual(again_cfg.variable, "ts")
            self.assertEqual(again_paths.data_source, paths.data_source)
            self.assertIsNone(again_paths.metadata)

    def test_missing_config_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, cfg = load_settings(Path(tmp) / "absent.json")
            self.assertEqual(cfg.swl, 30)
            self.assertIsInstance(paths, RunPaths)


if __name__ == "__main__":
    unittest.main()
