"""Tests for the pluggable analysis-method architecture.

The point of ``dcc.methods`` is that adding a second dynamic connectivity
measure must not require changes to the workflow, GUI, exporter or statistics.
These tests prove that claim by registering a throw-away method at runtime and
running the *real* six-stage workflow through it.

``AliasMethod`` deliberately reuses the built-in DCC computation: the goal is
to verify dispatch, naming and cache separation, not to introduce a second
algorithm into the test suite. Because it shares the numerical core, the run
stays fast and its outputs remain well defined.

Registry state is global, so every test snapshots ``_REGISTRY`` in ``setUp``
and restores it in ``tearDown``. That prevents a test-order dependency from
leaking a fake method into the GUI or the other suites.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dcc.config import DCCConfig, RunPaths
from dcc.methods import (AnalysisMethod, DCCMethod, available_methods,
                         clear_registry, get_method, method_choices,
                         method_descriptions, register_method, registered_names)
from dcc.methods.registry import _REGISTRY
from dcc.pipelines import run_analysis


class AliasMethod(DCCMethod):
    """Test-only method: identical arithmetic, different registered identity."""

    name = "alias"
    title = "Alias DCC"
    description = "Test-only registered method used to verify dispatch."
    metric = "ALIAS"


class TestRegistry(unittest.TestCase):
    """Registration, lookup, validation and UI-facing discovery helpers."""

    def setUp(self):
        self._saved = dict(_REGISTRY)

    def tearDown(self):
        clear_registry()
        _REGISTRY.update(self._saved)

    def test_builtin_dcc_is_registered_and_is_the_default(self):
        self.assertIn("dcc", registered_names())
        self.assertIsInstance(get_method("dcc"), DCCMethod)
        self.assertEqual(get_method(None).name, "dcc")
        self.assertEqual(get_method("").name, "dcc")
        self.assertEqual(available_methods()[0].name, "dcc")

    def test_unknown_name_lists_available_choices(self):
        with self.assertRaisesRegex(ValueError, "Available"):
            get_method("not-registered")

    def test_registering_a_new_method_updates_all_discovery_helpers(self):
        register_method(AliasMethod)
        self.assertEqual(get_method("alias").title, "Alias DCC")
        self.assertIn(("alias", "Alias DCC"), method_choices())
        self.assertIn("alias", method_descriptions())
        self.assertEqual(len(available_methods()), len(self._saved) + 1)

    def test_registration_rejects_unsafe_names_and_conflicts(self):
        class BadName(AnalysisMethod):
            name = "bad name"

            def compute(self, *args, **kwargs):
                raise NotImplementedError

        class Conflict(AnalysisMethod):
            name = "dcc"

            def compute(self, *args, **kwargs):
                raise NotImplementedError

        with self.assertRaises(ValueError):
            register_method(BadName)
        with self.assertRaises(ValueError):
            register_method(Conflict)
        self.assertEqual(set(registered_names()), set(self._saved))

    def test_registering_the_same_class_twice_is_idempotent(self):
        register_method(AliasMethod)
        register_method(AliasMethod)
        self.assertEqual(len(available_methods()), len(self._saved) + 1)


class TestMethodDrivenWorkflow(unittest.TestCase):
    """A registered method changes naming and metadata without code changes."""

    def setUp(self):
        self._saved = dict(_REGISTRY)
        register_method(AliasMethod)

    def tearDown(self):
        clear_registry()
        _REGISTRY.update(self._saved)

    def test_run_tag_separates_method_caches(self):
        default = DCCConfig()
        alternate = DCCConfig(method="alias")
        self.assertFalse(default.run_tag().startswith("alias"))
        self.assertTrue(alternate.run_tag().startswith("alias_"))
        self.assertNotEqual(default.run_tag(), alternate.run_tag())

    def test_workflow_dispatches_to_the_selected_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            rng = np.random.default_rng(7)
            for name in ("s1", "s2"):
                np.save(data / f"{name}.npy", rng.normal(size=(12, 60)))
            labels = root / "labels.txt"
            labels.write_text("\n".join(["1"] * 4 + ["2"] * 4 + ["3"] * 4) + "\n",
                              encoding="utf-8")
            paths = RunPaths(data_source=data, atlas_labels=labels,
                             output_root=root / "results")
            cfg = DCCConfig(swl=10, n_perms=0, n_bootstraps=0, method="alias")
            result = run_analysis(paths, cfg)

            self.assertEqual(result.summary["method"], "alias")
            self.assertEqual(result.summary["method_title"], "Alias DCC")
            self.assertEqual(result.summary["n_samples"], 2)
            self.assertTrue((result.run_dir / "alias_matrix.csv").exists())
            self.assertTrue((result.run_dir / "alias_per_sample.csv").exists())
            self.assertFalse((result.run_dir / "dcc_matrix.csv").exists())
            self.assertEqual(len(result.summary["stages"]), 6)


if __name__ == "__main__":
    unittest.main()
