"""Tests for the Settings dialog on page 1.

The dialog is the single entry point for the two global choices: the analysis
method and the light/dark appearance. These tests verify that

* both "Settings" buttons open the same non-modal dialog;
* every registered method is selectable, while roadmap entries such as the
  planned "standard dynamic functional connectivity" are visible but disabled;
* choosing a method updates the in-memory configuration and the read-only
  Method field on page 2;
* shipping a planned method (registering that name) automatically removes its
  disabled placeholder, which is the extension path for future methods;
* the theme switch stays in sync with the main window.

A throw-away ``AliasMethod`` is registered before the window is constructed so
that method switching can be exercised with real registry data. Registry state
is global, so every test snapshots and restores it. PyQt6 is optional, hence
the module-level skip.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    from dcc.methods import (DCCMethod, clear_registry, register_method)
    from dcc.methods.registry import _REGISTRY
    from gui.app import MainWindow
    from gui.settings import PLANNED_METHODS, SettingsDialog
    from gui.theme import DARK, LIGHT, apply_theme
    HAS_QT = True
except ImportError:                     # GUI dependency is optional for CLI use
    HAS_QT = False


class AliasMethod(DCCMethod):
    """Second selectable method, registered only inside these tests."""

    name = "alias"
    title = "Alias DCC"
    description = "Test-only second method used to verify method switching."
    metric = "ALIAS"


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the Settings dialog tests")
class TestSettingsDialog(unittest.TestCase):
    """Build a real window with a second registered method and drive its dialog."""

    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        self.app = QApplication.instance() or QApplication([])
        self.addCleanup(lambda: apply_theme(self.app, DARK))
        self._saved = dict(_REGISTRY)
        register_method(AliasMethod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        store = QSettings(str(Path(self.tmp.name) / "ui.ini"),
                          QSettings.Format.IniFormat)
        self.window = MainWindow(ui_settings=store)

    def tearDown(self):
        clear_registry()
        _REGISTRY.update(self._saved)

    def test_both_settings_buttons_open_the_same_dialog(self):
        self.window.show()
        dialog = self.window.settings_dialog
        self.assertFalse(dialog.isModal())
        self.window.btn_page_settings.click()
        self.assertFalse(dialog.isHidden())
        self.assertIs(self.window.btn_settings.window(), self.window)
        dialog.hide()
        self.window.btn_settings.click()
        self.assertFalse(dialog.isHidden())
        dialog.hide()
        self.window.close()

    def test_dialog_is_seeded_from_the_window_state(self):
        dialog = self.window.settings_dialog
        self.assertEqual(dialog.current_method(), self.window.cfg.method)
        self.assertEqual(dialog.current_theme(), self.window.theme)
        self.assertIn("Method:", self.window.lbl_settings.text())
        self.assertIn("Appearance:", self.window.lbl_settings.text())

    def test_registered_methods_are_enabled_and_roadmap_entries_are_disabled(self):
        dialog = self.window.settings_dialog
        names = dialog.method_names()
        self.assertIn("dcc", names)
        self.assertIn("alias", names)
        self.assertTrue(dialog.is_method_enabled("dcc"))
        self.assertTrue(dialog.is_method_enabled("alias"))
        for name, _title, _description in PLANNED_METHODS:
            self.assertIn(name, names)
            self.assertFalse(dialog.is_method_enabled(name))
        labels = [dialog.method_combo.itemText(i)
                  for i in range(dialog.method_combo.count())]
        self.assertTrue(any("not yet supported" in text for text in labels))

    def test_selecting_a_method_updates_configuration_and_summary(self):
        dialog = self.window.settings_dialog
        index = dialog.method_combo.findData("alias")
        self.assertGreaterEqual(index, 0)
        dialog.method_combo.setCurrentIndex(index)
        self.assertEqual(self.window.cfg.method, "alias")
        self.assertEqual(self.window.param_widgets["method"].currentData(),
                         "alias")
        self.assertIn("Alias DCC", self.window.lbl_settings.text())

    def test_page_two_method_field_is_read_only(self):
        self.assertFalse(self.window.param_widgets["method"].isEnabled())

    def test_shipping_a_planned_method_removes_its_placeholder(self):
        class StandardMethod(DCCMethod):
            name = "standard_dfc"
            title = "Standard dFC"
            description = "A future implementation of classic dynamic FC."

        register_method(StandardMethod)
        dialog = SettingsDialog(method="dcc", theme=DARK)
        self.assertEqual(dialog.method_names().count("standard_dfc"), 1)
        self.assertTrue(dialog.is_method_enabled("standard_dfc"))
        labels = [dialog.method_combo.itemText(i)
                  for i in range(dialog.method_combo.count())]
        self.assertFalse(any("not yet supported" in text for text in labels))

    def test_switch_button_updates_theme_and_dialog_together(self):
        dialog = self.window.settings_dialog
        dialog.btn_switch_theme.click()
        self.assertEqual(self.window.theme, LIGHT)
        self.assertEqual(dialog.current_theme(), LIGHT)
        self.assertIn("dark", dialog.btn_switch_theme.text().lower())


if __name__ == "__main__":
    unittest.main()
