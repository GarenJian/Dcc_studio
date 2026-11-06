"""Tests for the light/dark appearance switch and parameter hint markers.

The pure-logic cases need no window: they check theme-name normalisation,
toggle round trips, that both rendered style sheets contain every required
selector and no unresolved template placeholder, that each Fusion palette uses
its theme's colours, that the preference survives a ``QSettings`` round trip,
and that button labels/tooltips describe the correct theme.

The GUI cases create a ``QApplication`` on an offscreen platform, construct the
real ``MainWindow``, and verify that changing the theme in the Settings dialog
changes the application style sheet and palette and persists the new
preference. A separate class checks that every parameter has a small, flat,
clickable ``!`` marker and that clicking one actually makes its tooltip visible.

PyQt6 is optional for CLI-only installations, so the whole module is skipped
when Qt is unavailable. Any global appearance change is reverted in cleanup.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from html import escape
from pathlib import Path

try:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QToolTip
    from gui.theme import (DARK, LIGHT, DEFAULT_THEME, THEMES, apply_theme,
                           build_palette, load_theme, normalize_theme, save_theme,
                           stylesheet, theme_button_text, theme_button_tooltip,
                           toggle_theme)
    HAS_QT = True
except ImportError:                     # GUI dependency is optional for CLI use
    HAS_QT = False


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the appearance tests")
class TestTheme(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_theme("dark"), DARK)
        self.assertEqual(normalize_theme(" LIGHT "), LIGHT)
        self.assertEqual(normalize_theme(None), DEFAULT_THEME)
        self.assertEqual(normalize_theme("solarized"), DEFAULT_THEME)
        self.assertEqual(normalize_theme(123), DEFAULT_THEME)
        self.assertEqual(THEMES, (DARK, LIGHT))

    def test_toggle_round_trip(self):
        self.assertEqual(toggle_theme(DARK), LIGHT)
        self.assertEqual(toggle_theme(LIGHT), DARK)
        self.assertEqual(toggle_theme("unknown"), LIGHT)
        self.assertEqual(toggle_theme(toggle_theme(DARK)), DARK)

    def test_stylesheets_are_complete_and_distinct(self):
        dark, light = stylesheet(DARK), stylesheet(LIGHT)
        self.assertNotEqual(dark, light)
        self.assertIn("#151b23", dark)
        self.assertIn("#f3f6fa", light)
        for css in (dark, light):
            for selector in ("QMainWindow", "QPushButton#stepActive",
                             "QPushButton#settings", "QToolTip",
                             "QProgressBar::chunk", "QPushButton#paramHint"):
                self.assertIn(selector, css)
            self.assertNotIn("$", css)  # every placeholder was substituted

    def test_palette_follows_theme(self):
        dark = build_palette(DARK)
        light = build_palette(LIGHT)
        self.assertEqual(dark.color(QPalette.ColorRole.Window).name(), "#151b23")
        self.assertEqual(light.color(QPalette.ColorRole.Window).name(), "#f3f6fa")
        self.assertEqual(dark.color(QPalette.ColorRole.Base).name(), "#111821")
        self.assertEqual(light.color(QPalette.ColorRole.Base).name(), "#ffffff")

    def test_preference_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.ini"
            store = QSettings(str(path), QSettings.Format.IniFormat)
            self.assertEqual(load_theme(store), DEFAULT_THEME)
            self.assertEqual(save_theme(LIGHT, store), LIGHT)
            self.assertEqual(load_theme(store), LIGHT)
            self.assertTrue(path.exists())
            reloaded = QSettings(str(path), QSettings.Format.IniFormat)
            self.assertEqual(load_theme(reloaded), LIGHT)
            self.assertEqual(save_theme("invalid", store), DEFAULT_THEME)
            self.assertEqual(load_theme(store), DEFAULT_THEME)

    def test_button_labels(self):
        self.assertIn("light", theme_button_text(DARK))
        self.assertIn("dark", theme_button_text(LIGHT))
        tip = theme_button_tooltip(DARK)
        self.assertIn("Dark", tip)
        self.assertIn("Light", tip)
        self.assertIn("saved", tip)

    def test_apply_without_application_is_a_noop(self):
        self.assertEqual(apply_theme(None, LIGHT), LIGHT)


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the appearance tests")
class TestSettingsTheme(unittest.TestCase):
    """The Settings dialog is now the only place that switches the theme."""

    def test_settings_combo_and_switch_button_apply_and_persist(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from gui.app import MainWindow

        app = QApplication.instance() or QApplication([])
        self.addCleanup(lambda: apply_theme(app, DARK))
        with tempfile.TemporaryDirectory() as tmp:
            store = QSettings(str(Path(tmp) / "ui.ini"), QSettings.Format.IniFormat)
            save_theme(DARK, store)
            window = MainWindow(ui_settings=store)
            dialog = window.settings_dialog
            self.assertEqual(window.theme, DARK)
            self.assertEqual(dialog.current_theme(), DARK)
            self.assertIn("light", dialog.btn_switch_theme.text().lower())

            dialog.theme_combo.setCurrentIndex(
                dialog.theme_combo.findData(LIGHT))
            self.assertEqual(window.theme, LIGHT)
            self.assertEqual(load_theme(store), LIGHT)
            self.assertEqual(app.styleSheet(), stylesheet(LIGHT))
            self.assertEqual(app.palette().color(QPalette.ColorRole.Window).name(),
                             "#f3f6fa")
            self.assertIn("dark", dialog.btn_switch_theme.text().lower())

            dialog.btn_switch_theme.click()
            self.assertEqual(window.theme, DARK)
            self.assertEqual(load_theme(store), DARK)
            window.close()


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the appearance tests")
class TestHintMarkers(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from gui.app import MainWindow

        self.app = QApplication.instance() or QApplication([])
        self.addCleanup(lambda: apply_theme(self.app, DARK))
        self.addCleanup(QToolTip.hideText)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        store = QSettings(str(Path(self.tmp.name) / "ui.ini"),
                          QSettings.Format.IniFormat)
        self.window = MainWindow(ui_settings=store)

    def test_every_parameter_has_a_clickable_marker(self):
        from gui.app import TOOLTIPS, HintMarker

        self.assertEqual(set(self.window.hint_markers), set(TOOLTIPS))
        self.assertIn("QPushButton#paramHint", self.app.styleSheet())
        for key, marker in self.window.hint_markers.items():
            self.assertIsInstance(marker, HintMarker)
            self.assertEqual(marker.objectName(), "paramHint")
            self.assertTrue(marker.isFlat())
            self.assertEqual(marker.width(), HintMarker.SIZE)
            self.assertEqual(marker.height(), HintMarker.SIZE)
            self.assertIn(escape(TOOLTIPS[key]), marker.toolTip())

    def test_click_displays_the_hint(self):
        QToolTip.hideText()
        self.assertFalse(QToolTip.isVisible())
        self.window.hint_markers["swl"].click()
        self.app.processEvents()
        self.assertTrue(QToolTip.isVisible())


if __name__ == "__main__":
    unittest.main()
