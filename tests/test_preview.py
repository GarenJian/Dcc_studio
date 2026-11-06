"""Tests for the bounded, cached Results-page preview.

The Results panel used to decode an entire file on every single click. These
tests pin down the three guarantees that keep it responsive:

* the ``gui.preview`` readers are bounded (rows, columns, characters, pixels);
* ``PreviewCache`` serves repeat clicks without re-reading and invalidates an
  entry as soon as the file is rewritten;
* ``MainWindow`` debounces selection, drops stale requests and fills the CSV
  grid in one bounded, signal-free batch.

PyQt6 is optional for CLI-only installations, so the Qt-dependent classes are
skipped when it is unavailable.
"""
from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

# QPixmap and QImageReader abort the interpreter if no QGuiApplication exists,
# so these tests create one headless instance up front and reuse it everywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings, Qt
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import (QApplication, QHeaderView, QTreeWidgetItem)
    from gui.preview import (CACHE_LIMIT, MAX_IMAGE_WIDTH, MAX_PREVIEW_COLS,
                             MAX_PREVIEW_ROWS, MAX_TEXT_CHARS, PreviewCache,
                             load_pixmap_preview, read_csv_preview,
                             read_text_preview)
    HAS_QT = True
except ImportError:                     # GUI dependency is optional for CLI use
    HAS_QT = False

if HAS_QT:
    APP = QApplication.instance() or QApplication([])


def write_csv(path: Path, columns: int, rows: int) -> None:
    """Write a deterministic, wide CSV without pulling pandas into the tests."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"c{i}" for i in range(columns)])
        for row in range(rows):
            writer.writerow([f"{row}.{i}" for i in range(columns)])


def write_png(path: Path, width: int, height: int) -> None:
    """Write a solid-colour PNG so decoding cost is dominated by its size."""
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(0xFF224466)
    if not image.save(str(path)):
        raise AssertionError(f"could not write {path}")


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the preview tests")
class TestPreviewCache(unittest.TestCase):
    """The cache must be a hit on repeat clicks and a miss after any edit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "data.txt"
        self.path.write_text("first", encoding="utf-8")

    def test_repeated_get_reuses_the_value(self):
        cache = PreviewCache()
        calls: list[Path] = []

        def loader(path):
            calls.append(path)
            return path.read_text(encoding="utf-8")

        self.assertEqual(cache.get(self.path, loader), "first")
        self.assertEqual(cache.get(self.path, loader), "first")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(cache), 1)

    def test_rewrite_invalidates_the_entry(self):
        cache = PreviewCache()
        calls: list[Path] = []

        def loader(path):
            calls.append(path)
            return path.read_text(encoding="utf-8")

        cache.get(self.path, loader)
        self.path.write_text("a longer second value", encoding="utf-8")
        self.assertEqual(cache.get(self.path, loader), "a longer second value")
        self.assertEqual(len(calls), 2)

    def test_mtime_change_invalidates_even_when_size_is_unchanged(self):
        cache = PreviewCache()
        calls: list[Path] = []

        def loader(path):
            calls.append(path)
            return path.read_text(encoding="utf-8")

        cache.get(self.path, loader)
        stat = self.path.stat()
        os.utime(self.path,
                 ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        cache.get(self.path, loader)
        self.assertEqual(len(calls), 2)

    def test_lru_evicts_the_oldest_entry_and_clear_empties_it(self):
        cache = PreviewCache(limit=2)
        paths = []
        for name in "abc":
            path = Path(self.tmp.name) / name
            path.write_text(name, encoding="utf-8")
            paths.append(path)
        loader = lambda path: path.read_text(encoding="utf-8")
        for path in paths:
            cache.get(path, loader)
        self.assertEqual(len(cache), 2)

        calls: list[Path] = []

        def counted(path):
            calls.append(path)
            return loader(path)

        cache.get(paths[0], counted)   # "a" was evicted
        self.assertEqual(len(calls), 1)
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_default_limit_bounds_a_long_browsing_session(self):
        cache = PreviewCache()
        loader = lambda path: path.read_text(encoding="utf-8")
        for index in range(CACHE_LIMIT + 6):
            path = Path(self.tmp.name) / f"item{index}.txt"
            path.write_text(str(index), encoding="utf-8")
            cache.get(path, loader)
        self.assertEqual(len(cache), CACHE_LIMIT)

    def test_missing_file_is_not_cached(self):
        cache = PreviewCache()
        missing = Path(self.tmp.name) / "missing"
        calls: list[Path] = []

        def loader(path):
            calls.append(path)
            return "fallback"

        self.assertEqual(cache.get(missing, loader), "fallback")
        self.assertEqual(cache.get(missing, loader), "fallback")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cache), 0)

    def test_failed_loader_is_not_cached(self):
        cache = PreviewCache()
        calls: list[Path] = []

        def loader(path):
            calls.append(path)
            raise ValueError("boom")

        for _ in range(2):
            with self.assertRaises(ValueError):
                cache.get(self.path, loader)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cache), 0)


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the preview tests")
class TestBoundedReaders(unittest.TestCase):
    """Reading a preview must cost O(visible cells), not O(file size)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _csv(self, name: str, columns: int, rows: int) -> Path:
        path = self.dir / name
        write_csv(path, columns, rows)
        return path

    def test_csv_limits_rows_and_columns_and_reports_truncation(self):
        total = MAX_PREVIEW_COLS + 20
        path = self._csv("wide.csv", total, MAX_PREVIEW_ROWS + 25)
        data = read_csv_preview(path)

        self.assertEqual(len(data.headers), MAX_PREVIEW_COLS)
        self.assertEqual(len(data.rows), MAX_PREVIEW_ROWS)
        self.assertTrue(all(len(row) == MAX_PREVIEW_COLS
                            for row in data.rows))
        self.assertEqual(data.total_columns, total)
        self.assertEqual(data.shown_columns, MAX_PREVIEW_COLS)
        self.assertTrue(data.truncated_rows)
        self.assertTrue(data.truncated_columns)
        self.assertEqual(data.headers[0], "c0")
        self.assertEqual(data.rows[0][0], "0.0")
        self.assertEqual(data.rows[-1][-1],
                         f"{MAX_PREVIEW_ROWS - 1}.{MAX_PREVIEW_COLS - 1}")

    def test_small_csv_is_not_marked_as_truncated(self):
        path = self._csv("small.csv", 3, 2)
        data = read_csv_preview(path)
        self.assertEqual(data.headers, ["c0", "c1", "c2"])
        self.assertEqual(data.rows, [["0.0", "0.1", "0.2"],
                                     ["1.0", "1.1", "1.2"]])
        self.assertEqual(data.total_columns, 3)
        self.assertEqual(data.shown_columns, 3)
        self.assertFalse(data.truncated_rows)
        self.assertFalse(data.truncated_columns)

    def test_exactly_the_row_cap_is_not_marked_as_truncated(self):
        path = self._csv("exact.csv", 2, MAX_PREVIEW_ROWS)
        data = read_csv_preview(path)
        self.assertEqual(len(data.rows), MAX_PREVIEW_ROWS)
        self.assertFalse(data.truncated_rows)

    def test_header_only_csv_has_no_rows(self):
        path = self._csv("header.csv", 4, 0)
        data = read_csv_preview(path)
        self.assertEqual(len(data.headers), 4)
        self.assertEqual(data.rows, [])
        self.assertFalse(data.truncated_rows)

    def test_text_read_is_capped_at_the_character_limit(self):
        payload = "x" * (MAX_TEXT_CHARS + 50_000)
        path = self.dir / "log.txt"
        path.write_text(payload, encoding="utf-8")
        self.assertEqual(read_text_preview(path), payload[:MAX_TEXT_CHARS])

    def test_text_read_replaces_undecodable_bytes(self):
        path = self.dir / "log.txt"
        path.write_bytes(b"\xff\xfeabc")
        self.assertTrue(read_text_preview(path).endswith("abc"))

    def test_large_png_is_scaled_to_display_width(self):
        path = self.dir / "figure.png"
        write_png(path, MAX_IMAGE_WIDTH * 4, 1200)
        pixmap = load_pixmap_preview(path)
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 0)
        self.assertLessEqual(pixmap.width(), MAX_IMAGE_WIDTH)

    def test_small_png_keeps_its_native_width(self):
        path = self.dir / "figure.png"
        write_png(path, 320, 120)
        self.assertEqual(load_pixmap_preview(path).width(), 320)

    def test_broken_image_returns_a_null_pixmap(self):
        path = self.dir / "figure.png"
        path.write_bytes(b"not a png")
        self.assertTrue(load_pixmap_preview(path).isNull())

    def test_cache_reuses_the_decoded_pixmap(self):
        path = self.dir / "figure.png"
        write_png(path, 320, 120)
        cache = PreviewCache()
        first = cache.get(path, load_pixmap_preview)
        second = cache.get(path, load_pixmap_preview)
        self.assertIs(first, second)


@unittest.skipUnless(HAS_QT, "PyQt6 is required for the preview tests")
class TestResultsPreviewGUI(unittest.TestCase):
    """Drive the real Results page without waiting for wall-clock timers."""

    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from gui.app import MainWindow
        from gui.theme import DARK, apply_theme

        self.app = QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        store = QSettings(str(self.dir / "ui.ini"),
                          QSettings.Format.IniFormat)
        self.window = MainWindow(ui_settings=store)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: apply_theme(self.app, DARK))
        self.addCleanup(self.window.close)
        self.addCleanup(self._stop_timer)
        self._stop_timer()

    def _stop_timer(self):
        self.window._preview_timer.stop()
        self.window._pending_preview = None

    def _csv(self, name: str, columns: int = 3, rows: int = 2) -> Path:
        path = self.dir / name
        write_csv(path, columns, rows)
        return path

    def _add_file(self, path: Path) -> QTreeWidgetItem:
        tree = self.window.tree_files
        tree.blockSignals(True)
        item = QTreeWidgetItem(tree, [path.name])
        item.setData(0, Qt.ItemDataRole.UserRole, str(path))
        tree.setCurrentItem(item)
        tree.blockSignals(False)
        return item

    def _select(self, path: Path) -> None:
        """Select a file and fire the debounced render synchronously."""
        item = self._add_file(path)
        self.window._on_file_selected(item, None)
        self.window._preview_timer.stop()
        self.window._render_pending_preview()

    def _spy_on_cache(self) -> list[Path]:
        calls: list[Path] = []
        original = self.window._preview_cache.get

        def spy(path, loader):
            calls.append(path)
            return original(path, loader)

        self.window._preview_cache.get = spy
        return calls

    def test_timer_is_single_shot_and_table_uses_fixed_column_widths(self):
        timer = self.window._preview_timer
        self.assertTrue(timer.isSingleShot())
        self.assertGreater(timer.interval(), 0)
        header = self.window.table.horizontalHeader()
        self.assertEqual(header.defaultSectionSize(), 110)
        self.window._fill_table(read_csv_preview(self._csv("cols.csv")))
        self.assertEqual(
            [header.sectionResizeMode(i) for i in range(3)],
            [QHeaderView.ResizeMode.Interactive] * 3)

    def test_debounce_renders_only_the_last_selected_file(self):
        first = self._csv("a.csv")
        last = self._csv("b.csv")
        item_first = self._add_file(first)
        self._add_file(last)
        calls = self._spy_on_cache()

        self.window._on_file_selected(item_first, None)
        self.window._on_file_selected(self.window.tree_files.currentItem(), None)
        self.assertEqual(self.window._pending_preview, last)
        self.assertTrue(self.window._preview_timer.isActive())

        self.window._preview_timer.stop()
        self.window._render_pending_preview()
        self.assertEqual(calls, [last])
        self.assertIsNone(self.window._pending_preview)
        self.assertEqual(self.window.preview.currentIndex(), 2)
        self.assertIn("b.csv", self.window.lbl_path.text())

    def test_stale_request_is_never_rendered(self):
        stale = self._csv("stale.csv")
        current = self._csv("current.csv")
        item_stale = self._add_file(stale)
        self._add_file(current)
        calls = self._spy_on_cache()

        self.window._on_file_selected(item_stale, None)
        self.window._preview_timer.stop()
        self.window._render_pending_preview()
        self.assertEqual(calls, [])
        self.assertIsNone(self.window._pending_preview)

    def test_refresh_cancels_a_pending_preview(self):
        self._add_file(self._csv("a.csv"))
        self.window._on_file_selected(
            self.window.tree_files.currentItem(), None)
        self.window._refresh_files()
        self.assertIsNone(self.window._pending_preview)
        self.assertFalse(self.window._preview_timer.isActive())

    def test_selecting_a_group_node_does_not_schedule_work(self):
        tree = self.window.tree_files
        group = QTreeWidgetItem(tree, ["Tables (1)"])
        group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.window._on_file_selected(group, None)
        self.assertIsNone(self.window._pending_preview)
        self.assertFalse(self.window._preview_timer.isActive())

    def test_wide_csv_grid_is_bounded_and_caption_explains_truncation(self):
        total = MAX_PREVIEW_COLS + 20
        path = self._csv("wide.csv", total, MAX_PREVIEW_ROWS + 5)
        self._select(path)
        table = self.window.table
        self.assertEqual(table.columnCount(), MAX_PREVIEW_COLS)
        self.assertEqual(table.rowCount(), MAX_PREVIEW_ROWS)
        self.assertEqual(table.item(0, 0).text(), "0.0")
        self.assertEqual(
            table.horizontalHeaderItem(MAX_PREVIEW_COLS - 1).text(),
            f"c{MAX_PREVIEW_COLS - 1}")
        caption = self.window.lbl_path.text()
        self.assertIn(str(path), caption)
        self.assertIn(f"first {MAX_PREVIEW_ROWS} rows", caption)
        self.assertIn(f"first {MAX_PREVIEW_COLS} of {total} columns", caption)

    def test_untruncated_csv_caption_is_just_the_path(self):
        path = self._csv("small.csv")
        self._select(path)
        self.assertEqual(self.window.lbl_path.text(), str(path))
        self.assertEqual(self.window.table.columnCount(), 3)
        self.assertEqual(self.window.table.rowCount(), 2)

    def test_png_preview_uses_the_image_page_at_display_size(self):
        path = self.dir / "figure.png"
        write_png(path, MAX_IMAGE_WIDTH * 3, 600)
        self._select(path)
        self.assertEqual(self.window.preview.currentIndex(), 1)
        pixmap = self.window.lbl_img.pixmap()
        self.assertIsNotNone(pixmap)
        self.assertFalse(pixmap.isNull())
        self.assertLessEqual(pixmap.width(), MAX_IMAGE_WIDTH)

    def test_text_preview_uses_the_text_page(self):
        path = self.dir / "summary.json"
        path.write_text('{"ok": true}', encoding="utf-8")
        self._select(path)
        self.assertEqual(self.window.preview.currentIndex(), 3)
        self.assertIn('"ok"', self.window.text_view.toPlainText())

    def test_binary_output_is_explained_instead_of_decoded(self):
        path = self.dir / "array.npy"
        path.write_bytes(b"\x00\x01\x02")
        self._select(path)
        self.assertEqual(self.window.preview.currentIndex(), 3)
        self.assertIn("numpy.load", self.window.text_view.toPlainText())

    def test_a_broken_csv_shows_an_error_instead_of_escaping_into_qt(self):
        path = self.dir / "broken.csv"
        path.write_text("", encoding="utf-8")
        self._select(path)
        self.assertEqual(self.window.preview.currentIndex(), 3)
        self.assertIn("Cannot preview", self.window.text_view.toPlainText())


if __name__ == "__main__":
    unittest.main()
