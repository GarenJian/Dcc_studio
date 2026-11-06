"""Fast, bounded readers for the Results page preview panel.

The preview used to read and render a whole result on every single click, on
the GUI thread. With real runs that is expensive: ``dcc_matrix.csv`` has one
column per sample (90 for MSC, 224 for UCLA), and publication figures are
several thousand pixels wide. Three rules keep the panel responsive:

* **Bounded work** -- a CSV preview reads at most ``MAX_PREVIEW_ROWS`` rows and
  ``MAX_PREVIEW_COLS`` columns, text is read with a character cap instead of
  ``read_text()[:n]`` (which still loaded the entire file), and images are
  downscaled *during decoding* rather than after a full-size decode.
* **Cached work** -- :class:`PreviewCache` keys entries by path, modification
  time and size, so clicking the same file again costs nothing and an edited
  file is never served from a stale entry. The cache is a small LRU so a long
  browsing session cannot grow without limit.
* **Deferred work** -- ``MainWindow`` debounces selection changes with a short
  ``QTimer``; this module only supplies the pure readers it invokes.

This module contains no window or widget state, which makes the readers and
the cache straightforward to unit-test without constructing the full UI.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImageReader, QPixmap

# A 200 x 40 grid is enough to judge a result table and stays cheap to insert
# into QTableWidget. The GUI tells the user when a preview was truncated.
MAX_PREVIEW_ROWS = 200
MAX_PREVIEW_COLS = 40
MAX_TEXT_CHARS = 200_000
MAX_IMAGE_WIDTH = 1000
CACHE_LIMIT = 24


@dataclass
class TablePreview:
    """A bounded CSV preview plus the metadata needed to explain truncation.

    ``rows`` already contains display strings, so the GUI can insert them
    without any further formatting work per cell.
    """

    headers: list[str]
    rows: list[list[str]]
    total_columns: int
    shown_columns: int
    truncated_rows: bool
    truncated_columns: bool


def _cache_key(path: Path) -> tuple[str, int, int]:
    """Build a key that changes whenever the file is rewritten or resized."""
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


class PreviewCache:
    """Tiny LRU cache of already-decoded previews.

    ``get(path, loader)`` calls ``loader(path)`` only on a miss. Including
    mtime and size in the key means a regenerated result directory is picked
    up automatically; callers never need to invalidate anything by hand.
    """

    def __init__(self, limit: int = CACHE_LIMIT) -> None:
        self._entries: OrderedDict[tuple, object] = OrderedDict()
        self._limit = max(1, limit)

    def get(self, path: Path, loader):
        """Return the cached value for ``path`` or compute and store it."""
        try:
            key = _cache_key(path)
        except OSError:
            # A missing/unreadable file should not poison the cache; let the
            # loader raise so the GUI can show a useful error message.
            return loader(path)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = loader(path)
        self._entries[key] = value
        while len(self._entries) > self._limit:
            self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every entry, e.g. when switching to another run directory."""
        self._entries.clear()

    def __len__(self) -> int:
        """Number of entries currently held (used by tests)."""
        return len(self._entries)


def read_csv_preview(path: Path) -> TablePreview:
    """Read a bounded rectangle from a CSV without parsing the whole table.

    The header is read with ``nrows=0`` first so the column count is known;
    ``usecols`` then makes pandas skip the remaining columns entirely. One
    extra row is read to detect truncation without scanning the full file.
    """
    import pandas as pd

    header = pd.read_csv(path, nrows=0)
    all_columns = [str(column) for column in header.columns]
    shown = min(len(all_columns), MAX_PREVIEW_COLS)
    frame = pd.read_csv(path, usecols=list(range(shown)),
                        nrows=MAX_PREVIEW_ROWS + 1)
    truncated_rows = len(frame) > MAX_PREVIEW_ROWS
    frame = frame.iloc[:MAX_PREVIEW_ROWS]
    return TablePreview(
        headers=[str(column) for column in frame.columns],
        rows=[[f"{value}" for value in row]
              for row in frame.itertuples(index=False, name=None)],
        total_columns=len(all_columns),
        shown_columns=shown,
        truncated_rows=truncated_rows,
        truncated_columns=len(all_columns) > shown,
    )


def read_text_preview(path: Path) -> str:
    """Read at most ``MAX_TEXT_CHARS`` characters, without loading the rest."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(MAX_TEXT_CHARS)


def load_pixmap_preview(path: Path) -> QPixmap:
    """Decode a PNG close to display size, then smooth-scale once.

    ``QPixmap(path)`` decoded the whole figure and smooth-scaled it on every
    click. Here the reader produces a 2x-target image during decoding, so the
    label never holds a multi-thousand-pixel pixmap and the final smooth
    downscale starts from a smaller source. A PNG has no cheap partial decode,
    so the *first* decode costs about what it used to; the decisive gains are
    the bounded output size and ``PreviewCache``, which makes repeat clicks
    free. Formats with real decode-time scaling (JPEG, SVG) also skip work.
    """
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and size.width() > MAX_IMAGE_WIDTH * 2:
        width = MAX_IMAGE_WIDTH * 2
        height = max(1, round(size.height() * width / size.width()))
        reader.setScaledSize(QSize(width, height))
    image = reader.read()
    if image.isNull():
        return QPixmap()
    if image.width() > MAX_IMAGE_WIDTH:
        image = image.scaledToWidth(
            MAX_IMAGE_WIDTH, Qt.TransformationMode.SmoothTransformation)
    return QPixmap.fromImage(image)
