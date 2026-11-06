"""Desktop GUI entry point for DCC Studio.

This module is intentionally tiny: it only prepares the process environment
and hands control to :class:`gui.app.MainWindow`. Keeping it separate from the
window implementation gives PyInstaller a stable, minimal entry script and
makes the frozen self-test able to construct the window without starting the
Qt event loop.

Two details matter when running from a frozen executable:

* stdout/stderr may use a console encoding that cannot represent every log
  character, so they are reconfigured with ``errors="replace"``;
* the project directory must be on ``sys.path`` in a source checkout, while a
  frozen bundle already contains every package.
"""
from __future__ import annotations

import sys
from pathlib import Path

# A windowed build has no console; a source run may use a legacy code page.
# Reconfiguring avoids UnicodeEncodeError when a sample id contains a non-ASCII
# character. Failures here are harmless, so they are deliberately ignored.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication          # noqa: E402 (path setup above)
from gui.app import MainWindow                    # noqa: E402


def main() -> int:
    """Create the application, show the main window, and run the event loop.

    ``Fusion`` is set here as a baseline; ``MainWindow`` subsequently applies
    the saved light/dark theme (which also sets its own palette and style
    sheet). ``QApplication`` must exist before any QWidget is constructed.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("DCC Studio")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
