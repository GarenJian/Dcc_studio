"""PyQt6 desktop interface package.

The GUI is a thin presentation layer over ``dcc.pipelines``:

* ``app.py``     -- four-step main window, parameter help markers, result browser;
* ``workers.py`` -- ``QThread`` wrapper that keeps long runs off the UI thread;
* ``theme.py``   -- light/dark style sheets, Fusion palettes and the saved
  appearance preference.

No numerical computation lives here. Importing ``gui.app`` requires a
``QApplication`` to exist before widgets are constructed; the computation
core (``dcc``) remains importable without Qt.
"""
