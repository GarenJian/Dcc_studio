# -*- mode: python ; coding: utf-8 -*-
"""Minimal PyInstaller configuration for DCC Studio (PyInstaller 6.x).

There is exactly one distributable application: the GUI executable. The
command-line front end was removed, so every build shares the same PyQt6
dependency set and the same trimming rules. The only variant is a temporary
*self-test* executable used by ``build_exe.py --verify``; it is deleted after
the check and never shipped.

Called by ``build_exe.py``, or directly::

    .venv\\Scripts\\python.exe -m PyInstaller build_exe.spec --noconfirm

Environment variables
---------------------
``DCC_ONEFILE=1``
    Produce a single executable (default in ``build_exe.py``, smallest).
    Without it, an onedir folder is produced, which starts faster but is
    roughly 2.5x larger on disk.
``DCC_CONSOLE=1``
    Keep a console window for the GUI build. Useful when diagnosing a startup
    failure: tracebacks disappear in windowed mode.
``DCC_SELFTEST=1``
    Build ``packaging/selftest.py`` instead of the GUI. The self-test runs one
    synthetic analysis workflow and constructs the main window, which covers
    the complete GUI import chain inside the frozen environment.

Why the build is small
----------------------
1. Unused standard-library modules, optional scientific packages and Qt
   bindings are explicitly excluded. Qt is by far the largest contributor.
2. Only the DLLs and platform/image plugins actually needed are kept;
   WebEngine, Quick, 3D, translations and software OpenGL are dropped.
3. matplotlib sample data, mathtext-only fonts and package test data are
   removed, together with components verified unused (AVIF, OpenSSL).
4. UPX is disabled: compressing Qt DLLs can break startup and trigger
   antivirus false positives. Single-file output relies on PyInstaller's own
   zlib compression, about 40% of the onedir size.
"""
import os
from pathlib import Path

from PyInstaller.building.datastruct import TOC

ROOT = Path(SPEC).resolve().parent

# Three independent switches. SELFTEST replaces the entry point entirely;
# ONEFILE only changes how the EXE object is assembled at the bottom.
SELFTEST = os.environ.get("DCC_SELFTEST") == "1"
ONEFILE = os.environ.get("DCC_ONEFILE") == "1"
CONSOLE = SELFTEST or os.environ.get("DCC_CONSOLE") == "1"
NAME = "dcc_verify" if SELFTEST else "DCCStudio"
ENTRY = ROOT / ("packaging/selftest.py" if SELFTEST else "run_gui.py")

# Runtime dynamic imports that PyInstaller's static analysis cannot discover.
# Keeping this list short is part of the size strategy: every hidden import
# pulls in a whole dependency subtree.
HIDDEN = [
    "h5py",                             # HDF5-based .mat containers
    "scipy.io", "scipy.io.matlab",      # classic .mat containers
    "matplotlib.backends.backend_agg",  # plotting.py uses the Agg backend
    "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
]

# Anything listed here is excluded even if a bundled package references it.
EXCLUDES = [
    # Standard library. Keep unittest/doctest/pydoc: pyparsing needs them.
    "tkinter", "idlelib", "lib2to3", "test", "turtle", "curses",
    "ensurepip", "venv",
    # Optional scientific / interactive components.
    "IPython", "jupyter", "notebook", "sphinx", "sympy", "numba",
    "Cython", "pyarrow", "numexpr", "bottleneck", "tables", "zarr",
    "fsspec", "xlrd", "odf", "lxml", "sqlalchemy",
    # No longer used by the generic application.
    "nibabel", "openpyxl", "et_xmlfile",
]

# Qt ships many optional modules. Excluding them prevents PyInstaller from
# copying their DLLs; the remaining DLLs are filtered again by _trim().
QT_MODULES = [
    "QtWebEngineCore", "QtWebEngineWidgets", "QtWebChannel", "QtWebSockets",
    "QtWebView", "QtQuick", "QtQuick3D", "QtQuickWidgets", "QtQml",
    "QtMultimedia", "QtMultimediaWidgets", "Qt3DCore", "Qt3DRender",
    "Qt3DInput", "Qt3DLogic", "Qt3DAnimation", "Qt3DExtras",
    "QtCharts", "QtDataVisualization", "QtGraphs", "QtSql", "QtTest",
    "QtDesigner", "QtHelp", "QtOpenGL", "QtOpenGLWidgets", "QtPdf",
    "QtPdfWidgets", "QtSvgWidgets", "QtBluetooth", "QtNfc",
    "QtPositioning", "QtSerialPort", "QtSerialBus", "QtSensors",
    "QtTextToSpeech", "QtUiTools", "QtAxContainer", "QtRemoteObjects",
    "QtScxml", "QtStateMachine", "QtSpatialAudio", "QtShaderTools",
    "QtVirtualKeyboard", "QtGrpc", "QtProtobuf", "QtHttpServer",
    "QtLocation", "QtMqtt", "QtCoap", "QtOpcUa", "QtWaylandCompositor",
]
EXCLUDES += ["PyQt6." + module for module in QT_MODULES]

# Whitelists used by _trim() on the collected binaries and data files.
QT_DLL_KEEP = {
    "qt6core.dll", "qt6gui.dll", "qt6widgets.dll",
    "qt6network.dll", "qt6svg.dll", "qt6printsupport.dll",
}
QT_PLUGIN_KEEP = {
    "qwindows.dll",                                     # Windows platform plugin
    "qpng.dll", "qjpeg.dll", "qgif.dll", "qico.dll",    # QPixmap previews
    "qbmp.dll", "qwebp.dll", "qsvg.dll",
}
QT_DROP_FILES = {
    "d3dcompiler_47.dll", "opengl32sw.dll", "libegl.dll", "libglesv2.dll",
}
# Verified removable: the app only saves PNG, does no networking, no tkinter.
OPTIONAL_DROP = {"libcrypto-3.dll", "libssl-3.dll", "_ssl.pyd"}
OPTIONAL_DROP_PREFIXES = ("_avif.", "_imagingtk.")
# matplotlib hard-codes LastResortHE as its last-resort font; DejaVu is the
# default family. Only mathtext-only STIX / cm* fonts are removed.
MPL_FONT_KEEP = {"lastresorthe-regular.ttf"}


def _trim(toc, label):
    """Filter Qt / matplotlib files by a name whitelist.

    ``toc`` is a PyInstaller TOC: an iterable of ``(dest, src, typecode)``.
    Only file *names* are inspected, so the rules stay independent of the
    PyInstaller version's directory layout. The printed summary is what makes
    build-size regressions visible in the build log.
    """
    kept, dropped, saved = [], 0, 0
    for item in toc:
        dest = str(item[0]).replace("\\", "/")
        low = dest.lower()
        name = dest.rsplit("/", 1)[-1].lower()
        drop = False
        if name.startswith("qt6") and name.endswith(".dll"):
            drop = name not in QT_DLL_KEEP
        elif "pyqt6/" in low and "/plugins/" in low:
            drop = name not in QT_PLUGIN_KEEP
        elif "pyqt6/" in low and "/translations/" in low:
            drop = True
        elif (name in QT_DROP_FILES or name in OPTIONAL_DROP
              or name.startswith(OPTIONAL_DROP_PREFIXES)):
            drop = True
        elif "/mpl-data/fonts/ttf/" in low and not (
                name.startswith("dejavu") or name in MPL_FONT_KEEP):
            drop = True
        elif "/mpl-data/sample_data/" in low or "/mpl-data/fonts/pdfcorefonts/" in low:
            drop = True
        elif "/tests/" in low and name.endswith(
                (".py", ".pyc", ".csv", ".txt", ".npy", ".npz")):
            drop = True
        if drop:
            dropped += 1
            try:
                saved += os.path.getsize(item[1])
            except OSError:
                pass
        else:
            kept.append(item)
    print(f"[trim] {label}: {len(toc)} -> {len(kept)}, dropped {dropped} "
          f"({saved / 1048576:.1f} MB)")
    return TOC(kept)


a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT)],
    binaries=[],
    # The example settings file must be present in the frozen bundle: when no
    # local config.json exists, load_settings() falls back to it.
    datas=[(str(ROOT / "config.example.json"), ".")],
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
# Trimming after Analysis (rather than via excludes alone) is what removes the
# Qt DLL/plugin payload that hooks add implicitly.
a.binaries = _trim(a.binaries, "binaries")
a.datas = _trim(a.datas, "datas")

pyz = PYZ(a.pure)

exe_kwargs = dict(
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
if ONEFILE:
    # Everything is embedded in one EXE and unpacked to a temp folder on start.
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, **exe_kwargs)
else:
    # Onedir: the EXE holds only the bootloader and Python archive.
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_kwargs)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
                   upx_exclude=[], name=NAME)
