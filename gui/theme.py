"""Light and dark appearances for the DCC Studio desktop interface.

A theme consists of two related things:

1. A Qt style sheet rendered from one ``Template`` and a colour dictionary.
   Keeping a single template (rather than two hand-written sheets) guarantees
   that both themes style exactly the same selectors; only the values differ.
   The small ``!`` parameter markers, tooltips, progress bar, group boxes and
   scroll bars are all covered.
2. A ``QPalette`` for the Fusion style, so native widget parts that ignore the
   style sheet (menus, selection colours, disabled text, message boxes) also
   follow the theme.

The preference is presentation-only. It is stored with ``QSettings`` in the
user profile, separate from ``config.json``, so switching it never changes
computed results, never enters a result directory and behaves identically in
source and frozen builds. Tests inject a temporary ``QSettings`` object to
avoid touching the real user profile.

Public API: ``apply_theme`` (application-wide switch), ``stylesheet`` and
``build_palette`` (pieces), ``load_theme`` / ``save_theme`` (persistence),
``toggle_theme`` / ``theme_button_text`` / ``theme_button_tooltip`` (UI).
"""
from __future__ import annotations

from functools import lru_cache
from string import Template

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

DARK = "dark"
LIGHT = "light"
DEFAULT_THEME = DARK
THEMES = (DARK, LIGHT)

_ORGANIZATION = "DCC Studio"
_APPLICATION = "DCC Studio"
_KEY = "appearance/theme"

COLORS = {
    DARK: {
        "bg": "#151b23", "title": "#f2f7fb", "subtitle": "#8fa0b2",
        "hint": "#8496a8", "text": "#dfe7ef",
        "accent": "#6fe0c6", "accent_border": "#3bbda6",
        "accent_done": "#8fbfa8", "accent_done_border": "#2d6a5f",
        "accent_soft": "#1c3c3a", "hint_marker": "#64798c",
        "border": "#2b3644", "input_bg": "#111821",
        "input_border": "#33404f", "selection": "#2c6b74",
        "step_bg": "#1b232d", "step_hover_text": "#cdd9e5",
        "step_active_bg": "#243140", "stage_idle": "#6f7d8c",
        "button_bg": "#243140", "button_border": "#3a4b5e",
        "button_hover_bg": "#2f4257", "button_hover_border": "#4a6076",
        "disabled_bg": "#1a222c", "disabled_fg": "#5f6d7c",
        "primary": "#258f7d", "primary_hover": "#2ca78f",
        "primary_border": "#36b9a2", "on_accent": "#ffffff",
        "danger": "#7d3438", "danger_border": "#a24a4f",
        "danger_hover": "#984045",
        "header_bg": "#1d2836", "header_fg": "#aebdcc",
        "scroll_bg": "#151b23", "scroll_handle": "#3a4a5c",
        "splitter": "#2b3644",
        "grad_start": "#2a8f80", "grad_end": "#58d7bd",
        "tooltip_bg": "#1b232d", "tooltip_border": "#3a4b5e",
    },
    LIGHT: {
        "bg": "#f3f6fa", "title": "#10202e", "subtitle": "#56677a",
        "hint": "#5d6e80", "text": "#1b2733",
        "accent": "#0c7565", "accent_border": "#0e8a76",
        "accent_done": "#2c7a66", "accent_done_border": "#9ccfbf",
        "accent_soft": "#dcf3ec", "hint_marker": "#9aa9b8",
        "border": "#cbd6e2", "input_bg": "#ffffff",
        "input_border": "#b9c7d6", "selection": "#a6d9cf",
        "step_bg": "#ffffff", "step_hover_text": "#26374a",
        "step_active_bg": "#e8eef5", "stage_idle": "#8492a2",
        "button_bg": "#e8eef5", "button_border": "#b0c0d0",
        "button_hover_bg": "#dce7f2", "button_hover_border": "#8fa8bf",
        "disabled_bg": "#e9eef4", "disabled_fg": "#98a6b5",
        "primary": "#128c78", "primary_hover": "#0fa389",
        "primary_border": "#0b6e5f", "on_accent": "#ffffff",
        "danger": "#b34449", "danger_border": "#8f3338",
        "danger_hover": "#c95358",
        "header_bg": "#e4ecf5", "header_fg": "#33465b",
        "scroll_bg": "#f3f6fa", "scroll_handle": "#b6c5d5",
        "splitter": "#cbd6e2",
        "grad_start": "#128c78", "grad_end": "#4fc3aa",
        "tooltip_bg": "#ffffff", "tooltip_border": "#b9c7d6",
    },
}

_QSS = Template("""
QWidget { background:$bg; color:$text;
          font-family:"Segoe UI",Arial,sans-serif; font-size:12px; }
QMainWindow { background:$bg; }
QLabel#title { font-size:22px; font-weight:bold; color:$title; }
QLabel#subtitle { color:$subtitle; font-size:12px; }
QLabel#hint { color:$hint; }
QPushButton#paramHint { background:transparent; border:0; padding:0;
                        color:$hint_marker; font-size:9px; font-weight:normal; }
QPushButton#paramHint:hover, QPushButton#paramHint:pressed { color:$accent; }
QPushButton#step { background:$step_bg; color:$subtitle;
                   border:1px solid $border; border-radius:6px;
                   padding:8px 16px; text-align:left; }
QPushButton#step:hover { color:$step_hover_text;
                         border-color:$button_hover_border; }
QPushButton#stepActive { background:$step_active_bg; color:$accent;
                         border:1px solid $accent_border; border-radius:6px;
                         padding:8px 16px; text-align:left; }
QLabel#stageIdle { color:$stage_idle; padding:6px 10px;
                   border:1px solid $border; border-radius:6px; }
QLabel#stageActive { color:$accent; padding:6px 10px;
                     border:1px solid $accent_border; border-radius:6px;
                     background:$accent_soft; }
QLabel#stageDone { color:$accent_done; padding:6px 10px;
                   border:1px solid $accent_done_border; border-radius:6px; }
QGroupBox { border:1px solid $border; border-radius:9px; margin-top:16px;
            padding:14px 12px 10px; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 6px;
                   color:$accent; font-weight:bold; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QListWidget,
QTableWidget, QTextBrowser, QTreeWidget { background:$input_bg;
            border:1px solid $input_border; border-radius:6px; padding:6px;
            color:$text; selection-background-color:$selection; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color:$accent_border; }
QPushButton { background:$button_bg; border:1px solid $button_border;
              border-radius:7px; padding:8px 15px; color:$text; }
QPushButton:hover { background:$button_hover_bg;
                    border-color:$button_hover_border; }
QPushButton:disabled { color:$disabled_fg; background:$disabled_bg;
                       border-color:$border; }
QPushButton#settings { min-width:104px; padding:7px 12px; }
QPushButton#primary { background:$primary; border:1px solid $primary_border;
                      color:$on_accent; font-weight:bold; }
QPushButton#primary:hover { background:$primary_hover; }
QPushButton#danger { background:$danger; border:1px solid $danger_border;
                     color:$on_accent; }
QPushButton#danger:hover { background:$danger_hover; }
QProgressBar { border:1px solid $input_border; border-radius:7px;
               background:$input_bg; height:18px; text-align:center;
               color:$text; }
QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                      stop:0 $grad_start, stop:1 $grad_end); border-radius:6px; }
QHeaderView::section { background:$header_bg; color:$header_fg; border:0px;
                       border-right:1px solid $border; padding:6px; }
QTreeWidget::item { padding:3px; }
QScrollBar:vertical { background:$scroll_bg; width:10px; margin:0; }
QScrollBar::handle:vertical { background:$scroll_handle; border-radius:5px;
                              min-height:24px; }
QScrollBar:horizontal { background:$scroll_bg; height:10px; }
QScrollBar::handle:horizontal { background:$scroll_handle; border-radius:5px;
                                min-width:24px; }
QSplitter::handle { background:$splitter; }
QToolTip { background:$tooltip_bg; color:$text;
           border:1px solid $tooltip_border; padding:5px; }
""")


def normalize_theme(name) -> str:
    """Return a supported theme name, falling back to the default."""
    value = str(name if name is not None else "").strip().lower()
    return value if value in THEMES else DEFAULT_THEME


def toggle_theme(name) -> str:
    """Return the other theme."""
    return LIGHT if normalize_theme(name) == DARK else DARK


@lru_cache(maxsize=None)
def stylesheet(name: str) -> str:
    """Render the Qt style sheet for a theme."""
    return _QSS.substitute(COLORS[normalize_theme(name)])


def build_palette(name: str) -> QPalette:
    """Build a Fusion palette so native widgets follow the theme too."""
    colors = COLORS[normalize_theme(name)]
    palette = QPalette()
    active = {
        QPalette.ColorRole.Window: "bg",
        QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Base: "input_bg",
        QPalette.ColorRole.AlternateBase: "button_bg",
        QPalette.ColorRole.Text: "text",
        QPalette.ColorRole.Button: "button_bg",
        QPalette.ColorRole.ButtonText: "text",
        QPalette.ColorRole.Highlight: "selection",
        QPalette.ColorRole.HighlightedText: "text",
        QPalette.ColorRole.ToolTipBase: "tooltip_bg",
        QPalette.ColorRole.ToolTipText: "text",
    }
    for role, key in active.items():
        palette.setColor(role, QColor(colors[key]))
    disabled = QColor(colors["disabled_fg"])
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.HighlightedText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return palette


def apply_theme(app: QApplication | None, name=None, settings=None) -> str:
    """Apply a theme to the whole application and return its resolved name."""
    theme = normalize_theme(name) if name is not None else load_theme(settings)
    if app is None:
        return theme
    app.setStyle("Fusion")          # must precede the custom palette
    app.setPalette(build_palette(theme))
    app.setStyleSheet(stylesheet(theme))
    return theme


def _store(settings):
    """Return the injected QSettings, or the application's user-scope INI store."""
    if settings is not None:
        return settings
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                     _ORGANIZATION, _APPLICATION)


def load_theme(settings=None) -> str:
    """Read the saved appearance preference, or the default."""
    return normalize_theme(_store(settings).value(_KEY, DEFAULT_THEME))


def save_theme(name, settings=None) -> str:
    """Persist the appearance preference and return its resolved name."""
    theme = normalize_theme(name)
    store = _store(settings)
    store.setValue(_KEY, theme)
    store.sync()
    return theme


def theme_button_text(name) -> str:
    """Label the toggle with the theme it switches to."""
    return f"Switch to {toggle_theme(name)}"


def theme_button_tooltip(name) -> str:
    """Explain the current appearance, the target theme and that the choice is saved."""
    current = normalize_theme(name)
    return (f"Current appearance: {current.capitalize()}. "
            f"Click to switch to {toggle_theme(current).capitalize()}. "
            "The choice is saved and restored on the next launch.")
