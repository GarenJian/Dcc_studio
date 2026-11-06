"""Settings dialog: choose the analysis method and the appearance.

This dialog is the single entry point for the software's *global* choices. It
is opened from the "Settings" button on the first page (and from the shortcut
in the header), so a first-time user sees both options immediately instead of
hunting through the parameter page.

Two kinds of choices live here:

* **Analysis method** -- a ``QComboBox`` whose enabled rows come from the
  method registry (``dcc.methods.method_choices``). Registering a new method
  therefore makes it appear here with no change to the GUI. :data:`PLANNED_METHODS`
  lists measures from the roadmap that are not implemented yet; they are
  appended as *disabled* rows so users can see what is coming without being
  able to select something that would fail during a run. Once a planned method
  is actually registered, its placeholder disappears automatically, because
  the enabled row from the registry takes precedence.

* **Appearance** -- a ``QComboBox`` for Dark / Light plus a one-click
  "Switch to ..." button. Both emit :attr:`theme_changed`; ``MainWindow``
  applies the palette and persists the preference.

Changes apply immediately. A method change updates the in-memory
configuration and is written to ``config.json`` by the existing "Save
configuration" button on step 1, exactly like every other parameter. A theme
change is persisted at once, matching the behaviour of the old header toggle.

The dialog is non-modal and owned by ``MainWindow``. Tests can drive its
widgets directly without blocking on ``QDialog.exec()``.
"""
from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QGroupBox,
                             QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from dcc.methods import method_choices, method_descriptions
from gui.theme import (THEMES, normalize_theme, theme_button_text,
                       theme_button_tooltip, toggle_theme)

# Roadmap entries that are visible but not selectable. ``name`` is the stable
# identifier a future implementation would register; ``title`` is the label;
# ``description`` becomes the tooltip of the disabled row. Keep this tuple in
# sync with docs/METHOD.md when a new method is planned or shipped.
PLANNED_METHODS = (
    ("standard_dfc", "Standard dynamic functional connectivity",
     "Planned method: classic sliding-window dynamic functional connectivity "
     "without state coding or entropy. It is not implemented in this release; "
     "DCC is the only available method."),
)


class SettingsDialog(QDialog):
    """Non-modal dialog exposing the method selector and the theme switch.

    ``method_changed`` / ``theme_changed`` carry the *stable internal names*
    (for example ``"dcc"`` and ``"light"``), never the translated display
    text. The main window is the only listener; keeping the dialog free of
    application state makes it reusable and straightforward to unit-test.
    """

    method_changed = pyqtSignal(str)
    theme_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, method: str = "dcc",
                 theme: str = "dark") -> None:
        """Build the widgets, then seed them from the current window state.

        Signal connections are made only after the initial values are set, so
        constructing the dialog never emits a spurious change that would be
        written to disk.
        """
        super().__init__(parent)
        self.setWindowTitle("Settings - Method & Appearance")
        self.setMinimumWidth(540)
        self.setModal(False)

        root = QVBoxLayout(self)
        root.setSpacing(12)

        intro = QLabel(
            "Global choices for this workspace. The analysis method is applied "
            "to every sample; the appearance is presentation-only and never "
            "affects computed results.")
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        root.addWidget(intro)

        # ---------------------------------------------------------- Method
        method_group = QGroupBox("Analysis method")
        method_layout = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        self._method_model = QStandardItemModel()
        self._rebuild_method_model()
        self.method_combo.setModel(self._method_model)
        method_layout.addWidget(self.method_combo)

        self.lbl_method_note = QLabel("")
        self.lbl_method_note.setObjectName("hint")
        self.lbl_method_note.setWordWrap(True)
        method_layout.addWidget(self.lbl_method_note)

        persistence = QLabel(
            "Applies to the next run. Click 'Save configuration' on step 1 to "
            "keep this method for future launches.")
        persistence.setObjectName("hint")
        persistence.setWordWrap(True)
        method_layout.addWidget(persistence)
        root.addWidget(method_group)

        # ------------------------------------------------------- Appearance
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout(theme_group)
        theme_row = QHBoxLayout()
        self.theme_combo = QComboBox()
        for name in THEMES:
            self.theme_combo.addItem(name.capitalize(), name)
        theme_row.addWidget(self.theme_combo, 1)
        self.btn_switch_theme = QPushButton()
        theme_row.addWidget(self.btn_switch_theme)
        theme_layout.addLayout(theme_row)

        theme_note = QLabel(
            "Dark / Light changes the whole interface immediately. The choice "
            "is saved and restored on the next launch.")
        theme_note.setObjectName("hint")
        theme_note.setWordWrap(True)
        theme_layout.addWidget(theme_note)
        root.addWidget(theme_group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.hide)
        root.addWidget(buttons)

        # Wire signals first, then seed values through setters that block the
        # signals, so no change event fires during construction.
        self.method_combo.currentIndexChanged.connect(self._on_method_index)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_index)
        self.btn_switch_theme.clicked.connect(self._on_switch_clicked)
        self.set_method(method)
        self.set_theme(theme)

    # ------------------------------------------------------------- model
    def _rebuild_method_model(self) -> None:
        """Populate the method combo from the registry plus roadmap rows.

        Using an explicit ``QStandardItemModel`` (instead of ``addItem``) lets
        a planned method be shown in the list but marked disabled, which is
        the clearest way to communicate "not supported yet" in a combo box.
        """
        self._method_model.clear()
        descriptions = method_descriptions()
        choices = method_choices()
        registered = {name for name, _ in choices}

        for name, title in choices:
            item = QStandardItem(title)
            item.setData(name, Qt.ItemDataRole.UserRole)
            item.setToolTip(descriptions.get(name, ""))
            self._method_model.appendRow(item)

        for name, title, description in PLANNED_METHODS:
            if name in registered:      # shipped: the registry row wins
                continue
            item = QStandardItem(f"{title} (not yet supported)")
            item.setData(name, Qt.ItemDataRole.UserRole)
            item.setToolTip(description)
            item.setEnabled(False)
            self._method_model.appendRow(item)

    def _current_method_item(self) -> QStandardItem | None:
        """Return the model item behind the current selection, if any."""
        return self._method_model.item(self.method_combo.currentIndex())

    def _update_method_note(self) -> None:
        """Show the selected method's description, or the roadmap warning."""
        item = self._current_method_item()
        if item is None:
            self.lbl_method_note.setText("")
        elif not item.isEnabled():
            self.lbl_method_note.setText(
                "Not yet supported in this release. This entry is shown for "
                "the roadmap only and cannot be selected.")
        else:
            name = self.method_combo.currentData()
            self.lbl_method_note.setText(method_descriptions().get(name, ""))

    def _update_switch_button(self) -> None:
        """Label the one-click toggle with the theme it would switch to."""
        current = self.theme_combo.currentData()
        self.btn_switch_theme.setText(theme_button_text(current))
        self.btn_switch_theme.setToolTip(theme_button_tooltip(current))

    # ------------------------------------------------------------ handlers
    def _on_method_index(self) -> None:
        """Emit a change only for an enabled, registered method."""
        self._update_method_note()
        item = self._current_method_item()
        if item is None or not item.isEnabled():
            return
        name = self.method_combo.currentData()
        if name:
            self.method_changed.emit(str(name))

    def _on_theme_index(self) -> None:
        """Emit the newly selected theme and refresh the toggle button."""
        self._update_switch_button()
        name = self.theme_combo.currentData()
        if name:
            self.theme_changed.emit(str(name))

    def _on_switch_clicked(self) -> None:
        """Flip to the other theme with one click, then notify the window.

        ``set_theme`` blocks the combo's signal; the explicit emit below is
        therefore the single notification for this user action.
        """
        target = toggle_theme(self.theme_combo.currentData())
        self.set_theme(target)
        self.theme_changed.emit(target)

    # --------------------------------------------------------------- API
    def set_method(self, name: str) -> None:
        """Select a method by its stable name without emitting a change."""
        index = self.method_combo.findData(name)
        blocker = QSignalBlocker(self.method_combo)
        self.method_combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker                    # release the block before returning
        self._update_method_note()

    def set_theme(self, name: str) -> None:
        """Select a theme by name without emitting a change."""
        index = self.theme_combo.findData(normalize_theme(name))
        blocker = QSignalBlocker(self.theme_combo)
        self.theme_combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker
        self._update_switch_button()

    def current_method(self) -> str:
        """Stable name of the selected method (never the display title)."""
        return str(self.method_combo.currentData() or "")

    def current_theme(self) -> str:
        """Stable name of the selected theme."""
        return str(self.theme_combo.currentData() or "")

    def method_names(self) -> list[str]:
        """All names visible in the combo, including disabled roadmap rows."""
        return [str(self._method_model.item(row).data(Qt.ItemDataRole.UserRole))
                for row in range(self._method_model.rowCount())]

    def is_method_enabled(self, name: str) -> bool:
        """Whether the row for ``name`` can actually be selected."""
        for row in range(self._method_model.rowCount()):
            item = self._method_model.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == name:
                return item.isEnabled()
        return False
