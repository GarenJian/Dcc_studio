"""DCC Studio main window: a four-step, method-agnostic analysis workflow.

The window owns no numerical code. It collects paths and parameters, starts a
background ``AnalysisWorker``, and presents the resulting files:

    1 Data & Method -- inputs, validation, Settings shortcut, method notes
    2 Parameters    -- every parameter with a small clickable help marker
    3 Compute       -- six live stages, progress bar and run log
    4 Results       -- outputs grouped by workflow category, with previews

Themes live in ``gui.theme``; the list of analysis methods comes from
``dcc.methods``. Registering a new method therefore requires no change here:
its title, tooltip and Settings entry appear automatically.

Global choices are made in ``gui.settings.SettingsDialog``: the analysis
method and the light/dark appearance. The dialog is reachable from the
"Settings" button on page 1 and from the header shortcut, and is owned by
this window so it can stay open while the user browses the other pages.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from PyQt6.QtCore import Qt, QPoint, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QMainWindow, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                             QSizePolicy, QSpinBox, QSplitter, QStackedWidget,
                             QTableWidget, QTableWidgetItem, QTextBrowser,
                             QToolTip, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from dcc.atlas import Atlas
from dcc.methods import method_choices, method_descriptions
from dcc.config import (CONFIG_FILE, DCCConfig, PROJECT_ROOT, RunPaths,
                        load_settings, save_settings)
from dcc.timeseries import discover_samples
from gui.preview import (MAX_PREVIEW_ROWS, PreviewCache, TablePreview,
                         load_pixmap_preview, read_csv_preview,
                         read_text_preview)
from gui.settings import SettingsDialog
from gui.theme import (apply_theme, load_theme, normalize_theme, save_theme,
                       stylesheet)
from gui.workers import AnalysisWorker


METHOD_NOTES = """
<b>How DCC is computed</b><br>
Each sample is a matrix of ROI signals (rows = ROIs, columns = time points),
and the atlas assigns every ROI to a network.<br><br>
1. Average the ROI signals within each network.<br>
2. Correlate each ROI with each network inside a sliding window (SWL, SWS).<br>
3. Apply the chosen sign handling and threshold the correlations at
   <code>mean + k x std</code> to obtain a binary connectivity state per window.<br>
4. Count how often each of the 2<sup>N</sup>-1 non-zero states occurs for each ROI.<br>
5. DCC is the Shannon entropy (bits) of that state-frequency distribution.<br><br>
<b>Inputs</b><br>
Time series: one file, or a folder of files (.mat, .npy, .npz, .csv, .tsv, .txt).
Atlas labels: one integer network index per ROI. Optional: a network-names file,
a sample metadata table (sample_id, group, subject, session) and a numeric
behaviour table for PLS. Group comparisons, ICC and PLS run automatically when
the corresponding metadata is supplied.<br><br>
<b>Outputs</b><br>
Per-sample DCC, group and network summaries, state probabilities and figures,
written as CSV / JSON / PNG under the chosen output directory.
"""

TOOLTIPS = {
    "method": "Analysis method applied to every sample. Choose it in the "
              "Settings dialog on page 1; this read-only field shows the "
              "active method. The Settings list is built from the method "
              "registry, so newly added measures appear there automatically.",
    "swl": "Sliding window length in time points. Must be >= 2 and no longer than "
           "the shortest time series. Longer windows give smoother dFC.",
    "sws": "Step between consecutive windows, in time points. 1 uses every window.",
    "k": "Sparsification threshold: mean + k x std of the windowed correlations. "
         "Larger k keeps only the strongest connections.",
    "sign_type": "How correlation signs are handled before thresholding: abs "
                 "(magnitude), raw (signed), pos (positive part only).",
    "binarize_mode": "global: one threshold over all values; roi: a separate "
                     "threshold for each ROI.",
    "state_prob_threshold": "Minimum mean probability for a connectivity state to "
                            "be reported as high frequency.",
    "top_state_fraction": "Fraction of the most frequent states used to compute "
                          "top-state coverage.",
    "cohen_threshold": "Absolute Cohen's d used for group-comparison counts and "
                       "the dashed lines in effect-size figures.",
    "n_perms": "Number of within-group permutations for PLS significance. "
               "Set to 0 to skip the permutation test.",
    "n_bootstraps": "Number of within-group bootstrap resamples for PLS loading "
                    "stability. Set to 0 to skip bootstrapping.",
    "pls_n_lc": "Number of latent components shown when no component passes FDR.",
    "fdr_q": "False discovery rate level for selecting significant PLS components.",
    "random_seed": "Seed for permutation and bootstrap reproducibility.",
    "norm_img": "PLS normalisation of the DCC matrix: 0 none, 1 global z-score, "
                "2 within-group z-score, 3 global SD, 4 within-group SD.",
    "norm_behav": "PLS normalisation of the behaviour table; same modes as above.",
    "variable": "Name of the variable to read from a .mat container. Leave empty "
                "to use the first data variable.",
    "save_binary_dfc": "Store the binary dFC tensor in each per-sample cache. "
                       "Useful for debugging but much larger on disk.",
}

# Labels shown in the header step bar. The second compute stage name must stay
# in sync with dcc.pipelines.STAGES: the worker emits stage indices, not text.
STEP_LABELS = ("1  Data & Method", "2  Parameters", "3  Compute", "4  Results")
STAGE_LABELS = ("Input", "Compute", "Summary", "Optional analyses", "Figures",
                "Export")
# Result files are grouped by the workflow stage that produced them.
FILE_CATEGORIES = ("Input", "DCC results", "State analysis",
                   "Optional analyses", "Figures", "Run summary")


class HintMarker(QPushButton):
    """Small, low-contrast help marker; hover or click reveals the hint.

    A plain ``QLabel`` would only show a tooltip on hover, which many users
    never discover. A flat ``QPushButton`` keeps the same tiny visual footprint
    but also responds to a click: :meth:`show_hint` calls ``QToolTip.showText``
    so the explanation appears at the marker itself. The button never takes
    keyboard focus and never receives a border or background from the theme.

    ``title`` is the parameter's visible label and ``text`` comes from
    ``TOOLTIPS``. Both are HTML-escaped before being composed into one rich-text
    tooltip, so characters such as ``<`` or ``&`` cannot break the rendering.
    """

    SIZE: int = 12                      # small but still comfortably clickable

    def __init__(self, title: str, text: str, parent: QWidget | None = None):
        """Create the marker, store its rich-text hint and wire the click."""
        super().__init__("!", parent)
        self.setObjectName("paramHint")
        self.setFlat(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._hint: str = f"<b>{escape(title)}</b><br>{escape(text)}"
        self.setToolTip(self._hint)
        self.setAccessibleName(f"Help for {title}")
        self.clicked.connect(self.show_hint)

    def show_hint(self) -> None:
        """Show the hint on click, in addition to the usual hover tooltip."""
        pos = self.mapToGlobal(QPoint(self.width(), self.height()))
        QToolTip.showText(pos, self._hint)


class MainWindow(QMainWindow):
    """Four-step analysis window and the only Qt entry point of the product.

    Responsibilities are grouped into clearly separated helper families:

    * ``_build_ui`` / ``_page_*`` -- construct the header, step bar, four pages
      and the footer. Each parameter is registered through ``_add_param`` so it
      automatically gets a ``HintMarker`` and participates in ``_collect`` /
      ``_sync_from_config``.
    * ``_validate_inputs`` -- check paths and shapes before starting a run.
    * ``_start`` / ``_cancel`` / ``_on_*`` -- drive ``AnalysisWorker`` and map
      its signals to the progress bar, stage labels, log and Results page.
    * ``_refresh_runs`` / ``_refresh_files`` / ``_on_file_selected`` -- browse
      and preview previously produced runs.
    * ``_apply_appearance`` / ``_set_theme`` / ``_open_settings`` -- the
      Settings dialog, method selection and light/dark switching.

    The window never computes anything itself. It reads the method list from
    ``dcc.methods`` and the style from ``gui.theme``, so both can be extended
    without touching this file. ``ui_settings`` is injectable for tests that
    must not write to the real user profile.
    """

    def __init__(self, ui_settings=None):
        """Load settings, apply the saved theme, build the UI and show step 1.

        Order matters: the application-wide theme must be applied before any
        widget is created, and ``_sync_from_config`` must run after the widgets
        exist so their values can be populated from ``config.json``.
        """
        super().__init__()
        self.paths, self.cfg = load_settings()
        self.worker: AnalysisWorker | None = None
        self.current_stage = -1
        self.hint_markers: dict[str, HintMarker] = {}
        self.ui_settings = ui_settings
        # Results previews are deferred and cached. A click only schedules a
        # render; the timer fires after a short quiet period so rapid clicks
        # never queue several full decodes on the GUI thread.
        self._preview_cache = PreviewCache()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._render_pending_preview)
        self._pending_preview: Path | None = None
        self.theme = load_theme(self.ui_settings)
        self.setWindowTitle("DCC Studio  |  Dynamic Connectivity Complexity")
        self.resize(1240, 820)
        self.setMinimumSize(1060, 700)
        self._apply_appearance()
        self._build_ui()
        self._sync_from_config()
        self.set_step(0)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        """Assemble header, step bar, four stacked pages, footer and the theme button."""
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 10)
        outer.setSpacing(10)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("DCC Studio")
        title.setObjectName("title")
        subtitle = QLabel("Dynamic Connectivity Complexity Analysis")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setObjectName("settings")
        self.btn_settings.setToolTip(
            "Choose the analysis method and switch between dark and light.")
        self.btn_settings.clicked.connect(self._open_settings)
        header.addWidget(self.btn_settings)
        outer.addLayout(header)

        steps = QHBoxLayout()
        steps.setSpacing(8)
        self.step_buttons: list[QPushButton] = []
        for i, label in enumerate(STEP_LABELS):
            button = QPushButton(label)
            button.setObjectName("step")
            button.clicked.connect(lambda _, idx=i: self.set_step(idx))
            steps.addWidget(button)
            self.step_buttons.append(button)
        steps.addStretch(1)
        outer.addLayout(steps)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_data())
        self.stack.addWidget(self._page_parameters())
        self.stack.addWidget(self._page_compute())
        self.stack.addWidget(self._page_results())
        outer.addWidget(self.stack, 1)

        footer = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(lambda: self.set_step(self.stack.currentIndex() - 1))
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(lambda: self.set_step(self.stack.currentIndex() + 1))
        self.lbl_footer = QLabel("")
        self.lbl_footer.setObjectName("hint")
        footer.addWidget(self.lbl_footer, 1)
        footer.addWidget(self.btn_back)
        footer.addWidget(self.btn_next)
        outer.addLayout(footer)
        self.setCentralWidget(root)

        # The Settings dialog is created once and kept alive. It is non-modal,
        # so a user can leave it open while navigating the four pages, and
        # tests can drive its widgets without a blocking exec() call.
        self.settings_dialog = SettingsDialog(
            self, method=self.cfg.method, theme=self.theme)
        self.settings_dialog.method_changed.connect(self._set_method)
        self.settings_dialog.theme_changed.connect(self._set_theme)
        self._update_settings_summary()

    # ------------------------------------------------------------ appearance
    def _apply_appearance(self) -> None:
        """Apply the current theme application-wide, or fall back to a window style sheet."""
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            self.setStyleSheet(stylesheet(self.theme))
            return
        apply_theme(app, self.theme)
        self.setStyleSheet("")      # let the application style sheet win

    def _open_settings(self) -> None:
        """Show the shared Settings dialog, seeded with the current choices."""
        self._sync_settings_dialog()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _sync_settings_dialog(self) -> None:
        """Push window state into the dialog without triggering its signals.

        Configuration can change while the dialog is hidden (for example via
        "Restore defaults"), so every opening refreshes both combo boxes.
        """
        self.settings_dialog.set_method(self.cfg.method)
        self.settings_dialog.set_theme(self.theme)

    def _update_settings_summary(self) -> None:
        """Refresh the one-line method/appearance summary shown on page 1."""
        titles = dict(method_choices())
        method = self.cfg.method
        self.lbl_settings.setText(
            f"Method: {titles.get(method, method)}   |   "
            f"Appearance: {self.theme.capitalize()}")

    def _set_method(self, name: str) -> None:
        """Apply a method chosen in Settings to the configuration in memory.

        The read-only Method combo on page 2 remains the single source that
        ``_collect`` reads, so updating it here keeps saving, validation and
        the worker consistent. Like every other parameter, the choice is
        written to disk only by the "Save configuration" button.
        """
        widget = self.param_widgets["method"]
        index = widget.findData(name)
        if index < 0:
            return                     # unknown or disabled roadmap entry
        widget.setCurrentIndex(index)
        try:
            self._collect()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return
        self._update_settings_summary()
        self._append_log(f"Analysis method set to {name}")
        self.statusBar().showMessage(f"Method: {widget.currentText()}", 4000)

    def _set_theme(self, name: str) -> None:
        """Apply a theme chosen in Settings, persist it and repolish widgets."""
        theme = normalize_theme(name)
        if theme == self.theme:
            return                     # no work for a repeated selection
        self.theme = theme
        self._apply_appearance()
        save_theme(self.theme, self.ui_settings)
        self.settings_dialog.set_theme(self.theme)
        for widget in (*self.step_buttons, *self.stage_labels):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        self._update_settings_summary()
        self._append_log(f"Appearance switched to {self.theme} theme")
        self.statusBar().showMessage(f"{self.theme.capitalize()} theme", 4000)

    def _path_row(self, form: QFormLayout, label: str, attribute: str,
                  folder: bool = False, file: bool = True) -> None:
        """Create a labelled file/folder picker row and remember its line edit."""
        edit = QLineEdit()
        edit.setMinimumWidth(320)
        setattr(self, attribute, edit)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        if file:
            browse_file = QPushButton("File...")
            browse_file.clicked.connect(lambda: self._pick_file(edit))
            layout.addWidget(browse_file)
        if folder:
            browse_dir = QPushButton("Folder...")
            browse_dir.clicked.connect(lambda: self._pick_dir(edit))
            layout.addWidget(browse_dir)
        form.addRow(label, row)

    def _page_data(self) -> QWidget:
        """Build the Data & Method page: settings, inputs, validation and notes."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)

        # Page 1 is the landing page, so the global Settings entry sits at the
        # very top: method and appearance are chosen before anything else.
        settings_row = QHBoxLayout()
        self.lbl_settings = QLabel("")
        self.lbl_settings.setObjectName("hint")
        self.lbl_settings.setWordWrap(True)
        settings_row.addWidget(self.lbl_settings, 1)
        self.btn_page_settings = QPushButton("Settings...")
        self.btn_page_settings.setObjectName("primary")
        self.btn_page_settings.setToolTip(
            "Select the analysis method and switch the appearance.")
        self.btn_page_settings.clicked.connect(self._open_settings)
        settings_row.addWidget(self.btn_page_settings)
        layout.addLayout(settings_row)

        columns = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        inputs = QGroupBox("Input data")
        form = QFormLayout(inputs)
        self._path_row(form, "Time series (file or folder)", "ed_data",
                       folder=True, file=True)
        self._path_row(form, "Atlas labels", "ed_atlas")
        self._path_row(form, "Network names (optional)", "ed_atlas_names")
        self._path_row(form, "Sample metadata (optional)", "ed_metadata")
        self._path_row(form, "Behaviour table (optional)", "ed_behaviour")
        self._path_row(form, "Output directory", "ed_output",
                       folder=True, file=False)
        validate = QPushButton("Validate inputs")
        validate.clicked.connect(self._validate_inputs)
        form.addRow(validate)
        self.lbl_check = QLabel("Not validated yet")
        self.lbl_check.setObjectName("hint")
        self.lbl_check.setWordWrap(True)
        form.addRow(self.lbl_check)
        left.addWidget(inputs)

        buttons = QHBoxLayout()
        save = QPushButton("Save configuration")
        save.setObjectName("primary")
        save.clicked.connect(self._save_clicked)
        reset = QPushButton("Restore defaults")
        reset.clicked.connect(self._reset_clicked)
        buttons.addWidget(save)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        left.addLayout(buttons)
        left.addStretch(1)

        notes = QGroupBox("Method notes")
        notes_layout = QVBoxLayout(notes)
        browser = QTextBrowser()
        browser.setHtml(METHOD_NOTES)
        browser.setMinimumHeight(300)
        notes_layout.addWidget(browser)
        right.addWidget(notes, 1)

        columns.addLayout(left, 3)
        columns.addLayout(right, 2)
        layout.addLayout(columns)
        return page

    def _add_param(self, form: QFormLayout, key: str, label: str,
                   widget: QWidget) -> None:
        """Register one parameter widget with its label, tooltip and clickable help marker."""
        self.param_widgets[key] = widget
        hint = HintMarker(label, TOOLTIPS[key])
        self.hint_markers[key] = hint
        widget.setToolTip(TOOLTIPS[key])
        caption = QLabel(label)
        caption.setToolTip(TOOLTIPS[key])
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(caption)
        row_layout.addWidget(hint)
        row_layout.addStretch(1)
        form.addRow(row, widget)

    def _page_parameters(self) -> QWidget:
        """Build the Parameters page: method selector, window/state/optional settings and actions."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        intro = QLabel("Hover or click the small ! beside any parameter for an "
                       "explanation. All parameters are saved to config.json.")
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.param_widgets: dict[str, QWidget] = {}

        columns = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        # This page shows the active method read-only. The editable selector
        # lives in the Settings dialog on page 1; keeping this combo registered
        # as a parameter lets _collect / _sync_from_config stay unchanged.
        methods = QGroupBox("Analysis method (change in Settings)")
        method_form = QFormLayout(methods)
        method = QComboBox()
        descriptions = method_descriptions()
        for name, title in method_choices():
            method.addItem(title, name)
            method.setItemData(method.count() - 1, descriptions.get(name, ""),
                               Qt.ItemDataRole.ToolTipRole)
        self._add_param(method_form, "method", "Method", method)
        method.setEnabled(False)
        method_note = QLabel(
            "Selected in Settings on page 1. Shown here for reference only.")
        method_note.setObjectName("hint")
        method_note.setWordWrap(True)
        method_form.addRow(method_note)
        left.addWidget(methods)

        window = QGroupBox("Dynamic functional connectivity")
        form = QFormLayout(window)
        swl = QSpinBox(); swl.setRange(2, 100000)
        sws = QSpinBox(); sws.setRange(1, 10000)
        k = QDoubleSpinBox(); k.setRange(-3, 5); k.setDecimals(2); k.setSingleStep(0.1)
        sign = QComboBox(); sign.addItems(["abs", "raw", "pos"])
        binarize = QComboBox(); binarize.addItems(["global", "roi"])
        variable = QLineEdit(); variable.setPlaceholderText("optional .mat variable")
        self._add_param(form, "swl", "Window length (SWL)", swl)
        self._add_param(form, "sws", "Window step (SWS)", sws)
        self._add_param(form, "k", "Threshold coefficient k", k)
        self._add_param(form, "sign_type", "Sign handling", sign)
        self._add_param(form, "binarize_mode", "Threshold scope", binarize)
        self._add_param(form, "variable", "MAT variable", variable)
        left.addWidget(window)

        states = QGroupBox("State analysis")
        state_form = QFormLayout(states)
        state_thr = QDoubleSpinBox(); state_thr.setRange(0, 1); state_thr.setDecimals(2)
        state_thr.setSingleStep(0.01)
        top_frac = QDoubleSpinBox(); top_frac.setRange(0.01, 1); top_frac.setDecimals(2)
        top_frac.setSingleStep(0.05)
        cohen = QDoubleSpinBox(); cohen.setRange(0, 3); cohen.setDecimals(2)
        cohen.setSingleStep(0.1)
        self._add_param(state_form, "state_prob_threshold",
                        "High-frequency threshold", state_thr)
        self._add_param(state_form, "top_state_fraction",
                        "Top-state fraction", top_frac)
        self._add_param(state_form, "cohen_threshold",
                        "Cohen's d threshold", cohen)
        left.addWidget(states)
        left.addStretch(1)

        optional = QGroupBox("Optional analyses")
        opt_form = QFormLayout(optional)
        perms = QSpinBox(); perms.setRange(0, 1000000)
        boots = QSpinBox(); boots.setRange(0, 1000000)
        n_lc = QSpinBox(); n_lc.setRange(1, 20)
        fdr = QDoubleSpinBox(); fdr.setRange(0.001, 0.5); fdr.setDecimals(3)
        fdr.setSingleStep(0.01)
        seed = QSpinBox(); seed.setRange(0, 999999)
        norm_img = QSpinBox(); norm_img.setRange(0, 4)
        norm_behav = QSpinBox(); norm_behav.setRange(0, 4)
        self._add_param(opt_form, "n_perms", "PLS permutations", perms)
        self._add_param(opt_form, "n_bootstraps", "PLS bootstraps", boots)
        self._add_param(opt_form, "pls_n_lc", "Latent components to show", n_lc)
        self._add_param(opt_form, "fdr_q", "FDR q", fdr)
        self._add_param(opt_form, "random_seed", "Random seed", seed)
        self._add_param(opt_form, "norm_img", "DCC normalisation mode", norm_img)
        self._add_param(opt_form, "norm_behav", "Behaviour normalisation mode", norm_behav)
        right.addWidget(optional)

        output = QGroupBox("Output")
        out_form = QFormLayout(output)
        cache = QCheckBox("Keep binary dFC in per-sample caches")
        cache.setToolTip(TOOLTIPS["save_binary_dfc"])
        self.param_widgets["save_binary_dfc"] = cache
        hint = HintMarker("Binary dFC cache", TOOLTIPS["save_binary_dfc"])
        self.hint_markers["save_binary_dfc"] = hint
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(cache)
        row_layout.addWidget(hint)
        row_layout.addStretch(1)
        out_form.addRow(row)
        right.addWidget(output)
        right.addStretch(1)

        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        layout.addLayout(columns)
        return page

    def _page_compute(self) -> QWidget:
        """Build the Compute page: six stage labels, progress bar, log and action buttons."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        stages = QGroupBox("Workflow stages")
        stage_layout = QHBoxLayout(stages)
        self.stage_labels: list[QLabel] = []
        for label in STAGE_LABELS:
            item = QLabel(label)
            item.setObjectName("stageIdle")
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stage_layout.addWidget(item)
            self.stage_labels.append(item)
        layout.addWidget(stages)

        progress_group = QGroupBox("Run")
        progress_layout = QVBoxLayout(progress_group)
        row = QHBoxLayout()
        self.lbl_stage = QLabel("Waiting to start")
        self.lbl_stage.setObjectName("hint")
        self.btn_start = QPushButton("Start analysis")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.lbl_stage, 1)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_cancel)
        progress_layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)
        layout.addWidget(progress_group)

        log_group = QGroupBox("Run log")
        log_layout = QVBoxLayout(log_group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log)
        layout.addWidget(log_group, 1)
        return page

    def _page_results(self) -> QWidget:
        """Build the Results page: run list, grouped file tree and preview panel."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        top = QHBoxLayout()
        top.addWidget(QLabel("Run"))
        self.cb_run = QComboBox()
        self.cb_run.currentIndexChanged.connect(lambda: self._refresh_files())
        top.addWidget(self.cb_run, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_runs)
        open_dir = QPushButton("Open output folder")
        open_dir.clicked.connect(self._open_output)
        top.addWidget(refresh)
        top.addWidget(open_dir)
        layout.addLayout(top)

        self.lbl_summary = QLabel("No results yet")
        self.lbl_summary.setObjectName("hint")
        self.lbl_summary.setWordWrap(True)
        layout.addWidget(self.lbl_summary)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.tree_files = QTreeWidget()
        self.tree_files.setHeaderLabels(["Workflow outputs"])
        self.tree_files.setMinimumWidth(300)
        self.tree_files.currentItemChanged.connect(self._on_file_selected)
        split.addWidget(self.tree_files)

        self.preview = QStackedWidget()
        self.page_empty = QLabel("Select an output file to preview")
        self.page_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_empty.setObjectName("hint")
        self.preview.addWidget(self.page_empty)

        self.scroll_img = QScrollArea()
        self.scroll_img.setWidgetResizable(True)
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_img.setWidget(self.lbl_img)
        self.preview.addWidget(self.scroll_img)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # ResizeToContents measures every cell; with a wide result matrix that
        # alone can freeze the window. A fixed default width plus columns the
        # user may drag is both faster and more predictable.
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(110)
        self.preview.addWidget(self.table)

        self.text_view = QTextBrowser()
        self.preview.addWidget(self.text_view)
        split.addWidget(self.preview)
        split.setSizes([320, 840])
        layout.addWidget(split, 1)

        self.lbl_path = QLabel("")
        self.lbl_path.setObjectName("hint")
        layout.addWidget(self.lbl_path)
        return page

    # ------------------------------------------------------------ navigation
    def set_step(self, index: int) -> None:
        """Switch the visible page, highlight the step and refresh results on step 4."""
        index = max(0, min(self.stack.count() - 1, index))
        self.stack.setCurrentIndex(index)
        for i, button in enumerate(self.step_buttons):
            button.setObjectName("stepActive" if i == index else "step")
            button.style().unpolish(button)
            button.style().polish(button)
        self.btn_back.setEnabled(index > 0)
        self.btn_next.setEnabled(index < self.stack.count() - 1)
        self.lbl_footer.setText(f"Step {index + 1} of {len(STEP_LABELS)}")
        if index == 3:
            self._refresh_runs()

    def _pick_file(self, edit: QLineEdit) -> None:
        """Open a file chooser and write the selected path into the given line edit."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a file", edit.text() or str(PROJECT_ROOT),
            "Data files (*.mat *.npy *.npz *.csv *.tsv *.txt *.json);;All files (*)")
        if path:
            edit.setText(path)

    def _pick_dir(self, edit: QLineEdit) -> None:
        """Open a folder chooser and write the selected directory into the given line edit."""
        path = QFileDialog.getExistingDirectory(
            self, "Select a folder", edit.text() or str(PROJECT_ROOT))
        if path:
            edit.setText(path)

    # ----------------------------------------------------------- configuration
    def _sync_from_config(self) -> None:
        """Populate every parameter widget from the currently loaded configuration."""
        self.ed_data.setText(str(self.paths.data_source))
        self.ed_atlas.setText(str(self.paths.atlas_labels))
        self.ed_atlas_names.setText(str(self.paths.atlas_names or ""))
        self.ed_metadata.setText(str(self.paths.metadata or ""))
        self.ed_behaviour.setText(str(self.paths.behaviour or ""))
        self.ed_output.setText(str(self.paths.output_root))
        values = self.cfg.to_dict()
        for key, widget in self.param_widgets.items():
            value = values.get(key)
            if value is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                if key == "method":
                    # Method items store the stable name as item data; their
                    # visible text is a human-readable title.
                    index = widget.findData(str(value))
                    widget.setCurrentIndex(index if index >= 0 else 0)
                else:
                    widget.setCurrentText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
        self._sync_settings_dialog()
        self._update_settings_summary()

    def _collect(self) -> None:
        """Read every parameter widget into ``self.cfg`` and ``self.paths``."""
        self.paths = RunPaths.from_dict({
            "data_source": self.ed_data.text().strip(),
            "atlas_labels": self.ed_atlas.text().strip(),
            "atlas_names": self.ed_atlas_names.text().strip(),
            "metadata": self.ed_metadata.text().strip(),
            "behaviour": self.ed_behaviour.text().strip(),
            "output_root": self.ed_output.text().strip(),
        })
        values = self.cfg.to_dict()
        for key, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = (widget.currentData() if key == "method"
                               else widget.currentText())
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text().strip()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[key] = widget.value()
        self.cfg = DCCConfig.from_dict(values)

    def _save_clicked(self) -> None:
        """Persist current paths and parameters to ``config.json`` with confirmation."""
        try:
            self._collect()
            save_settings(self.paths, self.cfg)
            self._append_log(f"Configuration saved: {CONFIG_FILE}")
            self.statusBar().showMessage("Configuration saved", 4000)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))

    def _reset_clicked(self) -> None:
        """Reload the bundled example settings and refresh all widgets."""
        self.cfg = DCCConfig()
        self.paths = RunPaths()
        self._sync_from_config()
        self._append_log("Defaults restored (not written to disk)")

    def _validate_inputs(self) -> None:
        """Validate paths, atlas and sample discovery; show the outcome on the Data page."""
        try:
            self._collect()
        except Exception as exc:
            self.lbl_check.setText(f"Invalid parameters: {exc}")
            return
        problems = self.paths.check_paths()
        if problems:
            self.lbl_check.setText("\n".join(problems))
            self.statusBar().showMessage("Input validation failed", 4000)
            return
        try:
            atlas = Atlas.load(self.paths.atlas_labels, self.paths.atlas_names)
            samples = discover_samples(self.paths.data_source,
                                       self.cfg.variable or None,
                                       self.paths.metadata)
            if samples[0].data.shape[0] != atlas.n_roi:
                raise ValueError(f"Atlas has {atlas.n_roi} ROIs, data have "
                                 f"{samples[0].data.shape[0]}")
            shortest = min(s.data.shape[1] for s in samples)
            if self.cfg.swl > shortest:
                raise ValueError(f"SWL={self.cfg.swl} exceeds the shortest "
                                 f"series ({shortest} time points)")
            groups = {s.group for s in samples if s.group}
            self.lbl_check.setText(
                f"Valid: {len(samples)} samples | {atlas.n_roi} ROIs | "
                f"{atlas.n_net} networks | {atlas.n_states} states | "
                f"shortest series {shortest} | groups {len(groups) or 'none'}")
            self.statusBar().showMessage("Input validation passed", 4000)
        except Exception as exc:
            self.lbl_check.setText(f"Validation error: {exc}")
            self.statusBar().showMessage("Input validation failed", 4000)

    # ----------------------------------------------------------------- run
    def _append_log(self, msg: str) -> None:
        """Append one timestamped line to the Compute page log."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {msg}")

    @pyqtSlot(str, int, int)
    def _on_stage(self, stage: str, index: int, total: int) -> None:
        """Highlight the current workflow stage and reset the progress bar."""
        self.current_stage = index - 1
        for i, label in enumerate(self.stage_labels):
            state = ("stageDone" if i < self.current_stage else
                     "stageActive" if i == self.current_stage else "stageIdle")
            label.setObjectName(state)
            label.style().unpolish(label)
            label.style().polish(label)
        self.lbl_stage.setText(f"Stage {index}/{total}: {stage}")
        self._append_log(f"Stage {index}/{total}: {stage}")

    @pyqtSlot(str, int, int)
    def _on_progress(self, stage: str, done: int, total: int) -> None:
        """Update the progress bar and label from a worker progress signal."""
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if done:
            self.lbl_stage.setText(f"{stage}: {done}/{total}")

    @pyqtSlot(object)
    def _on_finished(self, result) -> None:
        """Store the RunResult, re-enable controls and refresh the Results page."""
        self._set_running(False)
        self.progress.setValue(self.progress.maximum())
        self.lbl_stage.setText("Completed")
        for label in self.stage_labels:
            label.setObjectName("stageDone")
            label.style().unpolish(label)
            label.style().polish(label)
        self._append_log(f"Completed: {len(result.figures)} figures, "
                         f"{len(result.files)} files -> {result.run_dir}")
        self.statusBar().showMessage("Analysis completed", 8000)
        self._refresh_runs()
        target = self.cb_run.findText(result.tag)
        if target >= 0:
            self.cb_run.setCurrentIndex(target)
        self.set_step(3)

    @pyqtSlot(str)
    def _on_error(self, text: str) -> None:
        """Show a worker error in the log and a message box, then restore the controls."""
        self._set_running(False)
        self.lbl_stage.setText("Failed / cancelled")
        self._append_log("Error: " + text)
        QMessageBox.critical(self, "Analysis failed", text[:1500])

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        """Report a cancelled run and restore the idle UI state."""
        self._set_running(False)
        self.lbl_stage.setText("Cancelled")
        self._append_log("Analysis cancelled; completed samples remain cached")

    def _set_running(self, running: bool) -> None:
        """Enable or disable the start, cancel and navigation controls as a group."""
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        if running:
            self.progress.setValue(0)

    def _cancel(self) -> None:
        """Ask the running worker to stop cooperatively at the next sample boundary."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._append_log("Cancellation requested...")

    def _start(self) -> None:
        """Collect and validate inputs, then create and start the background worker."""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Already running",
                                    "An analysis is already in progress.")
            return
        try:
            self._collect()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            return
        problems = self.paths.check_paths()
        if problems:
            QMessageBox.warning(self, "Missing inputs", "\n".join(problems))
            return
        save_settings(self.paths, self.cfg)
        self.current_stage = -1
        for label in self.stage_labels:
            label.setObjectName("stageIdle")
            label.style().unpolish(label)
            label.style().polish(label)
        self._append_log(f"Starting run: {self.cfg.run_tag()}")
        self.worker = AnalysisWorker(self.paths, self.cfg, self)
        self.worker.sig_log.connect(self._append_log)
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_stage.connect(self._on_stage)
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.sig_error.connect(self._on_error)
        self.worker.sig_cancelled.connect(self._on_cancelled)
        self._set_running(True)
        self.worker.start()

    # ------------------------------------------------------------- results
    def _output_root(self) -> Path:
        """Return the configured output directory as an absolute Path."""
        path = Path(self.ed_output.text().strip())
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    def _current_run_dir(self) -> Path | None:
        """Return the result directory for the current parameters, or None if absent."""
        tag = self.cb_run.currentText()
        return self._output_root() / tag if tag else None

    def _refresh_runs(self) -> None:
        """Populate the run selector with existing result directories."""
        root = self._output_root()
        runs = sorted((p.name for p in root.iterdir()
                       if p.is_dir() and (p / "summary.json").exists()),
                      reverse=True) if root.exists() else []
        self.cb_run.blockSignals(True)
        self.cb_run.clear()
        self.cb_run.addItems(runs)
        self.cb_run.blockSignals(False)
        self._refresh_files()

    @staticmethod
    def _category(path: Path) -> str:
        """Classify a result file into one of the workflow categories in the tree."""
        name = path.name.lower()
        if "figures" in path.parts:
            return "Figures"
        if name in ("group_summary.csv", "network_summary.csv", "dcc_matrix.csv",
                    "dcc_per_sample.csv"):
            return "DCC results"
        if name.startswith(("comparison_", "icc", "group_", "pls_")):
            return "Optional analyses"
        if name.startswith(("state_", "high_", "network_state", "top_",
                            "sample_similarity")):
            return "State analysis"
        if name == "run_manifest.json":
            return "Input"
        return "Run summary"

    def _refresh_files(self) -> None:
        """Fill the grouped file tree for the currently selected run."""
        # A pending preview belongs to the previous selection; cancel it so a
        # run switch or Refresh cannot render a stale file after the rebuild.
        self._preview_timer.stop()
        self._pending_preview = None
        self.tree_files.clear()
        run_dir = self._current_run_dir()
        if not run_dir or not run_dir.exists():
            self.preview.setCurrentIndex(0)
            self.lbl_summary.setText("No results yet")
            self.lbl_path.setText("Run an analysis to see outputs here")
            return
        info_path = run_dir / "summary.json"
        if info_path.exists():
            import json
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                self.lbl_summary.setText(
                    f"Samples: {info.get('n_samples')} | ROIs: {info.get('n_roi')} | "
                    f"Networks: {info.get('n_networks')} | States: {info.get('n_states')} | "
                    f"Mean DCC: {info.get('mean_dcc', 0):.3f} | "
                    f"Analyses: {', '.join(info.get('analyses', [])) or 'core DCC'}")
            except Exception:
                self.lbl_summary.setText("Run summary available")
        extensions = (".png", ".csv", ".json", ".npy", ".txt", ".md")
        files = sorted(p for p in run_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in extensions
                       and "samples" not in p.parts)
        grouped: dict[str, list[Path]] = {c: [] for c in FILE_CATEGORIES}
        for path in files:
            grouped[self._category(path)].append(path)
        for category in FILE_CATEGORIES:
            items = grouped[category]
            if not items:
                continue
            parent = QTreeWidgetItem(self.tree_files,
                                     [f"{category} ({len(items)})"])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for path in items:
                child = QTreeWidgetItem(parent,
                                        [str(path.relative_to(run_dir))])
                child.setData(0, Qt.ItemDataRole.UserRole, str(path))
        self.tree_files.expandAll()
        self.lbl_path.setText(f"{len(files)} output files | {run_dir}")
        if not files:
            self.preview.setCurrentIndex(0)

    def _on_file_selected(self, current, previous) -> None:
        """Schedule a preview instead of decoding the file on the click itself.

        Previously every selection synchronously read and rendered the whole
        result, so clicking quickly through a run with wide CSVs or large
        figures froze the GUI. Only the last selection is rendered after a
        short quiet period. The path label updates immediately, so the panel
        still responds to each click without doing any real work.
        """
        self._preview_timer.stop()
        self._pending_preview = None
        if current is None or not current.data(0, Qt.ItemDataRole.UserRole):
            return
        path = Path(current.data(0, Qt.ItemDataRole.UserRole))
        self.lbl_path.setText(str(path))
        self._pending_preview = path
        self._preview_timer.start()

    @pyqtSlot()
    def _render_pending_preview(self) -> None:
        """Render the last selected file once the debounce window elapsed.

        All readers are bounded and cached (``gui.preview``), so a repeated
        click on the same file costs nothing. Any exception is shown in the
        text panel rather than escaping into Qt's event loop.
        """
        path = self._pending_preview
        self._pending_preview = None
        if path is None:
            return
        # The selection may have moved on while the timer was running (a run
        # switch, Refresh or a page change). Never render a stale request.
        item = self.tree_files.currentItem()
        selected = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if selected != str(path):
            return
        suffix = path.suffix.lower()
        try:
            if suffix == ".png":
                pixmap = self._preview_cache.get(path, load_pixmap_preview)
                if pixmap.isNull():
                    self.lbl_img.setText("Cannot preview this image")
                else:
                    self.lbl_img.setText("")
                    self.lbl_img.setPixmap(pixmap)
                self.preview.setCurrentIndex(1)
            elif suffix == ".csv":
                data = self._preview_cache.get(path, read_csv_preview)
                self._fill_table(data)
                self.lbl_path.setText(self._preview_caption(path, data))
                self.preview.setCurrentIndex(2)
            elif suffix in (".json", ".txt", ".md"):
                self.text_view.setPlainText(
                    self._preview_cache.get(path, read_text_preview))
                self.preview.setCurrentIndex(3)
            else:
                self.text_view.setPlainText(
                    f"Binary output\n\nPath: {path}\n"
                    f"Size: {path.stat().st_size / 1024:.1f} KB\n\n"
                    "Use numpy.load to inspect this array.")
                self.preview.setCurrentIndex(3)
        except Exception as exc:
            self.text_view.setPlainText(
                f"Cannot preview {path.name}\n\n{exc}")
            self.preview.setCurrentIndex(3)

    def _fill_table(self, data: TablePreview) -> None:
        """Fill the CSV grid in one batched, signal-free pass.

        Suspending repaints and signals while inserting the (already bounded)
        cells avoids one layout/redraw per cell, which was the main source of
        the delay on wide result matrices.
        """
        table = self.table
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        try:
            table.clear()
            table.setRowCount(len(data.rows))
            table.setColumnCount(len(data.headers))
            table.setHorizontalHeaderLabels(data.headers)
            for i, row in enumerate(data.rows):
                for j, value in enumerate(row):
                    table.setItem(i, j, QTableWidgetItem(value))
        finally:
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

    @staticmethod
    def _preview_caption(path: Path, data: TablePreview) -> str:
        """Say when a CSV preview is truncated, without hiding the full path."""
        notes = []
        if data.truncated_rows:
            notes.append(f"first {MAX_PREVIEW_ROWS} rows")
        if data.truncated_columns:
            notes.append(f"first {data.shown_columns} of "
                         f"{data.total_columns} columns")
        suffix = f"  |  preview shows {', '.join(notes)}" if notes else ""
        return f"{path}{suffix}"

    def _open_output(self) -> None:
        """Open the current result directory in the operating system file manager."""
        directory = self._current_run_dir() or self._output_root()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
