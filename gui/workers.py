"""Background thread running the analysis workflow without blocking the GUI.

Qt requires all widget updates to happen on the main thread, while a full run
can take minutes. :class:`AnalysisWorker` therefore owns a ``QThread`` and
translates the plain callables of ``dcc.pipelines.Callbacks`` into Qt signals:

=============================  ==============================================
Signal                         Carries
=============================  ==============================================
``sig_log``                    one log line
``sig_progress``               stage label, done, total
``sig_stage``                  stage label, 1-based index, stage count
``sig_finished``               the ``RunResult`` on success
``sig_error``                  formatted error text
``sig_cancelled``              user interruption
=============================  ==============================================

Cancellation is cooperative: ``Callbacks.cancelled`` is polled between samples
and raises ``AnalysisCancelled`` inside the workflow. That means an in-flight
sample finishes, already cached samples are kept, and no partial result file is
written. Errors are caught and formatted with a traceback so the Compute page
can show the real cause instead of a silent failure.
"""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from dcc.config import DCCConfig, RunPaths
from dcc.pipelines import AnalysisCancelled, Callbacks, run_analysis


class AnalysisWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_progress = pyqtSignal(str, int, int)
    sig_stage = pyqtSignal(str, int, int)
    sig_finished = pyqtSignal(object)
    sig_error = pyqtSignal(str)
    sig_cancelled = pyqtSignal()

    def __init__(self, paths: RunPaths, cfg: DCCConfig, parent=None):
        """Store paths and configuration for the run; initialise the cooperative cancel flag."""
        super().__init__(parent)
        self.paths = paths
        self.cfg = cfg

    def cancel(self) -> None:
        """Set the cancel flag; the workflow notices it between two samples."""
        self.requestInterruption()

    def run(self) -> None:
        """Execute ``run_analysis`` with signal-emitting callbacks and report its outcome."""
        cb = Callbacks(
            log=self.sig_log.emit,
            progress=lambda stage, done, total: self.sig_progress.emit(
                stage, done, total),
            stage=lambda stage, index, total: self.sig_stage.emit(
                stage, index, total),
            cancelled=self.isInterruptionRequested,
        )
        try:
            result = run_analysis(self.paths, self.cfg, cb)
        except AnalysisCancelled:
            self.sig_cancelled.emit()
        except Exception as exc:
            self.sig_error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        else:
            self.sig_finished.emit(result)
