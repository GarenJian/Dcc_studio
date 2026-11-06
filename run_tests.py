"""One-click test runner with a complete, reproducible test report.

This script is the single entry point for verification. It

1. discovers **every** ``unittest`` case under ``tests/`` (unit tests, IO
   tests, end-to-end workflow tests, GUI/theme tests and the committed
   reference-regression cases);
2. executes them under ``coverage.py`` with branch measurement enabled;
3. computes **statement**, **branch** and **function** coverage. Branch
   coverage comes from coverage.py; function coverage is derived here by
   walking the AST of the production packages and checking whether at least
   one executable line of each function was executed;
4. writes a self-contained Markdown report plus machine-readable artefacts.

The report is deliberately asymmetric. Functional cases (comparisons against
the committed reference results) are documented in full: which dataset was
used, which reference file holds the expected value, the stored result, the
computed result, the observed deviation and the tolerance. Unit tests and
coverage are procedural checks, so they are reported as short summaries;
their machine-readable detail is kept in the artefacts instead.

Generated artefacts (all under ``reports/`` by default)
======================================================  ========================
``test-report.md``           Human-readable report (open this first)
``functional-details.json``  Every functional comparison with values/deviations
``test-output.log``          Raw stdout/stderr produced by the tests
``coverage.json``            coverage.py data, including missing branches
``coverage.xml``             Cobertura XML for CI systems
``htmlcov/index.html``       Line-by-line browsable coverage
``.coverage``                Raw coverage database
======================================================  ========================

Coverage scope: the production packages ``dcc/``, ``gui/`` and ``packaging/``.
Tests themselves, the copied reference data and the thin GUI entry point
``run_gui.py`` are intentionally excluded: including them would inflate the
numbers with test-only branches. The report states this scope explicitly.

This is the only test entry point. The former ``run_tests.ps1`` and
``run_tests.cmd`` wrappers were removed: a plain Python script is not subject
to the Windows PowerShell execution policy, and it now installs its own
dependencies, so no shell wrapper is needed.

Usage::

    python run_tests.py                 # install deps, run all tests, full report
    python run_tests.py --no-install    # skip dependency installation (offline)
    python run_tests.py --quick         # Markdown + JSON only (same as --no-html)
    python run_tests.py --open           # open the HTML coverage browser afterwards
    python run_tests.py --report-dir out

The old PowerShell-style switches (``-NoInstall``, ``-Quick``, ``-Open``) are
accepted as aliases. Exit code is 0 only when every collected test passed or
was skipped.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import unittest
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                          # keep coverage paths relative and stable
TEST_DIR = ROOT / "tests"
FIXTURE_DIR = TEST_DIR / "fixtures"

# Production code that the coverage numbers describe. Entry-point scripts are
# excluded on purpose; see the module docstring for the rationale.
SOURCE_DIRS = (ROOT / "dcc", ROOT / "gui", ROOT / "packaging")

try:
    import coverage as coverage_module
except ImportError:                     # pragma: no cover - environment issue
    coverage_module = None


@dataclass
class CaseRecord:
    """Outcome of exactly one ``unittest`` test method."""

    name: str
    status: str = "passed"
    seconds: float = 0.0
    message: str = ""
    traceback: str = ""


@dataclass
class SuiteResult:
    """Aggregated outcome of the whole discovery run."""

    cases: list[CaseRecord] = field(default_factory=list)
    started: float = 0.0
    duration: float = 0.0
    collected: int = 0

    def count(self, status: str) -> int:
        """Number of cases with the given status."""
        return sum(1 for case in self.cases if case.status == status)

    @property
    def ok(self) -> bool:
        """True when nothing failed and no unexpected error occurred."""
        return self.count("failed") == 0 and self.count("error") == 0


class DetailedTestResult(unittest.TestResult):
    """``TestResult`` that records status, timing and failure text per test.

    The stock ``TextTestRunner`` only prints dots. Per-case records let the
    report produce an accurate pass/fail summary and, when something breaks,
    attach the full traceback without expanding successful procedural cases.
    """

    def __init__(self):
        super().__init__()
        self.records: list[CaseRecord] = []
        self._active: dict[str, tuple[CaseRecord, float]] = {}

    def startTest(self, test) -> None:
        """Remember the start timestamp so ``stopTest`` can measure duration."""
        super().startTest(test)
        record = CaseRecord(test.id())
        self.records.append(record)
        self._active[test.id()] = (record, time.perf_counter())

    def stopTest(self, test) -> None:
        """Finalise the elapsed time of the test that just ended."""
        entry = self._active.pop(test.id(), None)
        if entry:
            record, started = entry
            record.seconds = time.perf_counter() - started
        super().stopTest(test)

    def _fill(self, test, status: str, err=None, reason: str = "") -> None:
        """Attach a failure/error/skip description to the current record."""
        record = self._active.get(test.id(), (None,))[0]
        if record is None:
            for item in reversed(self.records):
                if item.name == test.id():
                    record = item
                    break
        if record is None:
            return
        record.status = status
        if err is not None:
            exc_type, exc_value, exc_tb = err
            record.message = f"{exc_type.__name__}: {exc_value}"
            record.traceback = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            record.message = reason

    def addSuccess(self, test) -> None:
        super().addSuccess(test)

    def addError(self, test, err) -> None:
        super().addError(test, err)
        self._fill(test, "error", err=err)

    def addFailure(self, test, err) -> None:
        super().addFailure(test, err)
        self._fill(test, "failed", err=err)

    def addSkip(self, test, reason) -> None:
        super().addSkip(test, reason)
        self._fill(test, "skipped", reason=str(reason))


@dataclass(frozen=True)
class FunctionInfo:
    """One function/method definition found by the AST walk."""

    qualified_name: str
    lineno: int
    body_start: int
    end_lineno: int


class _FunctionCollector(ast.NodeVisitor):
    """Collect every function definition with its enclosing class names.

    Nested functions are included (they are real branches of behaviour), and
    their qualified name uses dotted scope notation, e.g.
    ``Outer.method.<locals>.helper``. Only the body range is used for the
    coverage decision: a ``def`` line itself is executed on import even when
    the function is never called.
    """

    def __init__(self):
        self.items: list[FunctionInfo] = []
        self._scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _visit_function(self, node) -> None:
        name = ".".join(self._scope + [node.name])
        body_start = node.body[0].lineno if node.body else node.lineno
        self.items.append(FunctionInfo(name, node.lineno, body_start,
                                       node.end_lineno or node.lineno))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def _collect_functions() -> dict[str, list[FunctionInfo]]:
    """Parse all production modules and return ``{relative_path: functions}``."""
    collected: dict[str, list[FunctionInfo]] = {}
    for directory in SOURCE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = _FunctionCollector()
            visitor.visit(tree)
            collected[relative] = visitor.items
    return collected


def _percent(covered: int, total: int) -> float:
    """Safe percentage helper (returns 100.0 when there is nothing to cover)."""
    return 100.0 if total == 0 else 100.0 * covered / total


def _fixture_role(relative: str) -> str:
    """Describe what one copied reference file contributes to the test set.

    Roles are derived from the file name only. The reference folder has a
    stable naming convention, so this keeps the dataset table readable without
    maintaining a separate manifest that could drift out of sync with the files.
    """
    name = Path(relative).name
    if name == "ROI_idx.txt":
        return "ROI-to-network label vector"
    if name == "7Net_200.csv":
        return "Network names"
    if name == "TS_data.mat":
        return "Time-series input (200 x 818 x 90)"
    if name == "Ts_data.mat":
        return "Time-series input (224 samples)"
    if name == "node_flexibility.mat":
        return "Per-sample entropy / states / binary dFC"
    if name == "group_statis.mat":
        return "Group statistics"
    if name.endswith("_group_statis.mat"):
        return "Group DCC matrices for t tests"
    if name == "Post_vs_Pre_res.mat":
        return "Published t statistics and Cohen's d"
    if name == "PLSresults.mat":
        return "PLS inputs, SVD components and loadings"
    return "Reference data"


def _fixture_summary() -> dict:
    """Inventory every copied reference file for the functional dataset table.

    The fixtures are verbatim copies of the reference folder, not generated
    slices, so there is no manifest to parse. Listing all files explicitly lets
    the report state exactly which dataset each functional case ran on.
    """
    if not FIXTURE_DIR.exists():
        return {"available": False, "files": 0, "bytes": 0,
                "groups": [], "file_list": []}
    files = [path for path in sorted(FIXTURE_DIR.rglob("*")) if path.is_file()]
    groups: dict[str, int] = {}
    file_list: list[tuple[str, int, str]] = []
    for path in files:
        relative = path.relative_to(FIXTURE_DIR).as_posix()
        group = relative.split("/")[0]
        groups[group] = groups.get(group, 0) + 1
        file_list.append((relative, path.stat().st_size,
                          _fixture_role(relative)))
    return {
        "available": bool(files),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "groups": sorted(groups.items()),
        "file_list": file_list,
    }


def _load_functional_records() -> list[dict]:
    """Read the structured functional comparisons written by the test suite.

    ``tests/functional_report.py`` flushes one entry per numerical comparison
    in the regression suite's ``tearDownClass``. An absent or malformed file is
    not an error: the report then says plainly that no records were produced.
    """
    path = _REPORT_DIR / "functional-details.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    entries = payload.get("entries", [])
    return entries if isinstance(entries, list) else []


def _run_suite() -> tuple[SuiteResult, Path]:
    """Discover and execute all tests, capturing their raw output to a log."""
    report_dir = _REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(TEST_DIR), pattern="test*.py",
                            top_level_dir=str(ROOT))
    result = DetailedTestResult()
    started = time.perf_counter()
    log_path = report_dir / "test-output.log"
    with open(log_path, "w", encoding="utf-8") as log_file, \
            contextlib.redirect_stdout(log_file), \
            contextlib.redirect_stderr(log_file):
        # TestSuite.run returns the TestResult object it was given; we ignore
        # that return value and keep our own record list from DetailedTestResult.
        suite.run(result)
    duration = time.perf_counter() - started
    return SuiteResult(cases=result.records, started=started,
                       duration=duration,
                       collected=suite.countTestCases()), log_path


def _coverage_data(functions: dict[str, list[FunctionInfo]],
                   write_html: bool) -> tuple[dict, dict[str, dict],
                                             SuiteResult]:
    """Return ``(totals, per_file, suite_result)`` coverage facts.

    ``per_file`` merges the coverage.py JSON summary with the AST-derived
    function counts, so the Markdown table can show all three metrics per
    module. Missing statements/branches are reported as counts and as short
    line lists to keep the report readable.
    """
    cov = coverage_module.Coverage(
        branch=True,
        source=[str(directory) for directory in SOURCE_DIRS],
        omit=["*/__pycache__/*"],
        data_file=str(_REPORT_DIR / ".coverage"),
    )
    cov.start()
    try:
        suite_result, _ = _run_suite()
    finally:
        cov.stop()
        cov.save()

    json_path = _REPORT_DIR / "coverage.json"
    cov.json_report(outfile=str(json_path), pretty_print=True)
    if write_html:
        cov.xml_report(outfile=str(_REPORT_DIR / "coverage.xml"))
        cov.html_report(directory=str(_REPORT_DIR / "htmlcov"))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    totals = payload.get("totals", {})
    # coverage.py may key files by absolute or relative paths depending on its
    # version/configuration. Normalise both to project-relative POSIX paths so
    # they match the keys produced by the AST walk.
    files: dict[str, dict] = {}
    for key, value in payload.get("files", {}).items():
        try:
            relative = Path(key).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            relative = Path(key).as_posix()
        files[relative] = value

    per_file: dict[str, dict] = {}
    for relative, funcs in functions.items():
        entry = files.get(relative, {})
        summary = entry.get("summary", {})
        executed = set(entry.get("executed_lines", []))
        covered_funcs = [
            func for func in funcs
            if any(line in executed
                   for line in range(func.body_start, func.end_lineno + 1))
        ]
        missing_funcs = [func.qualified_name for func in funcs
                         if func not in covered_funcs]
        per_file[relative] = {
            "statements": int(summary.get("num_statements", 0)),
            "statements_covered": int(summary.get("covered_lines", 0)),
            "missing_lines": entry.get("missing_lines", []),
            "branches": int(summary.get("num_branches", 0)),
            "branches_covered": int(summary.get("covered_branches", 0)),
            "missing_branches": entry.get("missing_branches", []),
            "functions": len(funcs),
            "functions_covered": len(covered_funcs),
            "missing_functions": missing_funcs,
        }

    grand = {
        "statements": int(totals.get("num_statements", 0)),
        "statements_covered": int(totals.get("covered_lines", 0)),
        "branches": int(totals.get("num_branches", 0)),
        "branches_covered": int(totals.get("covered_branches", 0)),
        "functions": sum(item["functions"] for item in per_file.values()),
        "functions_covered": sum(item["functions_covered"]
                                 for item in per_file.values()),
    }
    return grand, per_file, suite_result


def _fmt(value, spec: str = ".8g") -> str:
    """Format a numeric table cell, tolerating non-finite measurements."""
    return "n/a" if value is None else format(value, spec)


def _format_stats(stats: dict) -> str:
    """Render min / max / mean of one array as a single compact table cell."""
    if stats.get("min") is None:
        return "n/a"
    return (f"{_fmt(stats['min'], '.6g')} / "
            f"{_fmt(stats['max'], '.6g')} / "
            f"{_fmt(stats['mean'], '.6g')}")


def _functional_chapter(add, fixture: dict, functional: list[dict]) -> None:
    """Write the detailed chapter describing every functional comparison.

    Each recorded comparison gets a row with the stored reference statistics
    and the freshly computed statistics. One comparison per case is also
    expanded into concrete element-by-element values, so a reader sees actual
    numbers rather than only aggregates. ``add`` is the bound ``list.append``
    supplied by the caller, which keeps this helper free of report-wide state.
    """
    add("## 1. Functional verification on the reference dataset")
    add("")
    add("This chapter reports every numerical comparison against the committed "
        "reference results. **Reference result** is the value stored in the "
        "copied result file; **Computed result** is what this run produced. "
        "Each functional case states the dataset it used, the reference file, "
        "the quantities compared and the observed deviation.")
    add("")

    add("### 1.1 Dataset under test")
    add("")
    if fixture["available"]:
        add(f"- Copied reference files: {fixture['files']} "
            f"({fixture['bytes'] / 1048576:.2f} MB total)")
        add("- Groups: " + ", ".join(
            f"`{name}` ({count})" for name, count in fixture["groups"]))
        add("- Provenance: verbatim copies of the reference project's input "
            "data and published results under `tests/fixtures/`; no generation "
            "step and no external folder or data path to configure.")
        add("- Comparison tolerances: DCC 1e-5, state probabilities 1e-6, "
            "statistics and PLS 1e-8; exact comparisons use 0.")
        add("")
        add("| Reference file | Size (MB) | Role |")
        add("| --- | ---: | --- |")
        for relative, size, role in fixture["file_list"]:
            add(f"| `{relative}` | {size / 1048576:.2f} | {role} |")
    else:
        add("- **Missing.** Download the reference dataset into "
            "`tests/fixtures/` (see `docs/test.md` for the download link); "
            "`tests/test_reference_parity.py` lists the required files.")
    add("")

    # Group comparisons by case while preserving the order in which they ran.
    cases: dict[str, list[dict]] = {}
    for entry in functional:
        cases.setdefault(entry["case"], []).append(entry)

    for number, (case, entries) in enumerate(cases.items(), start=2):
        add(f"### 1.{number} Functional case: `{case}`")
        add("")
        datasets = sorted({entry["dataset"] for entry in entries})
        sources = sorted({entry["source"] for entry in entries})
        add("- **Dataset**:")
        for item in datasets:
            add(f"  - `{item}`")
        add("- **Reference file(s)**:")
        for item in sources:
            add(f"  - `{item}`")
        add(f"- **Comparisons**: {len(entries)}")
        add("")
        add("| Quantity | Elements | Array shape | Reference result "
            "(min / max / mean) | Computed result (min / max / mean) | "
            "Max abs diff | Tolerance | Result |")
        add("| --- | ---: | --- | --- | --- | ---: | ---: | --- |")
        for entry in entries:
            shape = "x".join(str(v) for v in entry["shape"]) or "scalar"
            add(f"| {entry['quantity']} | {entry['elements']} | `{shape}` | "
                f"{_format_stats(entry['expected_stats'])} | "
                f"{_format_stats(entry['actual_stats'])} | "
                f"{_fmt(entry['max_abs_diff'], '.3g')} | "
                f"{_fmt(entry['tolerance'], '.3g')} | "
                f"{'PASS' if entry['passed'] else 'FAIL'} |")
        add("")

        # Expand the largest comparison into concrete values: the leading
        # elements plus the position with the greatest deviation.
        representative = max(entries, key=lambda item: item["elements"])
        add(f"**Representative element-by-element values** "
            f"(`{representative['quantity']}`, "
            f"{representative['elements']} elements):")
        add("")
        add("| Element index | Reference value | Computed value | Abs diff |")
        add("| ---: | ---: | ---: | ---: |")
        for sample in representative["samples"]:
            add(f"| {sample['index']} | "
                f"{_fmt(sample['expected'])} | "
                f"{_fmt(sample['actual'])} | "
                f"{_fmt(sample['diff'], '.3g')} |")
        add("")
        notes = sorted({entry["note"] for entry in entries if entry["note"]})
        if notes:
            add("- Notes: " + "; ".join(f"`{item}`" for item in notes))
            add("")

    add(f"### 1.{len(cases) + 2} Functional summary")
    add("")
    if functional:
        worst = max(functional, key=lambda item: item["max_abs_diff"] or 0.0)
        failed = [item for item in functional if not item["passed"]]
        add(f"- Functional cases: {len(cases)}; numerical comparisons: "
            f"{len(functional)}; failed comparisons: {len(failed)}")
        add(f"- Array elements compared: "
            f"{sum(item['elements'] for item in functional):,}; elements "
            f"outside tolerance: "
            f"{sum(item['mismatches'] for item in functional):,}")
        add(f"- Largest observed deviation: "
            f"`{_fmt(worst['max_abs_diff'], '.3g')}` in `{worst['case']}` / "
            f"`{worst['quantity']}` "
            f"(tolerance `{_fmt(worst['tolerance'], '.3g')}`)")
        add("- Overall functional verdict: "
            f"**{'PASS' if not failed else 'FAIL'}**")
    else:
        add("- No functional comparison records were produced. Run the "
            "reference parity suite to populate this chapter.")
    add("")


def _write_report(path: Path, suite: SuiteResult, grand: dict,
                  fixture: dict, functional: list[dict],
                  write_html: bool) -> None:
    """Render the report: rich functional detail, concise procedural summaries.

    Functional cases are documented per comparison (dataset, reference values,
    computed values, deviation). Unit tests and coverage are reported as
    summaries only; their machine-readable detail remains in the artefacts.
    """
    lines: list[str] = []
    add = lines.append
    add("# DCC Studio test report")
    add("")
    add(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- Command: `{sys.executable} run_tests.py`")
    add(f"- Python: {platform.python_version()} "
        f"({platform.system()} {platform.machine()})")
    add(f"- Overall result: **{'PASS' if suite.ok else 'FAIL'}**")
    add("")

    _functional_chapter(add, fixture, functional)

    # --------------------- Programmatic tests: summary only, no per-case rows
    add("## 2. Programmatic tests (summary)")
    add("")
    add("Unit, IO, workflow, GUI/theme and method-registry tests. Individual "
        "steps are intentionally not listed here; the raw log is in "
        "`reports/test-output.log`.")
    add("")
    add("| Collected | Passed | Failed | Errors | Skipped | Duration |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: |")
    add(f"| {suite.collected} | {suite.count('passed')} | "
        f"{suite.count('failed')} | {suite.count('error')} | "
        f"{suite.count('skipped')} | {suite.duration:.2f} s |")
    add("")
    failures = [case for case in suite.cases
                if case.status in ("failed", "error")]
    if failures:
        add("### Failure diagnostics")
        add("")
        for case in failures:
            add(f"#### `{case.name}`")
            add("")
            add("```text")
            add(case.traceback.rstrip()[:4000])
            add("```")
            add("")

    # --------------------- Coverage: three numbers only, no per-module tables
    add("## 3. Coverage (summary)")
    add("")
    add("| Statements | Branches | Functions |")
    add("| ---: | ---: | ---: |")
    add(f"| {_percent(grand['statements_covered'], grand['statements']):.1f}% "
        f"({grand['statements_covered']}/{grand['statements']}) | "
        f"{_percent(grand['branches_covered'], grand['branches']):.1f}% "
        f"({grand['branches_covered']}/{grand['branches']}) | "
        f"{_percent(grand['functions_covered'], grand['functions']):.1f}% "
        f"({grand['functions_covered']}/{grand['functions']}) |")
    add("")
    add("- Statements: executed source lines. Branches: both outcomes of "
        "conditional / loop arcs (`branch=True`). Functions: definitions whose "
        "body was entered (AST-derived).")
    add("- Scope: `dcc/`, `gui/`, `packaging/`. Test modules, reference data "
        "and thin launchers are excluded. Per-module and per-function detail "
        "is available in the artefacts below.")
    add("")

    # --------------------- Artefacts
    add("## 4. Artefacts")
    add("")
    add("| File | Purpose |")
    add("| --- | --- |")
    add("| `reports/test-report.md` | This report |")
    add("| `reports/functional-details.json` | All functional comparison records |")
    add("| `reports/test-output.log` | Raw stdout/stderr from the test run |")
    add("| `reports/coverage.json` | Full coverage data incl. missing branches |")
    if write_html:
        add("| `reports/coverage.xml` | Cobertura XML for CI |")
        add("| `reports/htmlcov/index.html` | Line-by-line coverage browser |")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_dependencies() -> bool:
    """Install runtime and test dependencies into the running interpreter.

    The former ``run_tests.cmd`` / ``run_tests.ps1`` wrappers handled this
    step. Keeping it here makes ``python run_tests.py`` a genuine one-click
    entry point with no shell script and no execution-policy workaround.
    """
    print("Installing runtime and test dependencies...", flush=True)
    command = [sys.executable, "-m", "pip", "install",
               "--disable-pip-version-check", "-q",
               "-r", str(ROOT / "requirements.txt"),
               "-r", str(ROOT / "requirements-test.txt")]
    return subprocess.run(command, cwd=str(ROOT)).returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Run the suite, build all reports and return a CI-friendly exit code."""
    global _REPORT_DIR, coverage_module
    parser = argparse.ArgumentParser(
        description="Run all DCC Studio tests and write a coverage report.")
    parser.add_argument("--quick", "--no-html", "-Quick", dest="no_html",
                        action="store_true",
                        help="skip HTML/XML coverage artefacts")
    parser.add_argument("--no-install", "-NoInstall", action="store_true",
                        help="skip dependency installation (offline use)")
    parser.add_argument("--open", "-Open", dest="open_report",
                        action="store_true",
                        help="open reports/htmlcov/index.html afterwards")
    parser.add_argument("--report-dir", default=str(ROOT / "reports"),
                        help="directory for generated reports")
    args = parser.parse_args(argv)

    if not args.no_install:
        if not _install_dependencies():
            print("ERROR: dependency installation failed.", file=sys.stderr)
            return 2
        # ``coverage`` may have been missing at import time. Installing it
        # above only changes the environment on disk, so the module handle
        # must be refreshed before the coverage context is created.
        if coverage_module is None:
            try:
                coverage_module = importlib.import_module("coverage")
            except ImportError:
                pass

    _REPORT_DIR = Path(args.report_dir).resolve()
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # The regression suite flushes its structured comparison records here.
    # Pointing the tests at the chosen report directory keeps custom
    # --report-dir values consistent with the renderer, and removing a stale
    # copy prevents a broken run from publishing the previous run's numbers.
    functional_path = _REPORT_DIR / "functional-details.json"
    os.environ["DCC_FUNCTIONAL_REPORT"] = str(functional_path)
    if functional_path.exists():
        functional_path.unlink()

    if coverage_module is None:
        print("ERROR: the 'coverage' package is required.\n"
              "Install it with: python -m pip install -r requirements-test.txt\n"
              "or run `python run_tests.py` without --no-install",
              file=sys.stderr)
        return 2

    functions = _collect_functions()
    # The suite runs exactly once, inside the coverage context; its detailed
    # result is returned from there and reused for the report below.
    grand, _per_file, suite = _coverage_data(functions,
                                             write_html=not args.no_html)
    fixture = _fixture_summary()
    functional = _load_functional_records()
    report_path = _REPORT_DIR / "test-report.md"
    _write_report(report_path, suite, grand, fixture, functional,
                  write_html=not args.no_html)

    print(f"Collected {suite.collected} tests: "
          f"{suite.count('passed')} passed, "
          f"{suite.count('failed')} failed, "
          f"{suite.count('error')} errors, "
          f"{suite.count('skipped')} skipped "
          f"({suite.duration:.2f}s)")
    print(f"Statements : {grand['statements_covered']}/"
          f"{grand['statements']} "
          f"({_percent(grand['statements_covered'], grand['statements']):.1f}%)")
    print(f"Branches   : {grand['branches_covered']}/"
          f"{grand['branches']} "
          f"({_percent(grand['branches_covered'], grand['branches']):.1f}%)")
    print(f"Functions  : {grand['functions_covered']}/"
          f"{grand['functions']} "
          f"({_percent(grand['functions_covered'], grand['functions']):.1f}%)")
    print(f"Functional : {len(functional)} comparisons, "
          f"{sum(1 for item in functional if not item['passed'])} failed")
    print(f"Report     : {report_path}")
    if not args.no_html:
        html_index = _REPORT_DIR / "htmlcov" / "index.html"
        print(f"Coverage   : {html_index}")
        if args.open_report and html_index.exists():
            webbrowser.open(html_index.as_uri())
    return 0 if suite.ok else 1


_REPORT_DIR: Path = ROOT / "reports"


if __name__ == "__main__":
    raise SystemExit(main())
