# 测试与数值回归 / Tests and numerical regression

> 中文 / English · 面向开发者与 CI（For maintainers and CI）

本文件说明如何复现 DCC Studio 的数值验证。科研用户若只想确认结果可信，
可直接阅读 `reports/test-report.md` 第 1 章的逐项对照，或参见
[`METHOD.md`](METHOD.md) 的 Numerical validation 一节；日常使用无需执行本文件的命令。

---

## 中文

### 1. 一键入口

`run_tests.py` 是唯一的测试入口。它是纯 Python 脚本，仓库中不再有 `.ps1` /
`.cmd` 包装，因此完全不受 Windows PowerShell 执行策略限制：

```powershell
python run_tests.py                   # 安装依赖 + 全部用例 + 全套报告
python run_tests.py --quick           # 跳过 HTML/XML 覆盖率产物（最快）
python run_tests.py --no-install      # 跳过依赖安装（离线环境）
python run_tests.py --open            # 结束后打开 HTML 覆盖率页
python run_tests.py --report-dir out  # 自定义报告输出目录
```

`python` 应指向项目虚拟环境，例如
`.\.venv\Scripts\python.exe run_tests.py`。旧的 PowerShell 风格参数
`-Quick` / `-NoInstall` / `-Open` 仍作为别名可用。脚本会自行安装
`requirements.txt` 与 `requirements-test.txt`；退出码为 0 仅当所有用例通过，
可直接用于 CI 或任务计划。

### 2. 报告结构

`reports/test-report.md` 按「功能性详细、程序性概要」组织：

- **第 1 章 功能性验证（详细）**：开篇列出 13 个原样拷贝的参考数据文件及用途；
  每个功能 case 单独成节，写明所用测试集、参考结果文件、逐项对比的**原测试结果**
  与**本次计算结果**（min / max / mean）、最大绝对差异、容差与判定，并附代表性
  元素的逐值对照。结构化明细另存于 `reports/functional-details.json`。
- **第 2 章 程序性 UT（概要）**：只给出 collected / passed / failed / errors /
  skipped / 耗时汇总，失败时才附 traceback，不逐条展开用例过程。
- **第 3 章 覆盖率（概要）**：只给出语句、分支、函数三项百分比；逐模块、逐函数
  及缺失分支明细保留在 `reports/coverage.json` 与 `reports/htmlcov/index.html`。
- **第 4 章 产物清单**：报告、功能明细 JSON、原始日志、coverage JSON / XML、
  HTML 覆盖率页路径。

`DCC_FUNCTIONAL_REPORT` 环境变量控制功能明细 JSON 的写入位置，由入口脚本自动设置，
普通调用无需关心。

### 3. 参考数据与功能回归

参考数据**不随仓库分发**（13 个文件、约 146 MB），需单独下载后解压到
`tests/fixtures/`：直接从参考文件夹**原样拷贝**的时间序列输入（`.mat`）、图谱
标签与网络名称（`.txt` / `.csv`）、逐样本 DCC 结果、组统计、t 检验与 Cohen's d、
PLS 分量及预期结果。

| 项目 | 说明 |
|---|---|
| 下载链接 | **（暂空，待补充）** |
| 解压位置 | 仓库根目录下的 `tests/fixtures/`，保留 `Brain_mask/`、`MSC_process/`、`TMS_process/`、`UCLA_process/` 子目录结构 |
| 完整性检查 | 13 个文件齐全且非空；完整清单见 `tests/test_reference_parity.py` 的 `REQUIRED_FILES` |

`tests/fixtures/` 已加入 `.gitignore`，下载的数据不会被提交。未放置数据时
`run_tests.py` 会在报告中标注 **Missing**，数值回归用例不可用，其余单元测试仍可运行。
数据就位后，`tests/test_reference_parity.py` 直接打开这些文件构造 test case，
逐元素重算并与存储结果对照，**不需要生成脚本，也不需要用户配置环境变量
或外部数据路径**。最近一次运行完成 35 项数值对比、0 项超容差。

### 4. 最近一次全套运行结果

2026-11-08，Windows / Python 3.12：

| 项目 | 结果 |
|---|---|
| 用例 | 117 collected / 117 passed / 0 failed / 0 errors |
| 覆盖率 | 语句 84.7% · 分支 69.9% · 函数 84.8% |
| 功能对照 | 35 项数值比较，0 项失败 |

这些数字随版本变化，始终以新生成的 `reports/test-report.md` 为准。

### 5. 覆盖率范围

覆盖 `dcc/`、`gui/`、`packaging/` 三个生产包；测试代码、参考数据与
`run_gui.py` 等薄入口脚本不计入，以免虚高。分支覆盖率由 `coverage.py`
的 `branch=True` 提供，函数覆盖率由入口脚本遍历 AST 推导。

---

## English

### 1. Single entry point

`run_tests.py` is the only test entry point. It is a plain Python script; no
`.ps1` or `.cmd` wrappers remain, so the Windows PowerShell execution policy is
irrelevant:

```powershell
python run_tests.py                   # deps + every case + full reports
python run_tests.py --quick           # skip HTML/XML coverage artefacts
python run_tests.py --no-install      # offline: skip dependency installation
python run_tests.py --open            # open the HTML coverage browser afterwards
python run_tests.py --report-dir out  # custom report directory
```

Point `python` at the project virtual environment, e.g.
`.\.venv\Scripts\python.exe run_tests.py`. The legacy PowerShell-style flags
`-Quick` / `-NoInstall` / `-Open` remain as aliases. The script installs
`requirements.txt` and `requirements-test.txt` itself, and returns 0 only when
every case passes, so it works unchanged in CI and Task Scheduler.

### 2. Report structure

`reports/test-report.md` follows "functional detail, procedural summary":

- **Chapter 1, functional verification (detailed)**: the inventory of all 13
  verbatim-copied reference files, then one section per functional case with
  the dataset, the reference result file, the **stored** versus **recomputed**
  values (min / max / mean), max absolute deviation, tolerance and verdict,
  plus representative element-by-element values. Structured records are also
  written to `reports/functional-details.json`.
- **Chapter 2, programmatic unit tests (summary)**: collected / passed /
  failed / errors / skipped / duration only, with tracebacks on failure; no
  per-case walkthrough.
- **Chapter 3, coverage (summary)**: statement, branch and function percentages
  only. Per-module, per-function and missing-branch detail stays in
  `reports/coverage.json` and `reports/htmlcov/index.html`.
- **Chapter 4, artefacts**: paths to the report, functional JSON, raw log,
  coverage JSON/XML and the HTML browser.

`DCC_FUNCTIONAL_REPORT` controls where the functional JSON is written; the
runner sets it automatically.

### 3. Reference data and regression

The reference dataset is **not distributed with the repository** (13 files,
~146 MB). Download it separately and unpack it into `tests/fixtures/`:
**verbatim copies** of the reference project's time-series inputs (`.mat`),
atlas labels and network names (`.txt` / `.csv`), per-sample DCC results, group
statistics, t tests / Cohen's d and PLS outputs with their expected values.

| Item | Value |
|---|---|
| Download link | **(TBD - to be filled in)** |
| Extract to | `tests/fixtures/` at the repository root, keeping the `Brain_mask/`, `MSC_process/`, `TMS_process/` and `UCLA_process/` subdirectories |
| Verify | 13 non-empty files; `REQUIRED_FILES` in `tests/test_reference_parity.py` is the full inventory |

`tests/fixtures/` is gitignored, so downloaded data is never committed. Without
it, `run_tests.py` marks the dataset as **Missing** in its report; the remaining
unit tests still run. Once in place, `tests/test_reference_parity.py` opens the
files directly and compares every recomputed quantity element-by-element. No
generation script and no user-configured environment variable is needed. The
latest run performed 35 numerical comparisons with 0 tolerance failures.

### 4. Latest full run

2026-11-08, Windows / Python 3.12:

| Item | Result |
|---|---|
| Cases | 117 collected / 117 passed / 0 failed / 0 errors |
| Coverage | 84.7% statements · 69.9% branches · 84.8% functions |
| Functional | 35 comparisons, 0 failed |

These numbers change with the codebase; always rely on the freshly generated
`reports/test-report.md`.

### 5. Coverage scope

The production packages `dcc/`, `gui/` and `packaging/`. Test modules, copied
reference data and thin launchers such as `run_gui.py` are excluded so the
percentages are not inflated. Branch coverage comes from `coverage.py` with
`branch=True`; function coverage is derived by the runner walking the AST.

---

相关文档：[`build.md`](build.md) 构建与分发 · [`develop.md`](develop.md) 项目结构 ·
[`METHOD.md`](METHOD.md) 方法原理 · [`../README.md`](../README.md) 科研向主文档.
