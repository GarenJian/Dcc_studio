# 项目结构与实现说明 / Project layout and internals

> 中文 / English · 面向二次开发者（For contributors）

本文件记录代码组织与实现约定。[`../README.md`](../README.md) 面向科研用户，
只讲输入、操作与结果；测试见 [`test.md`](test.md)，环境与打包见
[`build.md`](build.md)，算法与扩展指南见 [`METHOD.md`](METHOD.md)。

---

## 中文

### 1. 目录结构

```
run_gui.py                       GUI 入口
run_tests.py                     一键测试与覆盖率报告（自带依赖安装）
build_exe.py / build_exe.spec    一键打包入口与 PyInstaller 配置
requirements*.txt                运行时 / 测试 / 构建依赖（见 build.md）
dcc/                             纯计算核心（无界面依赖）
  config.py atlas.py timeseries.py complexity.py plotting.py ...
  methods/                       方法接口、注册表、内置 DCC
  pipelines.py                   六阶段通用 workflow
gui/                             四步界面、Settings 对话框、后台线程、主题、预览
  preview.py                     有界读取与 LRU 预览缓存
packaging/selftest.py            冻结环境自检入口
tests/                           单元、IO、端到端、主题、设置、注册表、预览、数值对照
tests/fixtures/                  单独下载的参考输入与结果（约 146 MB，gitignored）
docs/                            方法、工程与用户文档
reports/                         生成的测试报告与覆盖率产物（gitignored）
dist/ build/                     打包产物与缓存（gitignored）
```

### 2. 可扩展的方法架构

DCC 只是动态功能连接的一种度量。计算流程、统计、绘图、导出与界面都不依赖具体方法：

| 模块 | 作用 |
|---|---|
| `dcc/methods/base.py` | `AnalysisMethod` 抽象接口：名称、说明、度量缩写、单样本 `compute()` |
| `dcc/methods/registry.py` | 注册与查询；`method_choices()` 驱动 Settings 中的方法下拉框 |
| `dcc/methods/dcc_method.py` | 内置 DCC 实现（复用经过回归验证的核心函数） |
| `gui/settings.py` | 设置对话框；`PLANNED_METHODS` 中的规划方法以灰色禁用项展示 |

新增方法只需：继承 `AnalysisMethod` 实现 `compute()`、用 `@register_method` 注册、
在 `dcc/methods/__init__.py` 导入一次。界面选项、`config.json` 的 `method` 字段、
`run_tag()` 的缓存隔离、输出文件前缀与 `summary.json` 会自动适配；若该方法原本是
规划项，其「暂不支持」占位行会自动消失。完整的契约、参数与下游假设见
[`METHOD.md`](METHOD.md) 的 Extending 章节。

> 注意：当前 GUI 文案、参数集合、结果页分类与图表标签仍带有 DCC 语义。注册表分发
> 到 `method.compute()` 是通用的，但新增一种与 DCC 状态空间不同的方法时，还需扩展
> `SubjectResult` 契约及 summary / plotting / stats 阶段，不能只靠注册。

### 3. Results 预览实现

结果页的预览面向快速浏览，实现在 `gui/preview.py` 与 `gui/app.py`：

| 约束 | 值 |
|---|---|
| CSV 行数 / 列数 | 200 / 40（`read_csv_preview` 先读表头，再按位置 `usecols` 限列） |
| 文本字符数 | 200,000（`handle.read(n)`，不再 `read_text()[:n]` 全量加载） |
| 图片宽度 | 1000 px（`QImageReader.setScaledSize` 解码期缩放 + 一次平滑缩放） |
| LRU 缓存 | 24 条，key 为 `(path, st_mtime_ns, st_size)`，文件重写自动失效 |
| 选择防抖 | 单次 `QTimer` 90 ms，只渲染最后选中的文件并丢弃过期请求 |

表格采用 `QHeaderView.Interactive` + 默认列宽 110 px，批量填充时关闭信号与重绘，
避免 `ResizeToContents` 对每个单元格测量宽度。若内容被截断，路径标签通过
`_preview_caption()` 明确标注；磁盘上的原始文件始终完整。

### 4. 结果缓存隔离

`run_tag()` 把参数与方法编码进结果目录名，`samples/*.npz` 保存逐样本中间结果。
不同数据源但相同参数会共用同一目录，因此换数据集时应更换 `Output directory`。

---

## English

### 1. Layout

```
run_gui.py                       GUI entry point
run_tests.py                     one-click tests + coverage reports
build_exe.py / build_exe.spec    one-click build + PyInstaller config
requirements*.txt                runtime / test / build deps (see build.md)
dcc/                             pure computation core (no GUI dependency)
  config.py atlas.py timeseries.py complexity.py plotting.py ...
  methods/                       method contract, registry, built-in DCC
  pipelines.py                   six-stage generic workflow
gui/                             four-step UI, Settings dialog, worker, themes, preview
  preview.py                     bounded readers + LRU preview cache
packaging/selftest.py            frozen-environment self-test
tests/                           unit, IO, end-to-end, theme, settings, registry,
                                 preview and numerical-parity cases
tests/fixtures/                  separately downloaded reference data (~146 MB, gitignored)
docs/                            method, engineering and user documentation
reports/                         generated test reports (gitignored)
dist/ build/                     build artefacts and caches (gitignored)
```

### 2. Extensible method architecture

DCC is one dynamic functional connectivity measure. The workflow, statistics,
figures, exporter and UI are method-agnostic:

| Module | Responsibility |
|---|---|
| `dcc/methods/base.py` | `AnalysisMethod` contract: name, description, metric label, per-sample `compute()` |
| `dcc/methods/registry.py` | Registration and lookup; `method_choices()` drives the Settings drop-down |
| `dcc/methods/dcc_method.py` | Built-in DCC adapter over the regression-tested core function |
| `gui/settings.py` | Settings dialog; `PLANNED_METHODS` entries appear as disabled rows |

To add a method: subclass `AnalysisMethod`, implement `compute()`, decorate it
with `@register_method`, and import it once from `dcc/methods/__init__.py`.
The Settings drop-down, the `method` field in `config.json`, `run_tag()` cache
separation, output file prefixes and `summary.json` all adapt automatically;
a planned-method placeholder disappears once that name is registered. The full
contract, parameters and downstream assumptions are in the Extending chapter of
[`METHOD.md`](METHOD.md).

> Caveat: GUI copy, parameter sets, Results-page categories and figure labels
> still carry DCC semantics. Registry dispatch to `method.compute()` is generic,
> but a method with a different state space also needs a broader
> `SubjectResult` contract and changes to the summary / plotting / stats stages;
> registration alone is not enough.

### 3. Results preview internals

Previews are for quick browsing, implemented in `gui/preview.py` and `gui/app.py`:

| Bound | Value |
|---|---|
| CSV rows / columns | 200 / 40 (`read_csv_preview` reads the header, then limits by position with `usecols`) |
| Text characters | 200,000 (`handle.read(n)` instead of loading the whole file with `read_text()[:n]`) |
| Image width | 1000 px (`QImageReader.setScaledSize` at decode time + one smooth downscale) |
| LRU cache | 24 entries, keyed by `(path, st_mtime_ns, st_size)`; a rewrite invalidates automatically |
| Selection debounce | single-shot 90 ms `QTimer`; only the last selection renders, stale requests are dropped |

The table uses `QHeaderView.Interactive` with a 110 px default column width and
fills in one batched, signal-free pass, so `ResizeToContents` no longer measures
every cell. `_preview_caption()` states any truncation in the path label; the
files on disk always hold the full result.

### 4. Result cache isolation

`run_tag()` encodes parameters and method into the result directory name, and
`samples/*.npz` stores per-sample intermediates. Different data sources with
identical parameters would share that directory, so change `Output directory`
when switching datasets.

---

相关文档：[`test.md`](test.md) · [`build.md`](build.md) ·
[`METHOD.md`](METHOD.md) · [`../README.md`](../README.md).
