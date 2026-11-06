# 环境准备、构建与分发 / Setup, build and distribution

> 中文 / English · 面向开发者与打包者（For maintainers and packagers）

科研用户拿到 `dist/DCCStudio.exe` 后直接双击即可，无需阅读本文件。
[`../README.md`](../README.md) 与两份 Word 教程说明日常使用；本文件说明如何从源码
准备环境、生成独立 exe 以及在冻结环境中自检。

---

## 中文

### 1. 依赖分组

| 文件 | 何时需要 |
|---|---|
| `requirements.txt` | 运行时（numpy / scipy / pandas / PyQt6 / matplotlib / h5py） |
| `requirements-test.txt` | 运行测试时（coverage），由 `run_tests.py` 自动安装 |
| `requirements-build.txt` | 打包时（PyInstaller），由 `build_exe.py` 自动安装 |

构建依赖与运行依赖分离：只有产出分发包的机器需要 PyInstaller。

### 2. 从源码准备环境并启动

```powershell
cd dcc_studio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_gui.py
```

支持的 Python 版本见 `requirements.txt` 首行注释（3.11 – 3.13）。
`build_exe.py` 会优先使用 `.venv` 里的解释器，因此打包机与开发机保持一致。

### 3. 打包为独立 exe

`build_exe.py` 是唯一的打包入口（纯 Python，不受 PowerShell 执行策略限制，
无需任何包装脚本）：

```powershell
python build_exe.py                  # 默认：单文件 GUI exe（体积最小）
python build_exe.py --verify         # 额外在冻结环境执行完整自检
python build_exe.py --one-dir        # 文件夹模式（体积更大、启动更快）
python build_exe.py --clean          # 先清 PyInstaller 缓存
python build_exe.py --run            # 构建成功后启动 GUI
python build_exe.py --clean --verify --run
```

旧的 PowerShell 风格参数 `-Verify` / `-Clean` / `-OneDir` / `-Run` 仍作为别名可用。
脚本自行安装 `requirements-build.txt`，按 `build_exe.spec` 构建、输出体积，并透传
退出码供 CI / 任务计划使用。`dist/` 与 `build/` 由 `.gitignore` 忽略。

### 4. 冻结环境自检

`--verify` 用**同一份 spec** 额外构建一个临时 `dcc_verify` exe，设置
`DCC_SELFTEST=1` 运行 [`packaging/selftest.py`](../packaging/selftest.py)，
检查配置、图谱、Settings 对话框、方法列表与主题在冻结环境下可用，随后自动删除
临时 exe 与其 build 目录，因此 `dist/` 只留正式产物。自检输出以
`SELFTEST OK` 结尾。

### 5. 产物与实测

本机实测（Windows 11 x64 / Python 3.12 / PyInstaller 6.22）：
`dist\DCCStudio.exe` 单文件 **84.7 MB**。双击即可运行，无需安装 Python；
`config.json` 与 `results/` 写在 exe 所在目录。单文件模式启动时需解压到临时目录
（首次约数秒），若追求最快启动速度可用 `--one-dir`。产品没有 CLI 变体，
GUI exe 是唯一分发产物。

### 6. 最小体积策略

排除无用标准库与可选依赖；只保留 Qt 必需的 6 个 DLL 与平台/图片插件；删除
translations、软件 OpenGL、AVIF/OpenSSL 组件、matplotlib 示例数据与 mathtext
字体；不启用 UPX（压缩 Qt DLL 可能导致启动失败或杀毒软件误报），由 PyInstaller
自身的 zlib 压缩。详见 `build_exe.spec`。

---

## English

### 1. Dependency groups

| File | Needed when |
|---|---|
| `requirements.txt` | Runtime (numpy / scipy / pandas / PyQt6 / matplotlib / h5py) |
| `requirements-test.txt` | Running tests (coverage); installed by `run_tests.py` |
| `requirements-build.txt` | Packaging (PyInstaller); installed by `build_exe.py` |

Build dependencies are separate from runtime dependencies: only the machine
producing the distributable needs PyInstaller.

### 2. Prepare a source environment and launch

```powershell
cd dcc_studio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_gui.py
```

Supported Python versions are listed in the header of `requirements.txt`
(3.11 – 3.13). `build_exe.py` prefers the `.venv` interpreter, so the build
machine and the development machine stay consistent.

### 3. Build the standalone exe

`build_exe.py` is the single build entry point (plain Python; no PowerShell
execution-policy issues and no wrapper script):

```powershell
python build_exe.py                  # default: single-file GUI exe (smallest)
python build_exe.py --verify         # also run the frozen self-test
python build_exe.py --one-dir        # folder output (larger, faster startup)
python build_exe.py --clean          # clear the PyInstaller cache first
python build_exe.py --run            # launch the GUI after a good build
python build_exe.py --clean --verify --run
```

The legacy PowerShell-style flags `-Verify` / `-Clean` / `-OneDir` / `-Run`
remain as aliases. The script installs `requirements-build.txt`, builds against
`build_exe.spec`, reports the output size and preserves the exit code for CI /
Task Scheduler. `dist/` and `build/` are gitignored.

### 4. Frozen-environment self-test

`--verify` builds a temporary `dcc_verify` exe from the **same spec**, sets
`DCC_SELFTEST=1` and runs
[`packaging/selftest.py`](../packaging/selftest.py), which checks config, atlas,
the Settings dialog, the method list and theme handling inside the frozen
environment. The temporary exe and its build directory are then removed, so
`dist/` always contains only the product. A successful run ends with
`SELFTEST OK`.

### 5. Artifact and measurement

Measured on Windows 11 x64 / Python 3.12 / PyInstaller 6.22:
`dist\DCCStudio.exe`, single file, **84.7 MB**. Double-click it; no Python
installation is required. `config.json` and `results/` are written next to the
executable. Single-file mode unpacks to a temporary directory on startup (a few
seconds the first time); use `--one-dir` for the fastest startup. There is no
CLI variant: the GUI exe is the only shipped executable.

### 6. Size strategy

Exclude unused stdlib modules and optional dependencies; keep only the six DLLs
and the platform/image plugins Qt actually needs; drop translations, software
OpenGL, AVIF/OpenSSL components, matplotlib sample data and mathtext-only
fonts; no UPX (compressing Qt DLLs can break startup or trigger antivirus false
positives) -- PyInstaller's own zlib compression is used instead. See
`build_exe.spec`.

---

相关文档：[`test.md`](test.md) 测试与覆盖率 · [`develop.md`](develop.md) 项目结构 ·
[`../README.md`](../README.md) 科研向主文档.
