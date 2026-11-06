# DCC Studio | 动态连接复杂度分析软件
# DCC Studio | Dynamic Connectivity Complexity Analysis

> 中文 / English

---

## 中文

DCC Studio 是面向科研工作者的桌面软件，用于计算**动态连接复杂度**（Dynamic Connectivity Complexity, DCC）。给定每个样本的 ROI 时间序列和「ROI → 网络」图谱，它为每个 ROI 计算一个以 bit 为单位的复杂度值，并输出组级统计、连接状态分布与论文风格图。

### 1. 快速使用

- 双击 `DCCStudio.exe` 启动（无需安装 Python）；`config.json` 与 `results/` 写在 exe 所在目录。若需从源码运行，见 [`docs/build.md`](docs/build.md)。
- 跑通一次的最短路径：在 `1 Data & Method` 选好时间序列、图谱标签与输出目录 → 点 `Validate inputs` → `2 Parameters` 保持默认 → `3 Compute` 开始计算 → `4 Results` 查看表格与图。
- 首次使用建议照着保姆级 Word 教程逐步操作：[中文教程](docs/DCC_Studio_使用教程_中文.docx) ／ [English guide](docs/DCC_Studio_User_Guide_EN.docx)。教程内含基于 `tests/fixtures/` 真实发表数据的 file / folder 两组完整实操。该数据集（约 146 MB）**不随仓库分发**，需单独下载后放入 `tests/fixtures/`；下载链接与放置说明见 [`docs/test.md`](docs/test.md)（链接待补充）。

### 2. 功能介绍：DCC

**DCC 衡量什么。** 把一个 ROI 与各功能网络之间的时变连接看作一串离散状态，用这些状态的 Shannon 熵衡量其复杂度：

$$
\mathrm{DCC}_i = -\sum_s p_{i,s}\,\log_2 p_{i,s}
$$

`p[i,s]` 是第 `i` 个 ROI 落在状态 `s` 的频率，单位为 bit。**数值越高，说明该 ROI 在扫描过程中参与的连接状态越丰富、越不固定**；数值低则意味着连接模式长期停留在少数几种状态。

**每个样本独立计算，分五步：**

1. **网络信号**：按图谱把 ROI 信号平均到所属网络，得到每个网络的信号 `N_g(t)`。
2. **动态功能连接**：长度 `SWL` 的滑窗以 `SWS` 为步长滑动，计算每个 ROI 信号与各网络信号的 Pearson 相关 `D[i, g, w]`；窗口数为 `1 + floor((T − SWL) / SWS)`。
3. **符号处理与稀疏化**：按 `abs` / `raw` / `pos` 处理相关符号，再用阈值（全局 `mean(D) + k·std(D)` 或逐 ROI）把 `D` 二值化为 0/1。
4. **连接状态**：每个窗口、每个 ROI 得到一个 `G` 位模式，按二进制权重解码为状态编号；丢弃全零状态后共 `2^G − 1` 种，统计各状态出现频率。
5. **复杂度**：把频率归一化为 `p`，代入上式得到逐 ROI 的 DCC。

**需要的数据。** 每个样本一个 `(ROI 数 × 时间点数)` 矩阵，可放在单个容器文件里，也可一个目录每文件一个样本；支持 `.mat`、`.npy`、`.npz`、`.csv`、`.tsv`、`.txt`。另需一份图谱标签（每行一个整数网络编号，顺序与 ROI 一致，最多 16 个网络）。网络名称、样本元数据、行为数据表均为可选。样本可以是被试、会话、run 或实验条件，软件不预设研究设计。

**得到什么。** ROI × 样本 DCC 矩阵、组均值 / 标准差 / z 分数、网络均值、逐 ROI 与逐网络的状态概率、高频状态与 top-state coverage、样本间 DCC 相似度，以及论文风格图和记录输入与参数的 `summary.json`。每次运行按参数写入独立结果目录，互不覆盖。

**自动触发的可选分析。** 提供样本元数据时，软件自动追加组间比较（每网络 t 检验与 Cohen's d）、ICC(1,1) 重测信度、以及行为表与 DCC 的 PLS-C 关联；条件不满足时只在 `summary.json` 里给出 warning，核心 DCC 结果照常产出。

**核心参数与可复现性。** 滑窗长度 `SWL`、步长 `SWS`、阈值系数 `k`、符号模式与阈值模式。相同输入与参数产生完全相同的结果；随机种子仅影响置换 / bootstrap 等可选分析。完整推导、参数默认值与输出文件清单见 [`docs/METHOD.md`](docs/METHOD.md)。

---

## English

DCC Studio is a desktop application for researchers computing **Dynamic
Connectivity Complexity (DCC)**. Given each sample's ROI time series and an
ROI-to-network atlas, it produces one complexity value in bits per ROI, plus
group statistics, connectivity-state distributions and publication-style
figures.

### 1. Quick start

- Double-click `DCCStudio.exe` (no Python installation needed); `config.json`
  and `results/` are written next to the exe. To run from source, see
  [`docs/build.md`](docs/build.md).
- Shortest path to a first result: on `1 Data & Method`, pick the time series,
  atlas labels and output directory → click `Validate inputs` → leave
  `2 Parameters` at its defaults → `3 Compute` → `4 Results` for tables and
  figures.
- For a first run, follow the step-by-step Word guide:
  [Chinese](docs/DCC_Studio_使用教程_中文.docx) /
  [English](docs/DCC_Studio_User_Guide_EN.docx). Both contain two complete
  worked examples on the real published data in `tests/fixtures/` (file mode
  and folder mode). That dataset (~146 MB) is **not distributed with the
  repository**: download it separately into `tests/fixtures/`. See
  [`docs/test.md`](docs/test.md) for the download link (to be filled in) and
  placement instructions.

### 2. What DCC measures

**The idea.** Treat the time-varying connectivity between one ROI and each
functional network as a sequence of discrete states, and quantify those states
with Shannon entropy:

$$
\mathrm{DCC}_i = -\sum_s p_{i,s}\,\log_2 p_{i,s}
$$

`p[i,s]` is the fraction of windows in which ROI `i` occupies state `s`, and
the value is in bits. **A higher DCC means the ROI visits a richer, less
repetitive repertoire of connectivity states**; a low value means connectivity
stays in a few states for most of the scan.

**Five steps, computed independently for each sample:**

1. **Network signals.** Average ROI signals within each network to obtain
   `N_g(t)`.
2. **Dynamic functional connectivity.** Slide a window of length `SWL` in steps
   of `SWS` and correlate every ROI signal with every network signal, giving
   `D[i, g, w]`; the number of windows is `1 + floor((T − SWL) / SWS)`.
3. **Sign handling and sparsification.** Apply `abs`, `raw` or `pos`, then
   binarise `D` with a global (`mean(D) + k·std(D)`) or per-ROI threshold.
4. **Connectivity states.** Each window yields a `G`-bit pattern per ROI,
   decoded as an integer state. Discarding the all-zero state leaves
   `2^G − 1` states; count how often each occurs.
5. **Complexity.** Normalise the counts to `p` and apply the entropy formula
   above to obtain one DCC value per ROI.

**Inputs.** One `(n_roi × n_timepoints)` matrix per sample, either in a single
container file or as one file per sample in a folder; `.mat`, `.npy`, `.npz`,
`.csv`, `.tsv` and `.txt` are supported. An atlas label file is also required:
one integer network index per line in ROI order (up to 16 networks). Network
names, sample metadata and a behaviour table are optional. A sample may be a
subject, session, run or condition; no study design is assumed.

**Outputs.** The ROI × sample DCC matrix, group means / standard deviations /
z-scores, network means, per-ROI and per-network state probabilities,
high-frequency states and top-state coverage, between-sample DCC similarity,
publication-style figures, and `summary.json` recording inputs and parameters.
Each run writes to its own parameter-keyed result directory, so runs never
overwrite one another.

**Optional analyses, enabled by metadata.** With sample metadata, the software
automatically adds group comparisons (per-network t tests and Cohen's d),
ICC(1,1) test-retest reliability, and PLS-C relating behaviour to DCC. When
the prerequisites are not met it only records a warning in `summary.json`;
the core DCC results are always produced.

**Key parameters and reproducibility.** Window length `SWL`, step `SWS`,
threshold factor `k`, sign mode and threshold mode. Identical inputs and
parameters give identical results; the random seed affects only permutation /
bootstrap analyses. For the full derivation, parameter defaults and the
complete output file list, see [`docs/METHOD.md`](docs/METHOD.md).
