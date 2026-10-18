# SuperJUNO 赛题说明：深地探索——中微子事件定位模拟

## 赛题介绍

江门地下中微子观测站（Jiangmen Underground Neutrino Observatory，JUNO）是一所研究中微子的地下实验室，通过探测反应堆等来源的中微子，研究中微子振荡和质量顺序等问题。实验的物理目标与探测器设计可参阅 JUNO 合作组的综述论文 [JUNO Physics and Detector](https://arxiv.org/abs/2104.02565)。

反应堆电子反中微子可以通过逆贝塔衰变（inverse beta decay，IBD）与质子反应：

$$
\bar{\nu}_e + p \longrightarrow e^+ + n.
$$

其中，正电子在液体闪烁体中沉积能量并湮灭，形成瞬发信号；中子减速后被俘获，形成延迟信号。本任务关注含正电子的模拟事例，重建同一读出窗内的瞬发信号及叠加本底，不涉及完整的瞬发／延迟信号配对。

粒子沉积的能量产生闪烁光，光电倍增管（photomultiplier tube，PMT）将接收到的光转换为光电子（photoelectron，PE），经放大和电子学采样形成波形。多个 PE 的电脉冲可能相互重叠，还会受到增益涨落和噪声影响。单光电子响应与波形建模可参阅 [PMT waveform modeling at the Daya Bay experiment](https://authors.library.caltech.edu/records/16kk7-sft36)。

这项工作类似于根据大厅四周的录音判断拍手的位置：不同探测器接收到的信号强度和到达时间不同，需要联合分析才能还原光源。实际记录还受到光传播、电脉冲重叠和暗噪声等因素的影响，因此程序需要同时考虑探测器响应和观测中的不确定性。

SuperJUNO 从这一科学分析任务中提取了一条完整计算流程：先从 PMT 波形构造采样先验，再使用快速随机匹配追踪（Fast Stochastic Matching Pursuit，FSMP）采样 PE 的数量和到达时间，最后联合各通道完成最多双顶点的事例重建。FSMP 通过可逆跳跃马尔可夫链蒙特卡洛（RJMCMC）探索不同的 PE 序列，其方法与 GPU 加速研究见 [The Fast Stochastic Matching Pursuit for Neutrino and Dark Matter Experiments](https://arxiv.org/abs/2403.03156)。

“顶点”用一个等效点源描述发光位置，其状态包含位置 `x、y、z`、可见能量 `E` 和时刻 `t`。程序通过概率采样，探索一个或两个光源对观测的解释，并保存采样链。当前任务处理正电子瞬发信号与碳-14 本底叠加的模拟事例；顶点数表示模型中的等效光源数，并不直接等于中微子的数量。

本赛题要求选手在**保持规定数学模型、采样规则、计算规模和结果精度**的前提下，优化完整流程，尽可能缩短运行时间。

- 运行平台：**并行科技 NVIDIA GeForce RTX 4090 集群，限单节点，每次运行最多使用两张 GPU**。
- 软件环境：通过 module 加载 CUDA 12.4 和 Miniforge3 24.11，使用 Python 3.11 的 `juno` Conda 环境，其他依赖见 `pyproject.toml`。
- 作业调度：使用 **Slurm**，通过 `sbatch` 提交计算任务。
- 计算任务：完成先验生成、FSMP 采样和事例重建三个阶段，提交 PE 样本、重建采样链及完整计时记录。

## 参考链接

- [源码和数据下载地址](https://cloud.tsinghua.edu.cn/d/0bc5df5078e84dca81a2/)：源码与输入数据及参考结果。
- [README.md](README.md)：数据准备、环境配置与运行入口。
- [docs/recon.tex](docs/recon.tex)：物理背景、统计模型、数据格式和随机量映射。
- [test/README.md](test/README.md)：当前 PE、tvE 结果的比较规则。

本文的运行流程与评测要求以当前仓库的 `scripts/run.py` 和 `test/` 为准；论文用于理解科学背景和方法。

## 登录集群

参赛账号、SSH 密钥、登录节点和 GPU 资源由主办方发放。将以下示例中的 `TEAM_USER`、`LOGIN_NODE` 和密钥路径替换为实际信息：

```bash
ssh -i ~/.ssh/id_ed25519 TEAM_USER@LOGIN_NODE
```

登录节点仅用于编辑代码、安装环境和提交作业，**禁止直接在登录节点运行计算程序**。所有计算任务通过 Slurm 提交，限单节点、最多两张 GPU。代码、数据和权重文件统一保存在 `~/run/` 下；本文使用 `~/run/superjuno/` 作为仓库目录。

建议使用 `tmux new -s superjuno` 创建会话，断线后通过 `tmux a -t superjuno` 重连。

常用命令：`module avail` 查看可用软件，`module list` 查看已加载模块，`sinfo` 查看分区，`sbatch` 提交作业，`squeue -u "$USER"` 查看本队作业，`scancel <JOBID>` 取消作业。

## 代码获取

在 `~/run/` 下获取源码：

```bash
cd ~/run
git clone git@git.tsinghua.edu.cn:scc/thuscc26/superjuno.git superjuno
cd superjuno
```

主要代码结构如下：

```text
superjuno/
├── scripts/run.py         # 运行三阶段计算、计时并按需调用正确性比较
├── src/
│   ├── fsmp_prior.py      # 从原始波形构造采样先验
│   ├── fsmp.py            # 采样光电子数量和到达时间
│   ├── main.py            # 事例位置、时间和能量重建入口
│   ├── recon.py          # 似然、EM 更新和重建主循环
│   ├── mcmc.py           # 顶点数、位置与时间的采样更新
│   ├── detector.py       # PMT 几何和光学响应
│   └── analysis/          # 波形分析与 GPU 算子
├── test/
│   ├── compare_PE.py      # PE 结果比较
│   ├── compare_tvE.py     # 重建结果比较
│   └── comparison.py      # 公共比较逻辑与容差设置
├── pyproject.toml        # Python 项目与依赖配置
└── docs/recon.tex        # 算法和数据格式说明
```

### 数据准备

从 [数据下载地址](https://cloud.tsinghua.edu.cn/d/0bc5df5078e84dca81a2/) 获取输入和参考结果，放在 `~/run/superjuno/data/` 下：

```text
data/
├── raw/<id>.h5
├── ana_PE/<id>.pq
├── ana_tvE/<id>.h5
├── ser_truth_J25.7.4+.h5
├── junoid_J25.3.0.pq
├── PMT_J26.4.0.h5
├── sim.h5
└── J25.3.0.PMTPos_CD_LPMT.csv
```

`<id>` 为测例文件名主体。共提供一个示例点 `one` 和三个正式测试点 `0`、`1`、`2`：

| 测例 | 用途 | 是否计分 |
| --- | --- | --- |
| `one` | 熟悉流程、调试与验证环境 | 否 |
| `0`、`1`、`2` | 正式评测，均须提交日志和生成数据 | 是，三个测试点等权 |

每个测例对应 `raw/<id>.h5` 输入；`ana_PE/` 和 `ana_tvE/` 分别存放对应的 PE、重建参考结果。

| 输入文件 | 用途 |
| --- | --- |
| `raw/<id>.h5` | 原始 ADC 波形，以及用于筛选含正电子事例的模拟轨迹 |
| `ser_truth_J25.7.4+.h5` | 单电子响应、增益分布和电子学噪声标定 |
| `junoid_J25.3.0.pq` | 将 PMT identifier 映射到连续通道号 |
| `PMT_J26.4.0.h5` | 各 PMT 的暗噪声率、光子探测效率和时间偏移 |
| `sim.h5` | 光学响应表 |
| `J25.3.0.PMTPos_CD_LPMT.csv` | PMT 三维位置 |

输入和参考结果不随源码分发。参考结果仅用于校验，不得作为计算程序的输入或直接复制为输出。

## 编译

本项目以 Python 和 CuPy 为主要运行环境。在仓库根目录加载集群模块，创建并激活 `juno` 环境，然后安装依赖：

```bash
cd ~/run/superjuno
module load miniforge3/24.11
conda create -n juno python=3.11 pip
source activate juno
conda install cuda=12.4
pip install -e .
```

环境只需创建一次；后续登录和每次作业中仍需加载上述模块并激活 `juno`。`source` 命令用于初始化当前 Shell 的 Conda 激活功能。依赖见 [pyproject.toml](pyproject.toml)；从源码安装 CuPy 需要 CUDA 开发工具和编译环境。项目中的 CUDA 内核由 CuPy 在运行时编译并缓存，不需要单独生成可执行文件。

## 运行

### 计算流程与固定参数

当前统一入口为 [scripts/run.py](scripts/run.py)，依次执行以下三个阶段：

| 阶段 | 入口 | 计算内容与固定参数 | 输出 |
| --- | --- | --- | --- |
| 1. 先验生成 | `src/fsmp_prior.py` | 筛选含正电子的事例，取波形前 1,000 个采样点；执行 2,000 次 LucyDDM 反卷积，候选阈值为 `0.1` | `output/sparsify/<id>.h5` |
| 2. PE 采样 | `src/fsmp.py` | 执行 4,000 个 MCMC steps；舍弃前半链，每隔 50 步保存一次，共保存 40 组 PE 状态 | `output/PE/<id>.pq` |
| 3. 事例重建 | `src/main.py` | 执行 2,000 个 MCMC steps，能量估计使用 20 个 EM steps，允许一或两个活动顶点 | `output/tvE/<id>.h5` |

FSMP 通过增加、删除或移动一个 PE 来提出候选解释，按规定的概率接受或拒绝；重建阶段联合所有 PMT 的观测与响应，更新 PE 样本、顶点数、位置和时间。没有记录到 PE 的通道同样参与似然计算，不能删去。

所有随机阶段固定 seed 为 `0`，随机量与波形、事例、通道、采样步骤及用途的对应关系必须保持。参考运行器将事例重建分成 10 个事例块；分块数和 GPU batch 大小可以调整，但每条链的计算量与采样规则必须保持。

最终 HDF5 文件中的 `/Recon` 形状为 `(2, number_of_events, 2000)`，保存两个顶点槽位、各事例和完整采样步骤。主要字段包括 `EventID`、`step`、`count`、`x/y/z`、`E`、`t`、`NPE`、`Likelihood` 和 `epsilon`。输出必须保留规定的数据集、字段、单位及逻辑内容。

### Slurm 作业脚本

`scripts/run.py` 使用启动它的 Python 解释器执行计算并计时，运行前会检查所需的六个外部输入。
每次调用都会从第一阶段重新计算并覆盖对应产物和日志。加上 `--compare` 后，
流水线成功时会继续执行 PE 和 tvE 两项比较；省略该选项则只运行计算。

在 `~/run/superjuno/` 下新建 `run.slurm`，参考内容如下：

```bash
#!/bin/bash
#SBATCH --job-name=superjuno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/logs/%x-%j.out
#SBATCH --error=output/logs/%x-%j.err

set -eo pipefail
cd "$HOME/run/superjuno"

echo "Job ID: $SLURM_JOB_ID, Node: $(hostname), Start Time: $(date -Is)"
trap 'status=$?; echo "End Time: $(date -Is), Exit code: $status"; exit "$status"' EXIT

module load miniforge3/24.11
# Activate conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate juno

input_id="${1:-one}"
python scripts/run.py "$input_id" --compare
```

该示例申请单节点、1 张 GPU。

### 提交与查看结果

在仓库目录提交作业。**必须先创建日志目录**，Slurm 会在脚本执行前打开标准输出和标准错误文件。将 `GPU_PARTITION` 替换为主办方分配的 4090 分区名。先用示例点 `one` 验证流程：

```bash
cd ~/run/superjuno
mkdir -p output/logs
sbatch --partition=GPU_PARTITION run.slurm one
squeue -u "$USER"
```

正式评测需分别运行 `0`、`1`、`2` 三个测试点，每个测试点提交一个作业：

```bash
for input_id in 0 1 2; do
  sbatch --partition=GPU_PARTITION run.slurm "$input_id"
done
```

记录各测试点对应的作业编号，提交时用它关联运行日志与生成数据。

作业的标准输出和标准错误分别保存在 `output/logs/superjuno-<JOBID>.out` 和 `output/logs/superjuno-<JOBID>.err`；`%x` 和 `%j` 分别由 Slurm 替换为作业名和作业编号。

运行结束后查看作业标准输出文件 `output/logs/superjuno-<JOBID>.out`。`scripts/run.py` 将耗时、流水线退出码、比较结果或跳过原因，以及最终退出码直接打印到标准输出。其中 `TOTAL` 为三阶段总耗时（不含比较），`Timing:` 记录对应的 `output/logs/<id>-<runid>.time.csv` 文件名及路径；排队时间不计入性能用时。

各阶段的标准输出保存在 `<target>.log`，标准错误汇入上述 Slurm `.err` 文件。计时 CSV 包含各阶段及 `TOTAL` 的起止时间、实际耗时、退出码等信息。临时文件及 CuPy、CUDA 等缓存统一位于 `output/tmp/`。

## 评测指标

### 正确性

三个正式测试点 `0`、`1`、`2` 的提交结果必须分别在 Slurm 作业中由 `python scripts/run.py <id> --compare` 生成。每个测试点的流水线及 PE、tvE 两项比较均须成功，作业标准输出中的 `Pipeline exit code`、两项 `Comparison exit code` 和 `Final exit code` 均须为 `0`。

浮点字段采用以下判据，其中参考值为 $r$，待测值为 $x$：

$$
|x-r| \leq \mathrm{ATOL}+\mathrm{RTOL}\,|r|,
\qquad \mathrm{ATOL}=\mathrm{RTOL}=10^{-6}.
$$

- PE 结果按 `(eid, ch, step)` 对齐，比较 PE 时间、增益成分及相应权重等字段。
- tvE 结果按 `(EventID, step)` 对齐，逐步比较活动顶点和事例级状态；活动顶点按保存的槽位顺序比较。
- 编号、计数、标志和数组长度等离散数据必须精确匹配；记录缺失或多出均不能通过。详细字段与缺失值处理规则见 [test/README.md](test/README.md)。

可对已有输出单独执行校验，用于调试：

```bash
python test/compare_PE.py data/ana_PE/one.pq output/PE/one.pq
python test/compare_tvE.py data/ana_tvE/one.h5 output/tvE/one.h5
```

单独校验不产生完整流程计时；正式提交仍须使用统一运行入口。

### 性能

性能指标为完成规定评测任务的**完整流水线运行时间**，越短越好。单次运行以作业标准输出中的 `TOTAL` 为准，并与 `Timing:` 指向的 CSV 中 `TOTAL` 行的 `real_s` 核对。

计时包含三个计算阶段的执行及其数据读写，不包含随后进行的正确性比较。不得只取某个 GPU 内核或某一阶段的耗时。各队必须使用相同的规定输入、计算规模和资源限制。正式评测分别记录 `0`、`1`、`2` 的完整流水线耗时，按下文规则逐点计分后取平均；示例点 `one` 不参与评分。

`one` 测例通过校验时，作业标准输出中的结果片段如下，耗时和计时文件名仅为示例：

```text
TOTAL: 283.545612 s
Pipeline exit code: 0
Timing: output/logs/one-20261012T092955Z-162914.time.csv
==> [compare_PE] output/PE/one.pq

[PE]
Reference: data/ana_PE/one.pq
Candidate: output/PE/one.pq
0: PASS (atol=1e-06, rtol=1e-06)
Comparison exit code: 0
==> [compare_tvE] output/tvE/one.h5

[tvE]
Reference: data/ana_tvE/one.h5
Candidate: output/tvE/one.h5
0: PASS (atol=1e-06, rtol=1e-06)
Comparison exit code: 0

Final exit code: 0
```

## 评分方式

总分为 100 分，仅对正式测试点 `0`、`1`、`2` 计分。每个测试点先按 **30% 正确性 + 70% 性能** 计算百分制得分，再将三个得分求和取平均。三个测试点等权，每个测试点对总分的最高贡献为 $100/3$ 分，约 **33.33 分**；示例点 `one` 不计分。

对测试点 $i\in\{0,1,2\}$，流水线及 PE、tvE 两项校验全部通过且提交有效时，获得 30 分正确性分；性能分按该测试点最快有效运行时间与本队运行时间之比计算。该点的百分制得分为：

$$
s_i =
\begin{cases}
30 + 70\dfrac{T_{\mathrm{fastest},i}}{T_{\mathrm{team},i}}, & \text{该点正确性通过且提交有效},\\
0, & \text{该点正确性未通过或提交无效}.
\end{cases}
$$

其中，$T_{\mathrm{team},i}$ 为本队在测试点 $i$ 上的完整流水线耗时，$T_{\mathrm{fastest},i}$ 为相同资源限制下，所有通过该点正确性校验的有效提交中的最短运行时间。每个测试点分别确定最快时间。

总分为三个测试点得分的算术平均：

$$
S = \frac{s_0+s_1+s_2}{3}.
$$

例如，三个测试点的百分制得分分别为 100、65、45 分，则总分为 70 分。计分使用精确的 $1/3$ 权重，不将每点满分预先截为 33.33 分；三个测试点均满分时总分为 100 分。

以下情况视为对应测试点的无效提交，该点不计分：

- 修改校验脚本、比较容差、参考结果或计时逻辑；手工改写输出结果或日志。
- 减少规定迭代次数、提前终止采样、跳过计算阶段、删去规定事例／通道／PE，或改变数学模型、采样分布和随机量归属。
- 直接使用参考结果、复用预先计算的测例输出，或针对测例硬编码答案。
- 未通过 Slurm 提交计算任务、跨节点运行、使用超过两张 GPU，或超过主办方规定的其他资源配额。
- 缺失必需的正确性或计时记录，或提交的结果、日志和计时记录不属于同一次运行。

## 提交要求

提交源码或补丁、实际使用的 `run.slurm`、环境与运行说明，以及 `report.md` 或 `report.pdf`。报告应说明发现的性能瓶颈、优化方法、运行配置，并分别列出测试点 `0`、`1`、`2` 的作业编号、正确性结果、优化前后的完整流程耗时和加速比，鼓励提供分阶段耗时图表。提交渠道和截止时间以比赛通知为准。

**必须提交测试点 `0`、`1`、`2` 全部三个点的日志和生成数据；示例点 `one` 无需提交。** 对每个正式测试点，将下表中的 `<id>` 替换为 `0`、`1` 或 `2`，提交该点同一次运行产生的以下文件：

| 内容 | 文件 |
| --- | --- |
| 计时记录 | 作业标准输出中 `Timing:` 指向的 `output/logs/<id>-<runid>.time.csv` |
| 生成数据 | `output/sparsify/<id>.h5`、`output/PE/<id>.pq`、`output/tvE/<id>.h5` |
| 计算日志 | `output/sparsify/<id>.h5.log`、`output/PE/<id>.pq.log`、`output/tvE/<id>.h5.log` |
| 比较报告 | `output/PE/<id>.pq.compare.txt`、`output/tvE/<id>.h5.compare.txt` |
| 汇总结果与 Slurm 运行日志 | `output/logs/superjuno-<JOBID>.out`（含耗时、比较结果和退出码）、`output/logs/superjuno-<JOBID>.err`（对应本次测例作业） |

保留上述相对路径，不得手工改写结果或计时数据。同一测例再次运行前先归档，避免覆盖。原始数据和参考结果无需重复提交。
