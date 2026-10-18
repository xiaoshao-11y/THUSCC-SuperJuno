# PMT waveform reconstruction pipeline

本仓库提取了一条实际粒子物理实验数据分析流程中的核心计算：从 PMT 波形采样
光电子（PE）到达时间，再联合全部 PMT 完成最多双顶点的事例重建。统计模型、
公式和数据契约见
[`docs/recon.tex`](docs/recon.tex)。

## 仓库构成

| 路径 | 内容 |
|---|---|
| `scripts/run.py` | 使用当前 Python 环境运行三阶段计算、记录耗时并按需比较结果 |
| `src/fsmp_prior.py` | 用 LucyDDM 从原始波形生成 FSMP proposal |
| `src/fsmp.py` | 执行波形层 FSMP，保存 PE 时间序列样本 |
| `src/main.py` | 执行最多双顶点的事例层重建 |
| `src/recon.py`, `src/mcmc.py`, `src/detector.py`, `src/fsmp_reader.py` | 重建、采样、响应和输入读取实现 |
| `src/wave_reader.py`, `src/wave_reader_impl.py` | 原始波形读取与通道映射 |
| `src/analysis/` | LucyDDM、FSMP batch/proposal、GPU kernel 和输出辅助代码 |
| `test/` | 基准与待测输出的数值一致性检查 |
| `docs/recon.tex` | 算法和数据格式的完整技术文档 |

`data/` 是外部输入目录；`output/sparsify/`、`output/PE/` 和 `output/tvE/` 是运行时生成的中间或
最终产物。这些目录均不纳入 Git。

## 数据

数据不包含在仓库中，请从以下地址下载：

https://cloud.tsinghua.edu.cn/d/0bc5df5078e84dca81a2/

将下载的数据放在仓库根目录的 `data/` 下。`scripts/run.py` 所需的计算输入如下：

```text
data/
├── raw/
│   └── <id>.h5
├── J25.3.0.PMTPos_CD_LPMT.csv
├── PMT_J26.4.0.h5
├── junoid_J25.3.0.pq
├── ser_truth_J25.7.4+.h5
└── sim.h5
```

原始波形文件名中的 `<id>` 是传给 `scripts/run.py` 的测例标识。例如
`data/raw/one.h5` 对应 `output/tvE/one.h5`。

共提供示例点 `one` 和正式测试点 `0`、`1`、`2`。`one` 用于调试与验证环境，
不参与评分；三个正式测试点均需运行并提交各自的日志和生成数据。

## 运行

计算环境需要 Python 3.10 及以上、CUDA 12+、CuPy 及科学计算依赖（见 `pyproject.toml`）。
可在已经准备好的环境中安装：

```bash
python -m pip install -e .
```

然后使用示例点 `one` 运行完整流程：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py one
```

脚本使用启动它的 Python 解释器，也可以显式指定当前环境中的解释器：

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/python scripts/run.py one
```

`scripts/run.py` 会预先检查六个外部输入，再依次运行 `src/fsmp_prior.py`、
`src/fsmp.py` 和 `src/main.py`，记录分阶段及总耗时。任何阶段失败都会立即停止。
每次调用都会从第一阶段重新计算并覆盖对应产物和日志；需要保留的结果请先归档。

进程临时目录以及 CuPy、CUDA、XDG、Python 字节码和 probe 缓存统一位于
`output/tmp/`。每个阶段的标准输出写入 `<target>.log`，标准错误直接输出到
当前进程的 stderr。

每次运行还会生成 `output/logs/<id>-<runid>.time.csv`，记录各阶段的开始时间、结束
时间、wall time、退出码和产物字节数，最后以 `TOTAL` 行记录完整流水线耗时。

成功完成、阶段失败、输入缺失或中断时，脚本都会在 stdout 输出
`TOTAL: <秒数> s`，与 CSV 的 `TOTAL` 耗时一致，不包含正确性比较耗时。
运行结果直接打印到 stdout，包括耗时、流水线退出码、比较结果或跳过原因，
以及最终退出码；其中 `Timing:` 行记录本次 `.time.csv`
计时文件的路径（相对于仓库根目录）。

默认只运行流水线；加上 `--compare` 会在流水线成功后依次检查 PE 和 tvE：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py one --compare
```

比较使用下文列出的基准文件，并继续生成各自的 `.compare.txt` 报告。
一个比较失败后仍会执行另一个；任一比较失败时 `run.py` 返回非零退出码。
流水线失败或中断时会跳过比较，并在 stdout 中打印原因。

主要产物顺序为：

```text
data/raw/<id>.h5
  -> output/sparsify/<id>.h5
  -> output/PE/<id>.pq
  -> output/tvE/<id>.h5
```

每个阶段的产物旁会生成 `.log`。

## 正式评测与提交

正式评测需在 Slurm 作业中分别执行 `python scripts/run.py <id> --compare`，
其中 `<id>` 依次为 `0`、`1`、`2`。集群环境和作业脚本见
[`problem.md`](problem.md#slurm-作业脚本)。

三个测试点均须提交各自的生成数据（`output/sparsify/<id>.h5`、
`output/PE/<id>.pq`、`output/tvE/<id>.h5`）、三阶段 `.log`、两项比较报告、
计时 CSV 及 Slurm 标准输出和标准错误日志。同一测试点的所有文件须来自同一次运行，
并在报告中注明对应的作业编号。示例点 `one` 无需提交。

总分为 100 分。三个正式测试点各按 30% 正确性和 70% 性能计算百分制得分，
再将三个得分求和取平均，即 `S = (s_0 + s_1 + s_2) / 3`。
每个测试点对总分的最高贡献为 `100/3` 分（约 33.33 分）。完整评分规则和提交清单见
[`problem.md`](problem.md#评分方式)。

## 正确性比较

基准文件使用当前数据目录中的 `data/ana_PE/<id>.pq` 和
`data/ana_tvE/<id>.h5`。待比较的新版本输出位于 `output/PE/<id>.pq` 和
`output/tvE/<id>.h5`。在仓库根目录直接运行以下脚本可检查已有输出的数值一致性：

```bash
python test/compare_PE.py data/ana_PE/one.pq output/PE/one.pq
python test/compare_tvE.py data/ana_tvE/one.h5 output/tvE/one.h5
```

每次调用都会重新检查，以英文 `CODE: MESSAGE` 在 stdout 输出结果，并将相同内容
覆盖写入 `output/PE/<id>.pq.compare.txt` 或 `output/tvE/<id>.h5.compare.txt`。
浮点数的绝对和相对容差由 [`test/comparison.py`](test/comparison.py) 中的
`ATOL` 和 `RTOL` 统一设置，编号、计数等离散数据必须一致。
状态码 `0` 表示通过，`1` 表示结果不一致（最多输出前十条异常），`2` 表示输入
或执行错误；脚本退出码与状态码相同。
完整比较规则见 [`test/README.md`](test/README.md)。

## 编译技术文档

`docs/recon.tex` 使用 XeLaTeX：

```bash
cd docs
latexmk -xelatex recon.tex
```
