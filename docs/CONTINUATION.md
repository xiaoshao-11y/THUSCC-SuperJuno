# SuperJUNO 后续实施交接（CONTINUATION）

记录日期：2026-10-17。本文是 `README.md` / `IMPLEMENTATION.md` 之后的**当前状态与
后续路线**文件：进展快照、输入与参考结果边界、比赛 RTX4090 集群的重验清单、可继续
的本地任务与正式验收清单。历史实施与审查细节以 `IMPLEMENTATION.md` 为准；本文只记
当前事实，不重复其内容。状态词汇：`LOCAL_PASS` / `CLUSTER_PENDING` /
`DATA_PENDING` / `MODEL_PENDING`。

## 1. 进展快照（2026-10-17 核对；2026-10-18 修复见本节开头）

### 2026-10-20 下午：三点全部正式通过（当前最新状态，里程碑完成）

**正式结果（配置 `SJ_SHARDS=2 SJ_GPUS=2 SJRNG=1 SJ_CPUS=8`，钉 g0059,g0050）**：

| 测例 | 作业 | TOTAL (s) | vs 基线 | 判定 |
|---|---|---|---|---|
| 0 | 1334370 | 5624.2 | 1.48× | PE PASS tvE PASS Final 0 |
| 1 | 1334350 | 5624.8 | 1.53× | PE PASS tvE PASS Final 0 |
| 2 | 1334083 | 5558.1 | 1.53× | PE PASS tvE PASS Final 0 |

平均 **1.51×**。集群 `~/run/superjuno/output/` 三点产物/time.csv/compare 原地
保留；本地证据 `../records/cluster_20261019/`。**集群产物在正式提交前不得再动**
（下次提交同 id 会归档——若需重跑请先整体打包备份）。

**剩余提交工作（不在 zcode 职责内自动执行）**：report.md（瓶颈=阶段2 65%、方法
=位级一致分片+SJRNG+行序/CPU 上下文修复、作业号/正确性/加速比表、诚实的调查
叙述）、源码补丁（upstream 6 文件 diff + slurm 脚本 + 本层）、提交渠道待主办方
通知；sczf074/BSCC-N16R4 归属确认仍待客服。


**根因（判别实验 testR/testC 逐位证明，见 `../IMPLEMENTATION.md` §13）**：tvE 失败
= ①分片合并改变了 PE parquet 行序（FullReader 稳定排序按行序裁决平局→块成员变
→链发散）+ ②gpu:2 作业默认 16 CPU ≠ 参考上下文的 8 CPU。与节点/CVD/缓存环境无关。

**修复**：fsmp.py 分片 worker 每 part 写独立 parquet、父进程按 part 序号合并
（=单进程行序）；submit.sh `SJ_CPUS`（默认 8）。回归 52/52。

**官方跑（配置 `SJ_SHARDS=2 SJ_GPUS=2 SJRNG=1 SJ_CPUS=8`，钉 g0059,g0050）**：
- ✅ **case 2 = 1334083（g0050）：双 PASS、Final 0，TOTAL 5558.1s（1.53×）**，
  产物在集群原地未动（time.csv: 2-20261020T030856Z-156163），证据已下载本地。
- ⏳ case 1 = 1334350（g0050，~14:50 完成）
- ⏳ case 0 = 1334370（排队等 g0059/g0050 释放）
- 佐证：1333810（纯基线上下文复刻，g0059）双 PASS、8415.9s——基线可复现。

**恢复步骤（papp_cloud 令牌 13:5x 过期 E304）**：用户 `../doc/papp_cloud_login.sh
--login` 重新认证后：1) 逐个收 1334350/1334370 的
`grep -aE "TOTAL|PASS|FAIL|Comparison exit|Final exit|Timing:" output/logs/superjuno-<J>.out`；
2) 双 PASS + Final 0 → 三点齐 → 下载全部 .out/time.csv 到
`records/cluster_20261019/`；3) 更新 IMPLEMENTATION/README/CONTINUATION 与
session_record.json；4) 报告素材：瓶颈（stage2 66%）、方法（位级一致的双 GPU
分片 + C++ RNG + 行序保持 + CPU 上下文对齐）、作业号/正确性/加速比（1.53×）。
若任一点 FAIL：核对失败事例数与 PE 是否 PASS，检查该作业的节点与 `--cpus-per-task`
是否生效（.out 的 Opt-in 行 + `sacct -j <J> -o AllocTRES`）。

### 2026-10-19 深夜：优化实验与 tvE 发散调查



**基线（全部双 PASS、单 GPU 上游路径）**：case0 8320.1s（g0059）、case1 8601.5s
（g0050）、case2 8486.3s（g0050）；剖面 stage2 65-66%/stage3 27-29%/stage1 5-8%。

**已定结论**（详细判别实验链见 `../IMPLEMENTATION.md` §13 与
`../records/cluster_20261019/session_record.json`）：
- ✅ **阶段2 分片（SJ_SHARDS）逐位一致**：正式规模 PE 与基线归档逐位全等；
  阶段2 提速 **1.95×**（含 SJRNG）。SJRNG 正式规模逐位一致。
- ❌ **阶段3 分片、SJ_BLOCKS≠10、BETA_INV 均不可用于正式跑**：tvE 链发散机制
  （ulp 漂移放大越界）。`SJ_RECON_SHARDS` 已隔离（默认 1=上游）。
- ⚠️ **tvE 结果与节点绑定**：候选配置（1.49×，PE 逐位一致 + 阶段3 原样）在
  g0063 仍 tvE FAIL；g0059/g0050（基线节点）匹配参考，其余测试节点为另一位
  稳定签名。硬件/VBIOS/驱动全同，根因未定；缓解 = `SJ_NODE` 钉节点
  （submit.sh 已支持）。
- 🎯 **正式候选跑已提交（2026-10-19 23:45）**：`SJ_SHARDS=2 SJ_GPUS=2
  SUPERJUNO_SJRNG=1 SJ_NODE=g0059,g0050` → 作业 **1332822（case0）/1332823
  （case1）/1332824（case2）**，与钉节点对照 1332775/76 一起排队（节点满载）。

**收结果步骤**（下次会话）：`squeue -u sczf074` 确认完成后，对每个作业
`grep -aE "TOTAL|PASS|FAIL|Comparison exit|Final exit|Timing:" 
output/logs/superjuno-1332XXX.out`。三点全部双 PASS + Final 0 → 这就是正式提交
候选（time.csv、产物、compare、.out/.err 同次运行原地保留）；任一 FAIL →
分析失败事例数（若 ~2-3 个 → 节点签名仍未命中，检查对照 1332775/76 判定
g0059/g0050 是否自洽，必要时单钉 g0059 重跑）。加速比写报告时用
5577.2/8320.1≈1.49× 量级并注明节点。


### 2026-10-19 下午：官方优化方向实施


按官方明示可调项（分块数、GPU batch、≤2 GPU）实施优化，全部 env-gated、默认=上游；
详见 `../IMPLEMENTATION.md` §13。快照：

- `one` 上已验证（官方 `--compare` 双 PASS）：SJ_BLOCKS=5（1332124）、
  BETA_INV=1（1332185）；SJRNG=1 前轮已过。SUPERJUNO_VRAM_DIV=2 实测**回归**
  （U+F≈20GB 内存受限，已取消）——默认保持上游 /5。
- **在队/在跑（Slurm 自主运行，不依赖本机连接）**：基线 0/1/2 = 作业 1332023/24/25
  （运行 1h+，阶段2 大头）；SJ_SHARDS=2 验证 = 作业 1332236（待跑）。
- 集群侧代码已是最新（fsmp.py 分片版、run.py SJ_BLOCKS、run.slurm 回显）。
- **中断点**：2026-10-19 ~16:50 papp_cloud 令牌过期（E304）。恢复步骤：用户运行
  `../doc/papp_cloud_login.sh --login` 重新认证 → `papp_cloud ssh sczf074@bscc-n16r4`
  核验 → 收取上述作业结果 → 继续 §13 待做：正式点 SJ_BLOCKS 扫描、SJ_SHARDS=2
  实验、开关组合，全部过 `--compare` 后才可用于正式提交。

### 2026-10-19 集群开通：环境/数据/`one` 验证完成


集群（papp_cloud → ln01，`gpu` 分区 RTX 4090）在本日一次会话内完成部署与调试，
详细记录与证据见 `../IMPLEMENTATION.md` §12 与 `../records/cluster_20261019/`
（含 `session_record.json` 与 6 个作业的原始 .out/.err/time.csv）：

- **部署**：代码上传 `~/build/SuperJUNO`；官方 `data.tar.gz`（2.88GB）登录节点
  curl 下载并解压至 `~/run/superjuno/data`；juno 环境按官方命令创建
  （python 3.11.17、conda cuda 12.4、CuPy 14.2.0 sdist 编译、**numpy 2.4.6**
  ——与本地逐位移植验证同版本）。
- **修复的两个集群部署问题**（数值代码零改动）：① sbatch 批处理 shell 无
  `module` 命令 → `run.slurm` 加 modules 初始化兜底；② `conda activate` 导出
  `CXXFLAGS/LDFLAGS` 顶掉 Makefile `?=` 默认值（丢 `-shared`）→ 构建作业在
  make 前 `unset`（`build_test.slurm`）。
- **验证结果（全部经 sbatch + 官方 `--compare`）**：作业 1331820 重编 `.so` +
  位级对照 **18/18 OK**；作业 1331839 基线 `one` 双 PASS（冷 JIT 486.8s）；
  作业 1331852 `SUPERJUNO_SJRNG=1` 双 PASS、C++ 后端确认激活（312.1s）；
  作业 1331864 热缓存基线双 PASS（**305.5s**）。
- **诚实结论**：`one` 仅 1 事例，SJRNG 端到端无收益（305.5 vs 312.1，噪声内）；
  其集群结论是**正确性验证通过**。性能收益须在正式点 0/1/2 重测后再决定开关
  默认值。CuPy JIT 缓存在 `output/tmp/` 跨提交存活，冷/热差异 ~180s 只属首跑。
- **仍未发生**：`SUPERJUNO_BETA_INV` 的 `one` 验证、正式点 0/1/2 提交、任何
  加速比宣称；sczf074 / BSCC-N16R4 是否为比赛指定 4090 资源仍待确认。
- **下一步**（按 §3 清单序号）：第 4 步补 `SUPERJUNO_BETA_INV=1` 单开关 `one`
  验证 → 第 5 步用已验证开关组合提交 0/1/2 → 核对各点 TOTAL 与 time.csv。

### 2026-10-18 修复：upstream git 溯源环境回归（当前最新状态）

**回归**（Codex 2026-10-18 实测，本机复现）：2026-10-18 为把工作纳入备份仓库，
新建了 `SuperJUNO/.git`（单提交 `eb23a44`，远端为用户私有备份库），并把官方克隆的
`.git` 整体移至 `upstream/official_git_backup/`（过程见 `upstream/SOURCE.md`）。
此后 `git -C upstream …` 沿目录向上解析到**备份仓库**，其中没有官方 commit 对象，
`git -C upstream show 82f25ea…:src/analysis/fsmp/prepare.py` 以 exit 128 失败
（git 报 path does not exist in '82f25ea…'）。后果：`make test` 18 项 RNG 测试中
4 项错误（`TestAgainstExtractedUpstream` 全部 4 个方法，参照系经 `git show` 从
不可变 git 对象提取），监督脚本 `../doc/verify_juno_supervisor.py` 的同一提取
前置同样失败（外层 `make supervisor-check` 因此退出码 2）。这是**溯源环境回归，
不是 C++ RNG 算法错误**；此前"23/23 通过"的前提被破坏，恢复后需重新生成。

**修复方式**（未改任何验证脚本、参考提取逻辑、官方 runner、比较器，未动官方
对象库；仅两个文件级改动）：

1. 新增 `upstream/.git`：gitfile，单行 `gitdir: official_git_backup`（相对路径，
   Git 对 gitfile 的标准仓库/工作树发现机制；git 2.56 实测
   `git -C upstream rev-parse --show-toplevel` = `upstream/`、`--git-dir` =
   `upstream/official_git_backup`）。`official_git_backup/` 目录本身未移动，
   其 config/refs/objects/index 均未改动。
2. `official_git_backup/info/exclude` 追加一行 `/official_git_backup/`：该 gitdir
   位于自身工作树内，排除后 `git -C upstream status` 不再把它列为未跟踪噪音
   （不改被 git 跟踪的 `.gitignore`，只改 gitdir 私有的 exclude）。
3. 外层备份仓库不受影响：`.git` 路径 Git 永不跟踪/列出（其 status 仍只有原有
   3 个未跟踪的任务跟踪文件）；根 `.gitignore` 第 8 行本就排除
   `upstream/official_git_backup/`，备份工作流不变。

**恢复后核对**（实际命令与退出码，原始日志
`zcode_provenancefix20261018_gitverify.log`）：

- `git -C upstream show 82f25ea67f10ede8881f4dd4656ad78cc40cfd5f:src/analysis/fsmp/prepare.py`
  → exit 0，6931 字节（与 Codex 直接从 `official_git_backup` 实测的约 6.9KB 一致）。
- `git -C upstream rev-parse HEAD` → `82f25ea67f10ede8881f4dd4656ad78cc40cfd5f`
  （exit 0，与 `upstream_commit.txt` 一致）。
- `git -C upstream status --porcelain` → 仅 ` M src/analysis/fsmp/move.py`、
  ` M src/analysis/fsmp/prepare.py`（`diff HEAD --stat` = 2 文件 +36/−1，与
  IMPLEMENTATION.md §7 记录一致）+ 未跟踪 `src/sjrng.py`、`src/_sjrng_core.so`
  （候选 `.so` 及封装）、`SOURCE.md`（2026-10-18 新增的来源说明）。
- `git -C upstream diff HEAD --stat -- scripts/ test/ docs/ pyproject.toml` → 空
  （官方 runner、比较器、计时逻辑零改动）；上述全部工作文件均在磁盘上原样保留。

**两项本地验证（2026-10-18 本机复跑，恢复前提后重新生成）**：

- `make test` → **exit 0**：`cpp/test_sjrng.py` 18/18 OK（含此前失败的 4 项
  git 对象提取对照）、`cpp/test_fused_formula.py` 4/4 OK、`cpp/test_install.py`
  3/3 OK、`slurm/test_submit_controlflow.sh` 52/52（`control-flow regression:
  52 passed, 0 failed`，桩 sbatch/squeue，无真实集群交互）；`make install` 重建
  并安装 `.so`。原始日志 `zcode_provenancefix20261018_make_test.log`。
  微基准本轮 1.4×（numpy 循环 8.618 s vs C++ 6.055 s；受内存带宽/负载波动，
  仅 LOCAL 参考，与流水线耗时无关）。
- `python3 ../doc/verify_juno_supervisor.py` → **exit 0，23/23 LOCAL_PASS**
  （16 组随机位级对照 + 7 组监督者桩下提交控制流）；报告由脚本按其设计重写至
  `../doc/supervisor_verification_SuperJUNO.json`。原始日志
  `zcode_provenancefix20261018_verify.log`。

**剩余问题**：官方 `--compare` PE/tvE、GPU/CUDA 运行、真实 sbatch 提交、端到端
加速比仍全部 `CLUSTER_PENDING` / `DATA_PENDING`（§3/§6 清单不变）；本轮无任何
数值代码改动、无新增优化。一条环境注意项：从 GitHub 备份仓库重新克隆时
`official_git_backup/` 不随备份下发（根 `.gitignore` 排除），gitfile 会悬空——
灾备恢复方法（重新克隆官方仓库并 checkout `82f25ea` 后移回）已写入
`upstream/SOURCE.md`，本轮未执行。备份仓库中 `upstream/official_git_backup/`
的 `info/exclude` 改动不进备份（该目录整体被忽略），重建时按 SOURCE.md 重做。

以下为 2026-10-17 核对时的快照（git 溯源表述以本小节 2026-10-18 修复后为准）：

- 上游 `upstream/` HEAD = `82f25ea67f10ede8881f4dd4656ad78cc40cfd5f`，与
  `upstream_commit.txt` 一致。`git status` 仅两处 env-gated 修改
  （`src/analysis/fsmp/move.py`、`src/analysis/fsmp/prepare.py`）与两个纯新增文件
  （`src/sjrng.py`、`src/_sjrng_core.so`）；`scripts/`、`test/`、`docs/`、
  `pyproject.toml` 相对 HEAD **零改动**（`git diff HEAD --stat -- scripts/ test/ docs/`
  为空）。两处修改的默认分支保留上游原句原文；`SUPERJUNO_SJRNG` /
  `SUPERJUNO_BETA_INV` 均要求显式 `== "1"` 才生效，**默认关闭**。
- 本轮（2026-10-17）在本机复跑全部本地验证，全部通过：
  - `make test` → `cpp/test_sjrng.py` 18/18 OK、`cpp/test_fused_formula.py` 4/4 OK、
    `cpp/test_install.py` 3/3 OK、`slurm/test_submit_controlflow.sh` 52/52
    （桩 sbatch/squeue，无任何真实集群交互）；`make install` 重建并安装 `.so`。
  - `python3 ../doc/verify_juno_supervisor.py` → exit 0，23/23 LOCAL_PASS，报告
    重写至 `../doc/supervisor_verification_SuperJUNO.json`（脚本现位于公共
    `../doc/`，2026-10-16 运行时位于上级根目录，历史记录已加注）。
- 仍未发生（不宣称）：任何官方 `--compare` PE/tvE 结果、任何 GPU/CUDA 运行、任何
  真实 sbatch 提交、任何端到端加速比。全部仍为 `CLUSTER_PENDING` / `DATA_PENDING`。
- 环境入口（2026-10-17 用户纠正）：本题使用官网要求的**比赛 RTX4090 集群**——经并行
  科技云桌面 SSH 接入、Slurm sbatch 提交、单节点最多 2 GPU；**不创建独立 RTX4090
  云实例**，SuperShip 的 RTX3090 实例不用于本题。2026-10-19 只读 SSH 已恢复，
  `sbatch`/`squeue` 与 `gpu` 分区可见；状态为 `LOGIN_RESTORED / RTX4090_CONFIRMED / DATA_PENDING`
  （详见 §3 前提与 §4；资源现状 `../doc/RESOURCE_HANDOFF.md`）。
- 本轮未发现数值/控制流实现缺陷，**未改任何数值代码**；仅修正文档（监督脚本路径、
  submit.sh 输入检查的"共享文件数"表述，见 §5）。
- 附带观察：`make test` 内含的微基准本轮打印 1.2×（0.918 s vs 0.796 s；受内存带宽
  影响波动，2026-10-16 曾为 1.9×）。微基准仅是 LOCAL 参考，与流水线耗时无关。

## 2. 输入与参考结果边界（2026-10-17 只读核对）

`scripts/run.py`（`build_pipeline`，不可改）预检**六类计算输入**，缺一即
`Missing input` 退出：

| # | 路径 | 作用 |
|---|---|---|
| 1 | `data/raw/<id>.h5` | 原始 ADC 波形（per-id） |
| 2 | `data/ser_truth_J25.7.4+.h5` | 单电子响应/增益/噪声标定（共享） |
| 3 | `data/sim.h5` | 光学响应表（共享） |
| 4 | `data/PMT_J26.4.0.h5` | PMT 暗噪声/PDE/时间偏移（共享） |
| 5 | `data/junoid_J25.3.0.pq` | PMT id → 通道映射（共享） |
| 6 | `data/J25.3.0.PMTPos_CD_LPMT.csv` | PMT 三维位置（共享） |

- `<id>` ∈ {one, 0, 1, 2}：计算输入 = `data/raw/{one,0,1,2}.h5` 各一份 + 5 个共享
  文件；三正式点不共享 raw。
- `data/ana_PE/<id>.pq`、`data/ana_tvE/<id>.h5` **只被 `--compare` 时的
  `test/compare_PE.py` / `compare_tvE.py` 读取**，不是任何计算阶段的输入，不得作为
  输入或复制为输出（官方无效提交条款）。
- `slurm/submit.sh` 提交前检查全部 8 个文件（上述六类 + 两个该 id 的参考结果，后者
  为 `--compare` 所需），与 run.py 预检一致且更严格。
- **本地存在性结论（2026-10-17，只在 SuperJUNO/ 项目内查找，未搜索主目录）**：
  项目内没有任何 `data/` 目录，没有任何 `.h5`/`.pq`/`data.tar.gz` 输入或参考文件；
  也没有任何 `output/` 运行产物。状态 `DATA_PENDING`：数据需按官方地址
  （https://cloud.tsinghua.edu.cn/d/0bc5df5078e84dca81a2/ 的 data.tar.gz）在集群
  `~/run/superjuno/` 下解压获得，本机不擅自构造、不猜测未发布内容。

## 3. 比赛 RTX4090 集群必须重验的步骤（RTX4090_CONFIRMED / DATA_PENDING 清单）

前提：本题按官网要求使用**并行科技 RTX4090 集群**——经并行科技云桌面 SSH 接入、
Slurm sbatch 提交、单节点最多 2 GPU（`docs/official.txt`；2026-10-17 复核快照
`../doc/records/cluster_route_20261017/`）。**不创建独立 RTX4090 云实例**；用户当前
只创建了 SuperShip 的 RTX3090 实例，不得当作本题环境，不得用它做本题计算。

集群连接已恢复（2026-10-19）：本机 papp_cloud 重新认证后，`papp_cloud ssh
sczf074@bscc-n16r4` 只读进入登录节点 `ln01`，`sbatch`、`squeue` 可用。诊断作业
`1331701` 使用 `gpu` 分区、1 张 GPU，在计算节点 `g0050` 运行 `nvidia-smi`，实测
**NVIDIA GeForce RTX 4090、24564 MiB、驱动550.135、compute capability 8.9**；该作业
没有运行 SuperJUNO 程序。历史连接故障仍保留在
`../doc/history/2026-10-16/gpu_connection_probe.json`。记录见
`../doc/records/cluster_retry_20261019.json`、
`../doc/records/cluster_gpu_type_probe_20261019.json` 和 `../doc/RESOURCE_HANDOFF.md`。

本机 GCC 16 构建的 `_sjrng_core.so` 不可直接投放目标节点。数据未就位前一切流水线
命令都会在 run.py 预检处退出。登录节点仅允许查询与文件管理：**C++ 重编、NumPy
位级对照与一切计算都按规定在 sbatch 分配的计算资源上执行**（下列第 1、2 步）。

按序执行（详细部署命令见 README「集群部署与提交」）：

1. **目标系统重编 C++ 库**：经云桌面 SSH/快传上传本层后，用一次性 sbatch 作业在
   **计算节点**执行 `make install`（或 README 中等价的显式
   `g++ -std=c++17 … -pthread …` 命令），生成 `upstream/src/` 下的
   `sjrng.py` + `_sjrng_core.so`；**不在登录节点编译**。
2. **对当场 NumPy 版本位级对照**（启用 `SUPERJUNO_SJRNG` 的硬前置）：同样经
   sbatch 作业在计算节点用集群 python3（juno 环境）跑 `cpp/test_sjrng.py`
   （18 项全过，含从不可变 git 对象 AST 提取的上游真函数对照）；**不在登录节点
   跑数值对照**。不通过 → 该开关在集群禁用，只保留上游 numpy 路径。
   本机对照版本为 numpy 2.4.6，跨版本无位流保证（`docs/numpy_compatibility_notes.md`）。
3. **one 基线完整官方比较**（两开关全关）：`SJ_PARTITION=$P ./slurm/submit.sh one`
   → `./slurm/status.sh`；要求 `.out` 中 Pipeline exit code 0、两项 Comparison
   exit code 0、Final exit code 0；记录 `TOTAL` 与 `Timing:` CSV 为基线。
4. **逐开关 one 验证**：`SUPERJUNO_SJRNG=1`（预期输出与基线逐位一致）与
   `SUPERJUNO_BETA_INV=1`（数值等价路径，以 `--compare` 判据为准）分别提交 one；
   任一比较失败立即弃用该开关并归档证据。
5. **正式点 0/1/2**：仅用已通过第 4 步的开关组合
   `SJ_PARTITION=$P ./slurm/submit.sh 0 1 2`；每个点独立作业，作业 stdout 的
   `TOTAL` 须与 time.csv TOTAL 行 `real_s` 核对；三点的生成数据/.log/.compare.txt/
   time.csv/`superjuno-<JOBID>.out/.err` 保持同次运行原地不动。
6. run.slurm 作业头会打印 python/CuPy/numpy/GPU 版本与开关取值（不进计时），
   供环境差异审计。

## 4. 调度与提交规则边界（不可自行放宽）

- 官方快照（`docs/official.txt`；2026-10-17 复核 `../doc/records/cluster_route_20261017/`）
  明确：本题运行平台为**并行科技 RTX4090 集群**，正式提交经 Slurm `sbatch`，单节点、
  每次最多 2 GPU；先 `one` 验证再分别提交 0/1/2；登录节点禁止运行计算。
- 本题**不创建独立 RTX4090 云实例**：云服务器实例路线（`../doc/GPU_CLOUD_SERVER_GUIDE.md`）
  属于 SuperShip/RTX3090，与本题无关。**任何无 Slurm 的直接运行只能算调试
  （相当于 LOCAL），不得记为正式提交成绩**；不得自装假 Slurm、不得捏造作业编号、
  不得手写运行日志。若主办方后续明确新提交口径，以官方新文为准并更新本节。
- `slurm/run.slurm` 自检：非 Slurm 环境 / 多节点 / GPU>2 → exit 97；产物已存在未设
  `SJ_OVERWRITE` → exit 98；参数非法 → exit 99（均在任何输出路径创建之前）。
  `submit.sh` 另有锁、活跃作业（squeue 按用户清单求交，不用 `squeue -j`）、输入
  齐全、防覆盖归档检查。全部仅经本地桩验证（52/52），真实集群行为
  `CLUSTER_PENDING`。

## 5. 可继续的本地任务（无数据/无 GPU 也可做；不加新数值优化）

1. 复跑本地验证并记录（每次有意义的改动后）：
   `make test` 与 `python3 ../doc/verify_juno_supervisor.py`（预期 18+4+3+52 与
   23/23；监督脚本与报告在公共 `../doc/`）。
2. 官方材料跟踪：客服/主办方确认 4090 集群账号/分区（含 sczf074 / BSCC-N16R4
   归属核对）或发布本题数据/新提交口径时，核对 `docs/official.txt` 与 `../doc/`
   快照是否需要补充，并同步更新本文 §3/§4 与 README 部署命令（当前以 2026-10-17
   官网复核快照 `../doc/records/cluster_route_20261017/` 为准）。
3. 文档维护（本轮已完成一版）：`IMPLEMENTATION.md`/`README.md` 的监督脚本路径已
   改为 `../doc/verify_juno_supervisor.py`（报告 `../doc/…json`）；submit.sh 输入
   检查表述已更正为"共享五个输入文件"（`ser_truth`/`sim`/`PMT`/`junoid`/`PMTPos`，
   即 run.py 预检的六类计算输入）+ 两个该 id 参考结果，共 8 个文件。后续发现类似
   漂移照此最小修正，只改文字不改代码。
4. 缺陷处理原则：先记录触发条件与证据，再按原始随机映射与数学等效性做最小修复并
   跑直接相关测试；**未经官方 `--compare` 通过的优化一律不接入、不默认启用**
   （含 `gpu/vmove1_fused.cu` 草稿——保持未接入、不再扩展）。

## 6. 正式验收清单（全部满足才可宣称完成）

- [x] 比赛 RTX4090 集群接入可用（云桌面 SSH 恢复且账号/分区经客服/主办方确认——
      含 sczf074 / BSCC-N16R4 归属核对；`module load` CUDA 12.4、Miniforge3
      24.11、juno conda 环境，`pip install -e .` 成功，环境构建按官方允许的方式
      在规定节点上进行）
      —— 2026-10-19：环境/数据/one 验证完成（见 §1 最新快照与 IMPLEMENTATION
      §12）；**账号/分区归属确认仍待客服/主办方**，其余子项已满足。
- [x] `data/` 八类文件就位（§2；one/0/1/2 的 raw + 5 共享 + 各 id 参考结果）
      —— 2026-10-19 于 `~/run/superjuno/data`（2.9G）。
- [x] 目标节点 `make install` 重编 + `cpp/test_sjrng.py` 位级对照通过（或据此
      禁用 SJRNG 并记录）
      —— 2026-10-19 作业 1331820：18/18 OK（集群 numpy 2.4.6）。
- [x] one 基线（默认路径）`--compare` 三项 exit code 全 0
      —— 2026-10-19 作业 1331839 / 热缓存 1331864。
- [x] 每个启用中的开关单独在 one 上过 `--compare`
      —— 2026-10-19 作业 1331852（SUPERJUNO_SJRNG=1，双 PASS）；
      **SUPERJUNO_BETA_INV 仍未验证**（未启用即未违反；启用前必须补验）。
- [ ] 0/1/2 各自 Slurm 作业：Pipeline/两项 Comparison/Final exit code 全 0，
      TOTAL 与 time.csv `real_s` 一致，作业编号与产物同次运行对应
- [ ] 提交物齐全：time.csv、`output/sparsify|PE|tvE/<id>.*` 与 `.log`、
      `.compare.txt`、`superjuno-<JOBID>.out/.err`（0/1/2 三点，one 免提交）
- [ ] report.md 写明瓶颈、方法、配置、各点作业编号/正确性/耗时与加速比
      （只引用作业 TOTAL 与 time.csv，不引用本地微基准）

以上任何一项未完成时，相应结论只能标 `CLUSTER_PENDING` / `DATA_PENDING`，不得写成
已通过。
