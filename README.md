# SuperJUNO 优化交付（SuperJUNO/）

上游流水线（commit `82f25ea`，见 `upstream_commit.txt` 与 `upstream/`）之上的
**可审查优化候选**：一个逐位等价的 C++17 随机流后端、一个复用上游自带
CUDA 核的 2×2 求逆替换、一份未接入的融合核设计（草稿，不再扩展），以及带
防覆盖、防活跃作业误归档与审计记录的 Slurm 提交流程。**所有候选项默认关闭**：
不改环境变量时运行的是上游原路径。

完整依据、公式与源位置、验证状态见 [IMPLEMENTATION.md](IMPLEMENTATION.md)；
当前进展快照、RTX4090 重验清单与正式验收清单见
[docs/CONTINUATION.md](docs/CONTINUATION.md)（2026-10-17 起）。
本地已验证与待集群事项的边界：本机无 CUDA/CuPy/HDF5 数据，`make test` 的
通过是**单元级 LOCAL_PASS**，不是官方 PE/tvE 比较，也不是加速比。

环境入口（2026-10-17 修正）：本题按官网使用**并行科技 RTX4090 集群**——经并行科技
云桌面 SSH 接入、Slurm sbatch 提交、单节点最多 2 GPU；**不创建独立 RTX4090 云实例**，
也不使用 SuperShip 的 RTX3090 实例。集群接入现状为 CLUSTER_PENDING（云桌面 SSH
connection closed、CLI `loginCluster` 拒绝、快传错误码 12213 待客服核查；平台可见绑定
sczf074 / BSCC-N16R4，是否为比赛指定 4090 资源待确认），资源现状与客服反馈材料见
[../doc/RESOURCE_HANDOFF.md](../doc/RESOURCE_HANDOFF.md) 与
[../doc/GPU_LOGIN_SUPPORT.md](../doc/GPU_LOGIN_SUPPORT.md)（用户计划 2026-10-18
自行发送；不代发、不重试登录、不读取密码/证书），详见「集群部署与提交」。

## 目录

```
SuperJUNO/                       # 本层：构建、测试、提交脚本（开发层）
├── Makefile              # 构建 + 本地验证入口（含 -pthread）
├── README.md             # 本文件
├── IMPLEMENTATION.md     # 实施与验证记录（先读这个）
├── upstream_commit.txt   # 上游 commit（82f25ea67f10ede8881f4dd4656ad78cc40cfd5f）
├── cpp/
│   ├── sjrng_core.cpp            # C++17: SeedSequence+PCG64+ziggurat 逐位实现
│   ├── sjrng_ziggurat_tables.hpp # 由 numpy 2.4.6 ziggurat_constants.h 生成
│   ├── sjrng.py                  # ctypes 封装（安装到 upstream/src/）
│   ├── test_sjrng.py             # 逐位/边界/线程/拒绝路径测试 + 微基准（LOCAL）
│   ├── test_install.py           # 安装接缝测试（LOCAL）
│   ├── vmove1_fused_ref.cpp      # gpu/vmove1_fused.cu 的主机镜像（公式对照）
│   └── test_fused_formula.py     # 融合公式 vs vmove1 转录（LOCAL，公式级）
├── gpu/
│   └── vmove1_fused.cu    # 未接入的融合核草稿（接入条件见文件头）
├── third_party/
│   └── NUMPY_LICENSE.txt  # 本地已装 NumPy 2.4.6 随机模块/顶层许可声明的逐字副本
├── slurm/
│   ├── run.slurm          # 官方三阶段 + --compare 的作业脚本（1 GPU，可提交时覆盖到 2）
│   ├── submit.sh          # 锁 + 活跃检查 + 输入检查 + 防覆盖归档 + 提交记录
│   ├── status.sh          # 汇总作业的 TOTAL / exit code / PASS-FAIL
│   └── test_submit_controlflow.sh  # 本地控制流回归（sbatch/squeue 显式桩）
└── upstream/              # 官方仓库 + 两处 env-gated 补丁（git diff 可审计）
                           # + 新增 src/sjrng.py、src/_sjrng_core.so（make install）
                           # —— 部署时这一层的内容拷到 ~/run/superjuno（见下）
```

## 本地构建与验证（无需 GPU/数据）

```bash
make test        # 构建 + 全部本地测试：
                 #   sjrng 逐位 18/18、融合公式 4/4、安装接缝 3/3、
                 #   slurm 控制流回归 52/52 断言（桩 sbatch/squeue）
make install     # 将 sjrng.py/_sjrng_core.so 安装到 upstream/src/（幂等）
```

依赖：g++（C++17）、python3 + numpy、git（测试需从 upstream/ 提取上游函数）。
编译禁用 fast-math 类选项（位级一致的前提）；编译与链接**显式 `-pthread`**
（std::thread 所需；新 glibc 不加也可能碰巧能跑，旧发行版不能）。

## NumPy 来源、许可与版本边界（重要）

- C++ 随机后端逐语句移植自 **NumPy 2.4.6 源码**（`bit_generator.pyx`、
  `pcg64.{h,c}`、`_pcg64.pyx`、`distributions.c`、`ziggurat_constants.h`，
  均在 `numpy/random/` 下）。移植后的语句与常量**已编入本仓库的
  `cpp/sjrng_core.cpp`、`cpp/sjrng_ziggurat_tables.hpp`**（即构成对 NumPy
  源码片断的再分发），随源码使用的许可声明已从本机安装的 NumPy 2.4.6
  **逐字复制**到 `third_party/NUMPY_LICENSE.txt`（含来源与版本记录）：
  `numpy/random/` 模块为 **NCSA / 3-Clause BSD 双许可**（Copyright (c) 2019
  Kevin Sheppard），NumPy 顶层为 BSD-3-Clause（Copyright (c) 2005-2025,
  NumPy Developers）。该文件只收录实际随所用源码的声明，未自行撰写或
  改写任何许可文本。
- **不承诺跨 NumPy 版本的位级一致**：PCG64 的底层整数流稳定，但
  Generator 之上的分布/浮点实现没有跨版本位流保证。本机对照的是
  numpy 2.4.6（g++ 16.2.1，2026-10-16）。**集群 numpy 版本未知，启用
  `SUPERJUNO_SJRNG` 前必须在目标环境重跑按位对照**（`cpp/test_sjrng.py`，
  它含与监督者 `../doc/verify_juno_supervisor.py` 相同的 git 对象 AST 提取对照；
  该独立脚本与报告 `../doc/supervisor_verification_SuperJUNO.json` 位于公共 doc/）。
- 本机构建的 `_sjrng_core.so`（GCC 16 / 新 glibc）**不得当集群可移植产物**：
  在集群计算节点（一次性 sbatch 作业，见「集群部署与提交」）重新 `make install`。

## opt-in 开关（默认全关 = 上游原路径）

| 环境变量 | 作用 | 风险级别 |
|---|---|---|
| `SUPERJUNO_SJRNG=1` | FSMP 波形随机流改用 C++ 后端（`prepare_random`） | 低（本机已逐位验证；目标环境需重跑对照 + `--compare`） |
| `SUPERJUNO_BETA_INV=1` | `vmove1` 用上游自带 `beta_inversion` 核替代 `cp.linalg.inv` | 中（非位级等价；必须先过 `--compare`） |
| `SJ_GPUS=1|2`（提交脚本用） | 每作业 GPU 数（规则上限 2） | — |

开关经 sbatch `--export=ALL` 传播，`submit.sh` 把取值记入
`output/logs/submissions.tsv`；`run.slurm` 在作业日志开头回显。失败路径：
sjrng 加载/执行异常 → 自动回落上游 numpy 循环并在 stderr 打印原因。

## 集群部署与提交（数据就位后）

**环境入口与登录节点边界**：官网要求本题使用**并行科技 RTX4090 集群**——经并行科技
云桌面 SSH 接入、Slurm sbatch 提交、单节点最多 2 GPU（`docs/official.txt`；
2026-10-17 复核快照 `../doc/records/cluster_route_20261017/`）。云桌面 SSH 进入的
登录节点只用于查询与文件管理（上传/整理文件、`sinfo`/`squeue`、提交与查看作业）：
**`make install` 重编、`cpp/test_sjrng.py` 位级对照与一切数值计算都在 sbatch 分配的
计算节点上执行**，不经 srun/salloc；提交命令（`submit.sh`/`status.sh`）本身只在
登录节点发起 sbatch 与查询。接入恢复前以下均为待执行步骤（CLUSTER_PENDING）。

**先分清两层**：本层 `SuperJUNO/` 是构建/测试层；`upstream/` 的内容（含补丁
与 `make install` 装入的 `src/sjrng.py`、`src/_sjrng_core.so`）才是运行层，
部署目标是 `~/run/superjuno`。

```bash
# 以下远端命令在集群上执行：上传/文件整理在登录节点（云桌面 SSH 会话）进行，
# 构建、位级对照与计算一律经 sbatch 在计算节点执行。分区名以主办方/客服答复为准
# （当前可见绑定 sczf074 / BSCC-N16R4 是否为比赛 4090 分区待确认）。

# 1) 经云桌面 SSH/快传（恢复后）上传本层（含 upstream/）到集群，例如 ~/build/SuperJUNO/
#    （上传与目录整理属登录节点允许的文件管理）

# 2) 构建 + 安装 + NumPy 位级对照：一次性 sbatch 作业在计算节点执行
#    （启用 SUPERJUNO_SJRNG 的硬前置；.so 不可本机预编译直投；
#     不在登录节点编译或跑数值对照；分区/资源参数按集群实际调整）
sbatch --partition=$GPU_PARTITION --wrap '
  cd ~/build/SuperJUNO &&
  module load miniforge3/24.11 &&
  CONDA_BASE=$(conda info --base) &&
  source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate juno &&
  make install && python3 cpp/test_sjrng.py'
#    作业 Slurm 输出中 18 项全过才可在集群启用 SJRNG；不通过则该开关在集群禁用。
#    如目标环境没有 make，可在同一作业里改用显式命令:
#      g++ -std=c++17 -O2 -fPIC -pthread -fno-fast-math \
#          -fno-unsafe-math-optimizations -shared -pthread \
#          -o upstream/src/_sjrng_core.so cpp/sjrng_core.cpp && \
#      cp cpp/sjrng.py upstream/src/sjrng.py

# 3) 部署运行层：upstream/ 内容 → ~/run/superjuno；三个脚本放运行根（文件管理，
#    登录节点允许；在云桌面 SSH 会话中执行）
#    rsync -a ~/build/SuperJUNO/upstream/ ~/run/superjuno/ &&
#    cp ~/build/SuperJUNO/slurm/run.slurm ~/build/SuperJUNO/slurm/submit.sh \
#       ~/build/SuperJUNO/slurm/status.sh ~/run/superjuno/
#    data/ 按官方说明放置（data/raw/<id>.h5、ana_PE/、ana_tvE/ 等）

# 4) 先跑示例点验证（不计分；submit.sh 只在登录节点发起 sbatch，计算在计算节点）
SJ_PARTITION=$GPU_PARTITION ./slurm/submit.sh one
./slurm/status.sh        # 等 Final exit code: 0 且两项比较 PASS

# 5) 验证 opt-in 候选（每个开关单独在 one 上过 --compare，见 IMPLEMENTATION.md §8）
SUPERJUNO_SJRNG=1 SJ_PARTITION=$GPU_PARTITION ./slurm/submit.sh one

# 6) 正式点（等权计分）
SJ_PARTITION=$GPU_PARTITION ./slurm/submit.sh 0 1 2
```

`submit.sh` 对每个 id 依次执行：参数/重复 id/资源检查（在任何输出路径创建
之前）→ 取该 id 的锁（`output/locks/<id>.lock`）→ **活跃作业检查**：一次
`squeue -h -u "$USER" -o %A` 列出当前用户全部在队作业，再与 `submissions.tsv`
中该 id 的历史 job id 逐一求交（`sbatch --parsable` 记录的 `;cluster` 后缀
比较前剥离）；有交集则拒绝（不归档、不重复提交），历史 id 已被控制器清理
（不在队列）则允许归档重跑，squeue 查询本身失败也拒绝（宁拒不猜）
→ 官方输入文件齐全检查（缺 `data/raw/<id>.h5` 等直接拒绝；
`SJ_SKIP_DATA_CHECK=1` 仅限本地桩测试）→ 归档旧产物到
`output/archive/<id>/<UTC>[-N]/`（同秒冲突加后缀，移动不删除）→ sbatch →
记录 `submissions.tsv`。**单个 id 失败不影响同批其他 id**。
`run.slurm` 自身在参数非法（exit 99）、非 Slurm/多节点/GPU>2（97）、产物已
存在（98）时拒绝，且这些检查都在任何输出路径创建之前完成。

提交产物（time.csv、三阶段产物与 .log、.compare.txt、`superjuno-<JOBID>.out/.err`）
保持同次运行原地不动；下次提交同一 id 时先经活跃检查再归档，不会覆盖。

## 不要做

- 不要为本题创建独立 RTX4090 云实例，也不要把 SuperShip 的 RTX3090 实例当本题环境。
- 不要在登录节点运行计算、编译构建、数值测试或 agent；登录节点仅查询/文件管理，
  构建与位级对照经一次性 sbatch 作业、正式计算远端只 `sbatch`（已由脚本保证）。
- 不要把本地微基准数字当官方加速比；报告只引用作业 `TOTAL` 与 time.csv。
- 不要在未过 `--compare` 前给 `SUPERJUNO_BETA_INV`（或融合核）当默认值。
- 不要把本机构建的 `.so` 直接投放到集群；先在集群计算节点（sbatch 作业）重建
  并重跑按位对照。
- 不要把本地控制流回归（桩 sbatch/squeue）写成任何真实集群提交记录。
