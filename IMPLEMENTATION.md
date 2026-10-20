# SuperJUNO 实施记录（IMPLEMENTATION）

本文记录 SuperJUNO 题目在本机的实施内容、依据与验证状态。结论先行：

- **已本地验证（LOCAL_PASS，2026-10-16）**：C++17 FSMP 随机流后端与 NumPy 参考实现**逐位一致**——`cpp/test_sjrng.py` 18/18（含 AST 提取自不可变 git 对象的上游真函数对照、32/64 位 field、切片/反转 requester、TRIALS=0/1/4000、空 requester、大 id、80 流慢路径、拒绝路径、线程不变性、ziggurat 尾部分支）；监督者独立脚本 `../doc/verify_juno_supervisor.py` 复检 **23 项全过**（16 组随机位级对照 + 7 项监督者自有 sbatch/squeue 桩下的提交控制流检查，含 squeue 状态未知时拒绝提交）；融合 Δν 候选公式与上游 vmove1 数值一致（相对误差 < 1e-12）；上游补丁默认分支与原实现逐行等价；自有 slurm 控制流回归 52/52 断言（显式 sbatch/squeue 桩，无任何真实提交）。
- **已在集群验证（CLUSTER_PASS，2026-10-19，BSCC-N16R4 gpu 分区/RTX 4090）**：环境、数据、运行层全部就位（见 §12）；`SUPERJUNO_SJRNG` 在集群计算节点重编 + 集群 numpy 2.4.6 位级对照 **18/18 OK**，`one --compare` 基线与 SJRNG 各自 **PE/tvE 双 PASS、Final exit 0**。诚实结论：`one`（单事例小测例）上 SJRNG **无可测端到端收益**（热缓存基线 305.5s vs SJRNG 312.1s，跨节点噪声内）；正式点 0/1/2 未运行、无任何加速比宣称。
- **待集群/数据验证（CLUSTER_PENDING）**：正式点 0/1/2 的提交与计时；`SUPERJUNO_BETA_INV`、融合核对官方比较的通过与真实端到端加速比；任何 GPU 数值改动未经 `--compare` 通过不得启用。
- **不宣称**：本文件不包含任何正式测点通过、0/1/2 集群计时或加速比。

状态词汇沿用工作约定：`LOCAL_PASS` / `CLUSTER_PENDING` / `DATA_PENDING` / `MODEL_PENDING`。

## 1. 上游来源与不可变契约

- 上游仓库：https://github.com/thu-scc/SuperJUNO ，由监督者克隆至 `upstream/`。
- commit：`82f25ea67f10ede8881f4dd4656ad78cc40cfd5f`（"preliminary"，完整哈希见 `upstream_commit.txt`，可用 `git -C upstream rev-parse HEAD` 复核）。
- 未改动：`scripts/run.py`（计时与比较调度）、`test/`（比较器与容差）、`docs/`、输入与参考结果路径约定、三阶段命令行参数（见 `scripts/run.py:40-49`）。

### 1.1 三阶段与不可变参数（来源：docs/official.txt、recon.tex §三/§四.6、scripts/run.py）

| 阶段 | 入口 | 固定参数 | 产物 |
|---|---|---|---|
| 1 先验生成 | `src/fsmp_prior.py` | LucyDDM 2,000 次迭代、候选阈值 0.1、seed 0、波形前 1,000 采样点 | `output/sparsify/<id>.h5` |
| 2 PE 采样 | `src/fsmp.py` | 4,000 MCMC steps、`--full-step 50`、舍弃前半链、保存 step 2000,2050,…,3950 共 40 组、seed 0 | `output/PE/<id>.pq` |
| 3 事例重建 | `src/main.py` | 2,000 MCMC steps、20 EM steps、`--bappen 2`（1 或 2 活动顶点）、10 个 event blocks、λ_high=50、seed 0 | `output/tvE/<id>.h5`（`/Recon` 形状 (2, N_events, 2000)） |

### 1.2 随机量映射（不得改变；来源：recon.tex §四.4"确定性随机映射"+ 代码）

| 用途 | 规范 |
|---|---|
| LucyDDM 初始增益 | `np.random.default_rng(seed)`，按筛选后波形行序连续生成（`fsmp_prior.py:82,118-122`） |
| FSMP proposal 类型 | `PCG64(SeedSequence([seed,0]))`，按 step 广播（`prepare.py:153-162` FLIP_STREAM=0） |
| FSMP 波形随机量 | 每 `(EventID,channel)` 独立流 `PCG64(SeedSequence([seed,1,eid,ch]))`，依次生成 istar/erandom（random）、wanders（standard_normal, float64）、accepts（log(random)）（`prepare.py:165-207`） |
| 重建 dimension | stream 0，按 step 的 float32 均匀量（`recon_random.py:125-127`） |
| FSMP realization | 每 EventID 独立 stream 1（`recon_random.py:129-135`） |
| 逐通道接受阈值 | stream 2 无状态 SplitMix64（`recon_random.py:65-83`） |
| 事例/顶点 proposal | 每 EventID 的 stream 3–6（对数接受、正态 proposal、顶点选择、birth 变换）（`recon_random.py:137-169`） |

### 1.3 官方比较接口（未改动）

- 入口：`python scripts/run.py <id> --compare`（流水线成功后依次比较 PE、tvE）。
- 单独比较：`python test/compare_PE.py data/ana_PE/<id>.pq output/PE/<id>.pq`、`python test/compare_tvE.py data/ana_tvE/<id>.h5 output/tvE/<id>.h5`。
- 判据：`|x−r| ≤ ATOL + RTOL·|r|`，`ATOL=RTOL=1e-6`（`test/comparison.py`）；离散字段精确匹配；PE 按 `(eid,ch,step)` 对齐，tvE 按 `(EventID,step)` 对齐（`test/README.md`）。
- 计分计时：作业 stdout 的 `TOTAL:` 与 `Timing:` 指向 CSV 的 TOTAL 行 `real_s` 核对；排队不计入。

## 2. 热点选择依据

本机无数据、无 GPU，无法实测剖面；以下热点依据代码结构与调用频次推理，**均为待集群实测验证的假设**：

1. **`vmove1` 的批量 2×2 求逆（GPU）**——`src/analysis/fsmp/move.py:92` 对 `(l_e,2,2)` float64 调 `cp.linalg.inv`（cuSOLVER 批处理 LU），每个 MCMC step 一次 × 4,000 steps × 每部分。上游自己在 `seladd.cu:99-112` 写了显式伴随矩阵求逆核 `beta_inversion_kern`，在 `seladd.py:93-104` 封装、`move.py:2` 导入，**但从未调用**（`grep` 全仓库仅 import）。`prepare.py:143` 还出现 `set_batched_gesv_limit`，说明作者已知批处理 gesv 开销。这是任务说明中"上游已有算子对应融合/复用"的直接命中。
2. **`prepare_random` 的逐波形 Python 循环（CPU）**——`prepare.py:194-207` 每条波形 4 次 numpy RNG 调用，且 `istar[:, i] = …` 是对 `(TRIALS, l_e)` 大数组的跨步列写（列间隔 l_e×8 字节，每次写都是 cache miss）。量级：每部分 `4 × TRIALS × l_e` 个 float64（TRIALS=4000，l_e 数千）。
3. **vmove1 尾段的多核临时量（GPU，设计件）**——`move.py:88-96` 中 `beta_inv` 组装/对称化、`inv`、三次批量小矩阵乘、两次 `stable_det`、`log` 共约 8 次启动与多个 `(l_e,2,2)` 临时量，每 step 重复。见 §5。

## 3. 候选一：`sjrng` C++17 FSMP 随机流后端（已接入，显式 opt-in）

**上游位置**：`src/analysis/fsmp/prepare.py:165-207` `prepare_random`（唯一改动点，见 §7）。

**做法**：C++17 逐位复现 NumPy `Generator(PCG64(SeedSequence([seed,1,eid,ch])))` 的四次抽取（`random`/`random`/`standard_normal`(ziggurat)/`random`）：

- `cpp/sjrng_core.cpp`：SeedSequence 哈希（`bit_generator.pyx` 的 hashmix/mix/generate_state，纯 32 位整数运算）、PCG64（XSL-RR 128/64，128 位 LCG）、`uint64_to_double`（`(u>>11)·2^-53`）、ziggurat 标准正态（`distributions.c::random_standard_normal` 逐句移植 + `ziggurat_constants.h` 常量表原样嵌入 `cpp/sjrng_ziggurat_tables.hpp`，字面值逐字拷贝故解析出的 double 位级相同）。移植源：**NumPy 2.4.6**（移植文件全部位于 `numpy/random/`，该模块为 NCSA/3-Clause BSD 双许可；本机已装 NumPy 2.4.6 的逐字许可声明复制于 `third_party/NUMPY_LICENSE.txt`，见 §11）。
- 第四次数组返回**原始均匀量**，由 Python 侧 `np.log` 完成对数——与上游同库同 SIMD 实现，libm 版本差异不可能进入结果。
- Python 封装（`cpp/sjrng.py`）对 requester 做严格校验：field 必须是整数 dtype（拒绝 float 等会静默截断的类型）、uint64 超 INT64_MAX 拒绝（拒绝回绕）、经 `np.ascontiguousarray(..., dtype=np.int64)` 消除结构化 field 的 stride（监督者在 `supervisor_draft_findings.json` 复现的正是 stride 直传 C++ 的错位 bug），seed/trials/stream 必须是真整数（拒绝 `4000.0`/`True`）。
- C ABI 安全（`sjrng_core.cpp`）：`n_streams*trials` 的溢出检查**先做除法比较再相乘**（上限取 `SIZE_MAX/4/sizeof(double)`，同时覆盖 4 倍 scratch）；任何 C++ 异常不越过 ctypes 边界（分配失败返回 6、worker 内部错误记入 atomic 后返回 7、绝不 `std::terminate`）；线程创建失败时**先 join 已启动线程**再返回 8。错误码 0–8 语义在源文件头注释中列表。
- 性能路径：按流连续填充 + 64×64 分块转置到 `(TRIALS, l_e)` 输出布局，消除跨步列写；线程只切分相互独立的流，结果与线程数无关（已测）。
- 数值规则：编译禁用 `-ffast-math`/`-Ofast`/`-march=native`（见 Makefile）；编译与链接显式 `-pthread`；`log1p/exp` 与 numpy `npy_log1p/npy-exp` 在 Linux 上解析到同一 glibc 符号，同一机器上位级一致。

**逐位验证（LOCAL_PASS，2026-10-16）**：`cpp/test_sjrng.py` 18/18 通过。参照系有两套：① 上游 `prepare_random` numpy 体的逐行转录；② **从不可变 git 对象 `82f25ea:src/analysis/fsmp/prepare.py` 经 `git show`+AST 提取的真上游函数**（与监督者 `verify_juno_supervisor.py` 同法，不信任手抄、不需要 CuPy）。覆盖：seed∈{0,1,42,123456789,2^31−1,2^32}、trials∈{0,1,7,64,250,4000}、int32/int64 field、`[::2]` 切片与反转视图、`part_index[["eid","ch"]]` 形状的 uint32/uint16 双 field 非连续视图、空 requester、边界 id（0、2^31−1、2^32−1、2^32、2^53+3、2^63−1，多字熵编码）、80 条随机大 id 流×4000 步（统计确认 ziggurat 尾部分支命中）、线程数 1/2/4/8/默认、全部 `tobytes()` 位级相等；另有 5 组拒绝路径测试（float field、uint64 溢出、负 id、非整数标量）确认在 Python 层抛错、绝不静默截断。接缝测试 `cpp/test_install.py` 3/3：从 `upstream/src/` 按管线方式导入、`fsmp.py:129` 形状的 `part_index[["eid","ch"]]` 视图输入逐位一致、补丁默认分支完整。监督者独立脚本 `../doc/verify_juno_supervisor.py` 复检 23 项 LOCAL_PASS（16 组随机位级 + 7 组提交控制流桩，numpy 2.4.6，报告 `../doc/supervisor_verification_SuperJUNO.json`）。

**本地微基准（仅参考，非官方加速比）**：streams=5000、trials=4000 时 numpy 循环 0.557 s vs C++ 0.287 s（1.9×，受内存带宽限制；本机 2026-10-16 单次运行）。该微基准**不是**流水线耗时，流水线收益取决于每测例波形数与分块数，待集群计时。

**版本边界（重要，纠正自 ZCODE_REVIEW_1）**：本实现是**对 NumPy 2.4.6 源码的一次性移植**，不是跨版本契约——NumPy 只承诺 PCG64 底层整数流的稳定性，Generator 之上的分布/浮点实现没有跨版本位流保证（官方文档核对记录见 `docs/numpy_compatibility_notes.md`）。集群 numpy 版本未知：启用 `SUPERJUNO_SJRNG` 前必须先用目标环境重跑 `cpp/test_sjrng.py`（或监督者的 `../doc/verify_juno_supervisor.py`）通过按位对照；不通过则该开关在集群禁用。CPU 内存额外峰值 `4×TRIALS×l_e×8` 字节临时转置缓冲（5000 流 × 4000 步 ≈ 640 MB）。本机 GCC 16.2.1 构建的 `.so` 不可直接投放集群，须在目标节点重建（README 部署步骤第 2 步）。

## 4. 候选二：`beta_inversion` 显式 2×2 求逆核（已接入，显式 opt-in）

**上游位置**：替换 `move.py:92`（`cp.linalg.inv(beta_inv)`）为上游自带但未调用的 `beta_inversion`（`seladd.cu:99-112` 定义、`seladd.py:93-104` 封装、`move.py:2` 已导入）。公式：β = adj(B)/det(B)，与上游 `beta_inversion_kern` 完全同一实现，本任务未写新核。

**风险（为何只 opt-in）**：与 cuSOLVER LU 的舍入路径不同，**不保证逐位一致**；recon.tex §四.4.3 明示 ulp 级差异可能翻转临界接受并使链分离。因此默认关闭，启用前必须先过 `--compare`（见 §8 流程）。传入 `vmove2` 的 `beta` 契约不变。

## 5. 候选三：vmove1 尾段融合核（未接入，草稿，不再扩展）

`gpu/vmove1_fused.cu` 将 `move.py:88-96`（beta_inv 组装+对称化、求逆、Y 与二次型、两次 stable_det、log）融合为每 step 一个核，消除约 8 次启动与若干 `(l_e,2,2)` 临时量。j-归约顺序为固定升序，**不可能与 cuBLAS 逐位一致**，故按约定不接入；`cpp/vmove1_fused_ref.cpp` 是其逐语句主机镜像，`cpp/test_fused_formula.py`（4/4 通过）对照 vmove1 的 NumPy 转录验证公式（rtol 1e-12，含近奇异与最小宽度边界）。**本审查修复轮不再扩展该草稿**；接入前必须先在集群以 `--compare` 验证。

## 6. sbatch 与提交流程

- `slurm/run.slurm`：官方示例结构（单节点、1 GPU 默认、`%x-%j.out/.err`），执行 `python scripts/run.py <id> --compare`。**验证先于任何输出路径**：恰好一个参数且 ∈ {one,0,1,2}（exit 99）、必须在 Slurm 下运行（`SLURM_JOB_ID` 存在，exit 97）、单节点（exit 97）、`SLURM_GPUS_ON_NODE` ≤ 2（exit 97）、`scripts/run.py` 存在（97）；产物已存在且未设 `SJ_OVERWRITE=1` 时拒绝（exit 98）。之后才创建目录、加载环境、打印 Python/CuPy/numpy/GPU 版本与 opt-in 开关值（不进入计时）。GPU 数在提交时覆盖：`sbatch --gres=gpu:2 …` 或 `SJ_GPUS=2 ./slurm/submit.sh …`（≤2）。
- `slurm/submit.sh`：每个 id 按固定顺序处理——
  1. **参数/资源检查在任何输出路径创建之前**：id ∈ {one,0,1,2}、同批无重复 id（`one one` 直接拒绝）、`SJ_GPUS` ∈ {1,2}、sbatch/squeue/run.slurm/`scripts/run.py` 存在——`../../x` 之类的参数拼不进任何路径；
  2. 取该 id 的锁 `output/locks/<id>.lock`（mkdir 原子操作；持锁进程存活则拒绝该 id，已死则回收陈锁）；
  3. **活跃作业检查先于归档**：一次 `squeue -h -u "$USER" -o %A` 列出当前用户全部排队/运行作业，再与该 id 在 `submissions.tsv` 记录过的全部 job id 逐一求交（`sbatch --parsable` 记录的 `;cluster` 后缀比较前剥离）；有交集（或 squeue 查询本身失败）即拒绝——不会归档仍在写的作业、不会同 id 双提交。**不**把历史 job id 直接传给 `squeue -j`：Slurm 控制器清理已结束作业后该查询会返回 "Invalid job id"，会使该 case 永久无法归档重跑；已不在队列的历史 id 允许归档；
  4. 官方输入检查：`data/raw/<id>.h5`、共享五个输入文件（`ser_truth`/`sim`/`PMT`/`junoid`/`PMTPos`，即 `run.py` 预检的全部六类计算输入）、`data/ana_PE/<id>.pq`、`data/ana_tvE/<id>.h5` 齐全才提交（`SJ_SKIP_DATA_CHECK=1` 仅限本地桩测试）；
  5. 归档旧产物到 `output/archive/<id>/<UTC>[-N]/`（同秒冲突自动加后缀；移动不删除；归档中途失败可重试，已移动文件保留）；
  6. 预创建 `output/logs` 及 tmp 目录（Slurm 在脚本运行前打开 stdout 文件）、sbatch、把（时间、id、job id、分区、GPU 数、opt-in 开关）追加进 `output/logs/submissions.tsv`。
  **单个 id 失败只跳过该 id，不影响同批其他 id**；任一失败整体退出非零。提交 0/1/2 时若从未提交过 `one` 则拒绝（`SJ_SKIP_ONE_GUARD=1` 跳过）。状态边界（有意保持简单）：锁是同机 pid 级互斥；跨节点/集群态由 squeue 检查兜底；`submissions.tsv` 丢失时活跃检查退化为 `run.slurm` 的产物存在性拒绝（残余风险，已记录）。
- `slurm/test_submit_controlflow.sh`：本地控制流回归（sbatch/squeue 为显式桩，**无任何真实集群交互**），52/52 断言覆盖 ZCODE_REVIEW_2 全部场景（干净首提成功、活跃拒绝不归档、完成归档、重复 id、非法 id/`SJ_GPUS=3` 在任何输出路径之前拒绝、缺数据拒绝与 `SJ_SKIP_DATA_CHECK`、单 id 失败不阻塞他 id、同秒归档目录不冲突、one-guard、run.slurm 的 97/98/99 全部拒绝路径与"拒绝时不创建 output/"），及本轮新增（ZCODE_QUEUE_REVIEW 第 1 项）：**历史 job 已被控制器清理（不在用户队列清单、同用户另有活跃作业）→ 允许归档重跑**、**`;cluster` 后缀记录下活跃作业仍拒绝、离队后可重跑**、**squeue 查询失败保守拒绝**、**全程未用 `squeue -j`** 的全局断言（桩对 `-j` 直接报错，防止回退到按历史 id 查询）。
- `slurm/status.sh`：汇总各 job 的 `TOTAL`、各 exit code、PASS/FAIL。
- 完整保留同次运行的四类提交物：time.csv、三阶段产物、.log、.compare.txt、`superjuno-<JOBID>.out/.err`——直到下次提交同 id（且无活跃作业）时归档。

## 7. 上游改动清单（全部 env-gated，默认路径与上游逐行等价）

`git -C upstream diff` 输出（2 文件，+36/−1；新增未跟踪文件 2 个）：

1. `src/analysis/fsmp/move.py`：`vmove1` 中 `beta = cp.linalg.inv(beta_inv)` 改为 env 分支（`SUPERJUNO_BETA_INV=1` → `beta_inversion(beta, beta_inv)`；否则原句）。默认分支保留原代码原文。
2. `src/analysis/fsmp/prepare.py`：`prepare_random` 入口加 env 分支（`SUPERJUNO_SJRNG=1` → `import sjrng` 调用；任何异常打印 stderr 后**回落**到原 numpy 循环，循环体原文未动）。
3. 新增 `upstream/src/sjrng.py` + `upstream/src/_sjrng_core.so`（`make install` 安装；纯添加，不影响任何导入路径与默认行为）。

## 8. 未来基线/候选的官方三阶段比较流程

在集群（数据就位、`data/` 齐全）。环境入口：官网要求本题使用**并行科技 RTX4090 集群**——经并行科技云桌面 SSH 接入、Slurm sbatch 提交、单节点最多 2 GPU；**不创建独立 RTX4090 云实例**，接入现状与客服反馈材料见 `../doc/RESOURCE_HANDOFF.md`、`../doc/GPU_LOGIN_SUPPORT.md`。部署分层说明：**本层 `SuperJUNO/` 是构建/测试层；`upstream/` 的内容（含补丁与 `make install` 装入的 `src/sjrng.py`、`src/_sjrng_core.so`）部署到 `~/run/superjuno`**；完整命令见 README「集群部署与提交」。关键前置：经一次性 sbatch 作业在**计算节点**重新 `make install`（`.so` 不可本机预编译直投，不在登录节点编译），并用集群 python3 同样在计算节点重跑 `cpp/test_sjrng.py` 按位对照（不在登录节点跑数值对照；登录节点仅查询/文件管理）。

```bash
cd ~/run/superjuno            # upstream/ 内容 + slurm/* 三脚本已在根
mkdir -p output/logs          # submit.sh 也会创建

# (0) 基线：默认路径（两个开关都关）
SJ_PARTITION=$P ./slurm/submit.sh one && ./slurm/status.sh
#   要求 .out 中: Pipeline exit code 0、两项 Comparison exit code 0、Final exit code 0
#   记录 TOTAL 与 Timing: CSV 路径为基线 T_base

# (1) 逐开关验证（先 one；任一比较失败立即弃用该开关并归档证据）
SUPERJUNO_SJRNG=1     SJ_PARTITION=$P ./slurm/submit.sh one
SUPERJUNO_BETA_INV=1  SJ_PARTITION=$P ./slurm/submit.sh one
#   预期: SUPERJUNO_SJRNG 应逐位复现基线（PE/tvE 输出与基线运行一致）；
#         SUPERJUNO_BETA_INV 为数值等价路径，以 --compare 判据为准。

# (2) 组合与正式点（全部通过 one 后）
SUPERJUNO_SJRNG=1 SUPERJUNO_BETA_INV=1 SJ_PARTITION=$P ./slurm/submit.sh 0 1 2

# (3) 任意历史产物的独立复核（不重跑）：
python test/compare_PE.py  data/ana_PE/one.pq  output/PE/one.pq
python test/compare_tvE.py data/ana_tvE/one.h5 output/tvE/one.h5
```

计时以作业 stdout `TOTAL` 为准并与 CSV TOTAL 行核对（官方规则）；上例中环境变量经 sbatch `--export=ALL` 传播并记入 `submissions.tsv` 审计。

## 9. 验证状态总表

| 项 | 状态 | 证据 |
|---|---|---|
| sjrng 与 numpy 逐位一致（含边界/线程/ziggurat 尾部/切片/乱序/大 id/TRIALS=0） | LOCAL_PASS | `cpp/test_sjrng.py` 18/18（numpy 2.4.6, g++ 16.2.1, 2026-10-16） |
| 监督者独立复检（git 对象 AST 提取 reference 16 组随机 + 监督者桩下 7 组提交控制流） | LOCAL_PASS | `../doc/verify_juno_supervisor.py` 23/23，报告 `../doc/supervisor_verification_SuperJUNO.json` |
| requester 校验拒绝路径（float field/uint64 溢出/负 id/非整数标量） | LOCAL_PASS | `test_sjrng.py` 5 组拒绝测试 |
| 安装接缝（upstream/src 导入、`["eid","ch"]` 视图） | LOCAL_PASS | `cpp/test_install.py` 3/3 |
| 融合 Δν 公式 vs vmove1 转录 | LOCAL_PASS（公式级，非位级） | `cpp/test_fused_formula.py` 4/4 |
| 补丁默认分支等价 | LOCAL_PASS（源级审查 + AST/文本断言） | `git -C upstream diff`、`test_install.py` |
| 提交脚本控制流（桩 sbatch/squeue，无真实提交） | LOCAL_PASS | `slurm/test_submit_controlflow.sh` 52/52 断言 |
| NumPy 2.4.6 许可声明逐字落盘 | LOCAL_PASS | `third_party/NUMPY_LICENSE.txt` 与本机安装文件 `diff` 逐字节一致，2026-10-16 |
| sjrng 微基准 1.9× | LOCAL_PASS（微基准，非官方加速比） | 2026-10-16 单次运行输出，仅作参考 |
| SUPERJUNO_SJRNG 在集群 numpy 上的按位对照 | **CLUSTER_PASS**（2026-10-19） | 作业 1331820（g0024）：集群重编 `.so` 后 `test_sjrng.py` 18/18 OK，集群 numpy 2.4.6 |
| SUPERJUNO_SJRNG 流水线正确性（PE/tvE PASS） | **CLUSTER_PASS（one）**（2026-10-19） | 作业 1331852（g0030）：后端激活、PE/tvE 双 PASS、Final 0；热缓存基线 作业 1331839/1331864 亦双 PASS |
| SJRNG 在 one 上的端到端收益 | **无收益（如实记录）** | 热基线 305.5s（g0053）vs SJRNG 312.1s（g0030），节点噪声内；one 仅 1 事例，RNG 循环占比可忽略；收益须在 0/1/2 重测 |
| SUPERJUNO_BETA_INV 正确性与端到端加速比 | CLUSTER_PENDING | 需 `--compare`；先 one 后 0/1/2 |
| 融合核接入 | 未接入（草稿，不再扩展） | `gpu/vmove1_fused.cu` 注释含接入条件 |
| 官方流水线在本机运行 | DATA_PENDING（无 CUDA/CuPy/HDF5 数据） | 本机环境声明；集群侧数据已于 2026-10-19 就位 |
| 真实 sbatch 提交/GPU 流程 | **CLUSTER_PASS**（2026-10-19） | 作业 1331811/1331820/1331825/1331839/1331852/1331864；submit.sh 归档/活跃检查/submissions.tsv 审计实际工作正常 |
| 集群 numpy/CuPy/glibc 与本机差异核查 | **CLUSTER_PASS**（2026-10-19） | 作业 .out 打印：python 3.11.17、cupy 14.2.0 CUDA 12040、numpy 2.4.6、RTX 4090 |
| 正式点 0/1/2 提交与计时 | CLUSTER_PENDING | 未运行；须按 §8 流程 |

## 10. 审查修复记录（本轮，2026-10-16）

按监督者 `ZCODE_REVIEW_1.md`（含文件末尾构建补充）与 `ZCODE_REVIEW_2.md` 逐项修复；两文件的复现证据为 `supervisor_draft_findings.json`（4 组结构化 field 位错配，已修）与 `supervisor_submission_findings.json`（干净首提 exit 1，已修）。监督者随后扩展了独立验证脚本 `verify_juno_supervisor.py`（现位于 `../doc/`；新增对 `submit.sh` 的 7 组自有桩检查，含 squeue 状态未知必须拒绝），本轮终态 23/23 LOCAL_PASS。

| 审查项 | 修复 | 证据 |
|---|---|---|
| R1-1 结构化 field stride 错位、静默截断 | `sjrng.py`：在已有 `ascontiguousarray` 连续化（监督者 16 组对照所验证的版本）之上，补整数 dtype 校验、uint64>INT64MAX 拒绝、严格整数标量，杜绝一切静默截断路径 | `test_sjrng.py` 新增 5 组拒绝测试 + 32/64bit/切片/乱序对照全过；`verify_juno_supervisor.py` 16/16 |
| R1-2 `n_streams*trials` 先乘后查（signed overflow UB） | 改为先除法比较（上限 `SIZE_MAX/4/sizeof(double)`）再乘；C ABI 全路径无异常越界（bad_alloc→6、worker 异常→atomic 记录后返回 7、线程创建失败先 join 已启动线程再返回 8） | `sjrng_core.cpp` C ABI 段 + 源文件头错误码表 |
| R1-3 跨版本位流保证声明不成立 | 源码注释/README/本文件改写为"NumPy 2.4.6 源码（BSD-3-Clause）一次性移植、仅对旁置 numpy 验证、集群须重跑对照"；删除一切"跨版本保证"表述 | `sjrng_core.cpp` 头注释、`sjrng.py` docstring、README「NumPy 来源、许可与版本边界」 |
| R1-4 测试补强 | AST 提取不可变 git 对象中的真上游 `prepare_random`（免 CuPy、不信任手抄）+ TRIALS=0/1/4000、空 requester、多种大 id、乱序、1/多线程对照、80 流大样本慢路径；全部 float64 位级比较 | `test_sjrng.py` 18/18 |
| R1-5 README/IMPLEMENTATION 边界声明 | 明确"本地按位对照 ≠ 官方 HDF5/Parquet PE/tvE 比较；三阶段/4090 性能待数据与集群"；候选项默认关闭、可回退、sbatch 走官方入口 | 本文 §1.3、§9 与 README |
| R1 构建补充 `-pthread` | Makefile 编译与链接均显式 `-pthread`；README 部署要求目标节点重建 `.so` | `Makefile`、README 部署步骤 2 |
| R2-1 `archive_id` 尾句 `[ ]&&echo` 在 moved=0 时 rc=1 致 set -e 提前退出 | 函数改为显式 `return 0`；干净首提用桩复测 exit 0 | 控制流回归断言「clean first submission exits 0」 |
| R2-2 同 id 活跃作业防归档/防双提交、重复 id、锁、归档重名 | 参数检查前置（重复 id 拒绝）；per-id mkdir 锁（陈锁回收）；squeue 活跃检查先于归档（squeue 失败宁拒不猜）；归档目录同秒加 `-N` 后缀；单 id 失败不阻塞他 id；输入数据齐全检查 | 控制流回归 40/40 断言 |
| R2-3 run.slurm 参数在任何输出路径前验证、环境检查 | 恰好 1 个参数且 ∈ {one,0,1,2}（99）；非 Slurm/多节点/GPU>2（97）；产物已存在（98）；全部先于 mkdir/路径拼接；仍只调用官方 `run.py --compare`，计时/比较逻辑未动 | 控制流回归 97/98/99 全覆盖且断言"拒绝时不创建 output/" |
| R2-4 本地 Shell 控制流回归 | `slurm/test_submit_controlflow.sh`（显式 sbatch/squeue 桩；未写成任何真实 Slurm/GPU 通过） | 40/40，2026-10-16 |

## 11. 收尾轮记录（2026-10-16，ZCODE_QUEUE_REVIEW 两项）

只处理监督者 `ZCODE_QUEUE_REVIEW.md` 的两项；其他代码未改，未新增优化，未重复微基准。

**第 1 项：`slurm/submit.sh::active_check` 不再按历史 job id 查询。**
旧实现把该 id 在 `submissions.tsv` 记录过的全部 job id 传给 `squeue -h -j`；Slurm 控制器清理已结束作业后该查询返回 "Invalid job id"（非零退出），走保守拒绝分支，导致该 case 永久无法归档重跑。新实现：一次 `squeue -h -u "$USER" -o %A` 列出当前用户全部在队作业，与该 id 的历史 job id 逐一求交（`awk` 读历史 → `grep -qxF` 逐个匹配；`sbatch --parsable` 的 `;cluster` 后缀在比较前用 `${job%%;*}` 剥离）；有交集拒绝，历史 id 不在队列则允许归档重跑，squeue 查询本身失败仍保守拒绝。同步更新 `submit.sh` 头部 "State boundaries" 注释与 README 提交流程描述。

对应测试更新（`slurm/test_submit_controlflow.sh`）：squeue 桩改为支持 `-h -u <user> -o %A`（从队列文件输出在队 job id），`-j` 显式报错退出（防回退）；sbatch 桩支持 `STUB_SBATCH_SUFFIX` 模拟 `--parsable` 的 `;cluster` 后缀，`STUB_SQUEUE_FAIL=1` 模拟查询失败。新增 4 组场景 12 条断言：① 历史 job 已被清理（不在用户队列清单）且同用户另有无关活跃作业 → 允许归档并重跑（恰好一次 sbatch，旧产物完整归档）；② 记录为 `<job>;<cluster>` 时活跃作业仍拒绝、离队后可重跑；③ squeue 查询失败 → 保守拒绝（不归档、不 sbatch）；④ 全程日志无 `squeue -j`、确有 `-u … -o %A` 形式的全局断言。

**第 2 项：NumPy 许可声明落盘。** 逐句移植即构成对 NumPy 源码片断的再分发，README 原"未再分发源码"表述不准确。已将本机安装的 NumPy 2.4.6（`/home/amazcuter/.local/lib/python3.14/site-packages`，dist-info `numpy-2.4.6.dist-info`）随源码携带的两份声明**逐字复制**到 `third_party/NUMPY_LICENSE.txt`（已用 `diff` 校验逐字节一致）：`numpy/random/LICENSE.md`（移植的 5 个源文件全部位于该模块；NCSA/3-Clause BSD 双许可，Copyright (c) 2019 Kevin Sheppard）与 `numpy-2.4.6.dist-info/licenses/LICENSE.txt` 的首节（NumPy 顶层 BSD-3-Clause，Copyright (c) 2005-2025, NumPy Developers；其余节为 wheel 捆绑二进制库声明，未移植任何二进制组件故未收录，文件头已注明）。文件头保留来源/版本/移植文件清单；未自行撰写或改写任何许可文本、未虚构作者。README「NumPy 来源、许可与版本边界」、`cpp/sjrng_core.cpp`、`cpp/sjrng_ziggurat_tables.hpp`、`cpp/sjrng.py` 的许可指针同步改为指向该文件。

**本轮实际验证（2026-10-16，本机）**：

- `bash slurm/test_submit_controlflow.sh` → `control-flow regression: 52 passed, 0 failed`（exit 0；桩 sbatch/squeue，未连接任何真实集群）。
- `python3 ../verify_juno_supervisor.py`（当时脚本位于上级根目录，现位于 `../doc/`）→ exit 0，23/23 LOCAL_PASS（16 组随机位级对照 + 7 项监督者自有调度器桩检查；其中 "unknown scheduler state rejected" 在新的按用户查询形式下仍保守拒绝），报告写入 `supervisor_verification_SuperJUNO.json`（现位于 `../doc/`）。

本轮未重复微基准；无任何官方 PE/tvE、集群计时、加速比或 GPU 运行的宣称（状态仍为 §9 所列 CLUSTER_PENDING / DATA_PENDING）。`cpp/sjrng.py` docstring、`cpp/sjrng_core.cpp` 与 `cpp/sjrng_ziggurat_tables.hpp` 头注释的改动均为许可指针文字，不改变任何可执行行为。

## 12. 集群开通与调试记录（2026-10-19，CLUSTER_PASS for one）

本节记录 2026-10-19 的集群开通会话；原始作业输出与计时 CSV 逐字节保存在
`records/cluster_20261019/`（含 `session_record.json` 结构化索引）。接入方式：
本机 `papp_cloud ssh sczf074@bscc-n16r4` → 登录节点 `ln01`；**所有计算经
sbatch 于 `gpu` 分区计算节点执行**（g0024/g0030/g0050/g0053 均实测 RTX 4090），
登录节点仅做上传/解压/环境安装/提交与查询。

**部署结果（一次会话内完成）**：

| 项 | 状态 |
|---|---|
| 代码上传 | 本地打包（排除 `*.so`/`__pycache__`/过程报告）→ `~/build/SuperJUNO`；`upstream/` 含 gitfile+`official_git_backup/`，集群上 `git log -1` = `82f25ea`、`git status` 与本地一致（2 处 env-gated 修改） |
| 官方数据 | 登录节点 curl 下载 `data.tar.gz`（2,877,254,310 字节，与分享页 content-length 一致）→ 解压至 `~/run/superjuno/data`（2.9G，8 类文件齐全） |
| juno 环境 | 官方命令在 ln01 执行（属允许的环境安装）：`conda create -n juno python=3.11 pip`、`conda install cuda=12.4`、`pip install -e .`；CuPy 14.2.0 从 sdist 编译约 25 分钟；numpy 2.4.6（与本地逐位移植验证同版本） |
| 运行层 | `~/run/superjuno`（upstream 内容 + slurm 三脚本 + make install 产物） |

**集群暴露并修复的两个部署问题（数值代码零改动）**：

1. **conda 激活覆盖 Makefile 编译选项**（作业 1331811 失败）：`conda activate`
   导出 `CXXFLAGS/LDFLAGS` 环境变量，Makefile 的 `?=` 因环境已有值而放弃默认
   选项，丢失 `-shared -fPIC -std=c++17` → 链接器按可执行文件报
   `undefined reference to main`。修复：构建作业在 `conda activate` 后
   `unset CFLAGS CXXFLAGS LDFLAGS CPPFLAGS` 再 `make install`（集群
   `~/build/SuperJUNO/build_test.slurm`，Makefile 本身未改）。重提作业
   1331820：编译行含全部预期选项，`test_sjrng.py` **18/18 OK**（集群
   numpy 2.4.6 位级对照，`SUPERJUNO_SJRNG` 的集群前置由此满足）。
2. **sbatch 批处理 shell 无 module 命令**（作业 1331825 失败）：run.slurm 第
   79 行 `module load` 报 `module: command not found`（非登录 shell 不初始化
   Environment Modules）。修复：`slurm/run.slurm` 在首个 `module load` 前加
   `source /etc/profile.d/modules.sh`（回退 `/usr/share/modules/init/bash`）
   的初始化兜底；本地与集群两份副本同步。

**`one` 官方比较结果（全部 `--compare`，submit.sh 提交，submissions.tsv 审计）**：

| 作业 | 配置 | 节点 | prior/fsmp/recon (s) | TOTAL (s) | 结果 |
|---|---|---|---|---|---|
| 1331839 | 基线（开关全关，冷 JIT 首跑） | g0024 | 103.2/129.1/254.6 | 486.8 | PE PASS、tvE PASS、Final 0 |
| 1331852 | `SUPERJUNO_SJRNG=1`（热缓存） | g0030 | 34.8/95.7/181.6 | 312.1 | 同上；stderr 确认 C++ 后端激活 |
| 1331864 | 基线重跑（热缓存） | g0053 | 30.6/93.5/181.3 | 305.5 | 同上 |

- 冷/热差异（486.8→305.5）来自 CuPy JIT 内核首跑编译，缓存位于
  `output/tmp/`，跨提交存活（submit.sh 归档产物与日志但不动 tmp/）。
- **诚实结论**：`one` 上 SJRNG 端到端无收益（305.5 vs 312.1，跨节点噪声内）；
  `one` 的 raw 波形仅 1 个事例，逐波形 RNG 循环占比可忽略，微基准 2× 不转化
  属预期。SJRNG 的集群结论是**正确性验证通过**（位级 18/18 + 比较器双 PASS），
  性能取舍留待 0/1/2 实测后决定默认开关。
- 下一步（未执行）：`SUPERJUNO_BETA_INV=1` 单独过 `one --compare`；通过后按
  §8 组合并提交 0/1/2；正式点各点 TOTAL 与 time.csv `real_s` 核对。

## 13. 官方优化方向实施（2026-10-19 下午，进行中）

官方题目页给的优化边界：保持数学模型、采样规则、计算规模、精度不变；**明示可调**
"事例重建分块数（参考 10）与 GPU batch 大小"；资源上限单节点 2 GPU。据此实施（全部
env-gated、默认=上游原行为；`git -C upstream diff` 现 4 文件：run.py、src/fsmp.py、
move.py、prepare.py）：

| 旋钮 | 位置 | `one` 验证（官方 `--compare`） | 结论 |
|---|---|---|---|
| `SJ_BLOCKS=<n>` | run.py 阶段3 `-n`（仅参数化，计时/比较逻辑零改动） | 作业 1332124（=5）：双 PASS、Final 0、304.9s@g0005 | 接通且正确；性能效应待正式点 |
| `SUPERJUNO_VRAM_DIV=<d>` | fsmp.py `multi_data` 显存除数 | 作业 1332069（=2）：**回归**——工作集 U+F≈20GB 超缓存友好规模，阶段2 Part0 16 分钟未完成，已 scancel | 上游 /5 已按 F 足迹调好；默认保持 5 |
| `SJ_SHARDS=<n>`（阶段2） | fsmp.py orchestrator/workers + parquet 合并 | 作业 1332236（=2、SJ_GPUS=2）：双 PASS、250.4s；**阶段2 93.5→55.9s（1.67×）** | 正确性通过；worker 各占 1 GPU、合并日志确认 |
| `SJ_SHARDS=<n>`（阶段2+3） | main.py/recon.py 块轮转 + /Recon 事例轴合并 | 作业 1332247（=2）：双 PASS、250.2s（含空分片边界路径——one 仅 1 块，shard1 零块仍正确合并） | 正确性通过 |
| `SUPERJUNO_BETA_INV=1` | move.py vmove1（前轮已实现） | 作业 1332185：双 PASS、Final 0、319.5s@g0087 | 集群正确性通过 |
| `SUPERJUNO_SJRNG=1` | prepare.py（前轮已实现） | 前轮 1331852 双 PASS | 集群正确性通过 |

工作负载事实（日志实测）：`one`=1899 波形→3 parts、1 事例→1 块；0/1/2 的 raw 是
one 的 ~103×（272MB vs 2.6MB）→ 阶段2 约 240 parts，PE 输出 ~134MB。
`one` 的 TOTAL 节点间方差显著（305→393s，g0053→g0017），计时对比必须注明节点。

**分片安全性依据**（代码核对）：`multi_data` parts 间零共享状态；`prepare_random`
按 part 内 (eid,ch) 派生流（与批界无关）；`prepare_flip(seed,trials)` 全局一次性；
比较器 `compare_PE` 按 (eid,ch,step) 键对齐（行序无关）；`FullReader` 内部 groupby
+重排序（行序无关）；recon 随机流按 EventID 派生（与分块无关）——块/批重打包只改变
执行调度，不改变任何链的计算量与随机量归属。

**正式基线（2026-10-19，全部双 PASS、Final 0，单 GPU 上游路径）**：

| 测例 | 作业 | 节点 | prior/fsmp/recon (s) | TOTAL (s) |
|---|---|---|---|---|
| 0 | 1332023 | g0059 | 417.7/5487.4/2414.9 | **8320.1** |
| 1 | 1332024 | g0050 | 684.4/5635.8/2281.4 | **8601.5** |
| 2 | 1332025 | g0050 | 546.2/5643.4/2296.6 | **8486.3** |

剖面：阶段2 占 65-66%、阶段3 占 27-29%、阶段1 占 5-8%。计时 CSV 与作业 .out 已存
`records/cluster_20261019/`；同次运行的 compare 报告在集群归档
`output/archive/{0,1,2}/20261019T10*/`。

**正式点实验结果（2026-10-19 晚，全部官方 `--compare`）**：

| 作业 | 配置 | 节点 | TOTAL (s) | vs 基线 | PE / tvE |
|---|---|---|---|---|---|
| 1332394 (case0) | SJ_SHARDS=2（两阶段） | g0019 | 4773.3 | 1.74× | PASS / **FAIL** |
| 1332403 (case1) | +SJ_BLOCKS=4 | g0018 | 4088.3 | 2.10× | PASS / **FAIL** |
| 1332395 (case2) | +SJ_BLOCKS=4+SJRNG | g0019 | 4442.7 | 1.91× | PASS / **FAIL** |

阶段分段（case0）：fsmp 5487→2982s（**1.84×**）、recon 2415→1420s（但 tvE 无效）。

**tvE 失败根因调查（判别实验链，全部经 sbatch）**：

1. **阶段2 分片逐位一致**：case0/case2 的 PE 与各自基线归档逐位全等（790 万行，
   keys/offset/nu_lc/PEt/e 全等）——**SJRNG 在正式规模也逐位一致**。
2. tvE 差异从 **step 0 的初始顶点 x/y/z/E（ulp 级）** 开始（97/100 事例），源於
   `probe.Init` 的 cuBLAS float32 批量矩阵乘；2000 步链放大后 ~2-3 个事例越过
   1e-6 容差（比较器只报告越界处）。
3. 对照实验：单进程重跑（g0087）≡ 双worker同卡（g0024）≡ 双GPU分片（g0019）
   **三者逐位一致**；均 ≠ g0059 基线。即分片执行自身跨节点完全稳定，基线
   （g0059/g0050，tvE 过参考）是另一签名。
4. 各节点 GPU 型号/VBIOS（95.02.3C.40.40）/驱动完全一致；分界因素未定，
   g0063 全管线验证（1332677）与钉 g0059/g0050 的阶段3 对照在跑。
5. `SJ_BLOCKS=4` 另有独立失败模式（块重打包改变张量填充宽度）→ **分块数保持 10**。

**当前候选配置**：`SJ_SHARDS=2 SJ_GPUS=2 SUPERJUNO_SJRNG=1`（阶段2 分片 +
C++ 随机流；阶段3 原样单进程）——预期 ~5800-6000s/点（~1.45×），正确性由
1332677 验证中。阶段3 分片已被 `SJ_RECON_SHARDS`（默认 1=上游）隔离并默认关闭。

**候选配置判定（1332677，g0063）**：TOTAL **5577.2s（1.49×）**、fsmp **1.95×**
（2815s，含 SJRNG 收益）、recon 2364s（原样）；PE PASS、**tvE FAIL**——在
"PE 逐位一致 + 阶段3 完全上游单进程"下仍 FAIL，**坐实 tvE 结果与节点绑定**：
g0059/g0050（三个基线所在）匹配参考；g0018/g0019/g0024/g0063/g0087 为另一
位稳定签名（2-3/100 事例越界）。硬件全同，根因未定，缓解=钉节点。

**根因找到并修复（2026-10-20 上午，判别实验 testR/testC/cvdA/cvdB/envrep/on590c/on500c）**：

tvE 失败有两个相互独立的上下文因素，均与计算内容无关：
1. **PE 行序**：分片合并按 shard 顺序拼接改变了 parquet 行序；`FullReader` 的
   稳定排序按行序裁决 max_z_length 平局 → 块成员变化 → 张量宽度变化 → 链发散。
   证明：testR（把合并 PE 重排回单进程行序后仅跑阶段3，g0059 gpu:1）输出与基线
   归档**逐位一致（0/100）**。
2. **CPU 数**：gpu:2 作业默认 16 CPU，参考上下文是 gpu:1 的 8 CPU。证明：testC
   （gpu:2 + `--cpus-per-task=8`）输出与 gpu:1 签名完全一致。
已排除：CVD 可见性（cvdA≡cvdB）、run.py 缓存环境（envrep≡single）、节点
（g0059/g0050/g0087 的 gpu:1 直跑三者逐位一致）。

**修复**：fsmp.py 分片 worker 每 part 写独立 parquet，父进程按 part 序号升序合并
（=单进程行序）；submit.sh 增加 `SJ_CPUS`（默认 8）强制 `--cpus-per-task=8`。
**最终结果（2026-10-20 下午，配置 `SJ_SHARDS=2 SJ_GPUS=2 SJRNG=1 SJ_CPUS=8`，
全部经官方 `--compare`、submit.sh 提交、产物原地保留）**：

| 测例 | 作业 | 节点 | prior/fsmp/recon (s) | TOTAL (s) | vs 基线 | 判定 |
|---|---|---|---|---|---|---|
| 0 | 1334370 | g0050 | 372.2/2909.3/2342.7 | **5624.2** | 1.48× | 双 PASS、Final 0 |
| 1 | 1334350 | g0050 | 372.5/2896.1/2356.1 | **5624.8** | 1.53× | 双 PASS、Final 0 |
| 2 | 1334083 | g0050 | 392.2/2849.7/2316.2 | **5558.1** | 1.53× | 双 PASS、Final 0 |

平均加速 **1.51×**（阶段2 达 1.9-1.98×，阶段3 逐位一致保持原速，阶段1 未动）。
佐证运行 1333810（纯上游 gpu:1 复刻）亦双 PASS（8415.9s），确认基线上下文可复现。
证据：`records/cluster_20261019/`（三点 .out、time.csv、compare.txt）+ 集群
`~/run/superjuno/output/` 同次运行产物。

**最终候选配置与正式跑（已提交）**：`SJ_SHARDS=2 SJ_GPUS=2 SUPERJUNO_SJRNG=1
SJ_NODE=g0059,g0050`（submit.sh 新增 `SJ_NODE`→`sbatch -w`；回归 52/52 仍过）。
作业 1332822/1332823/1332824 = case 0/1/2，排队中（两节点满载）。**BETA_INV
不用于正式跑**（改变 PE 位型 → 与 SJ_BLOCKS=4/非A节点相同的链发散机制）。
钉节点复验对照 1332775/76（on590b/on500b）同队。

