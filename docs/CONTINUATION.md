# SuperJUNO 后续实施交接（CONTINUATION）

记录日期：2026-10-17。本文是 `README.md` / `IMPLEMENTATION.md` 之后的**当前状态与
后续路线**文件：进展快照、输入与参考结果边界、比赛 RTX4090 集群的重验清单、可继续
的本地任务与正式验收清单。历史实施与审查细节以 `IMPLEMENTATION.md` 为准；本文只记
当前事实，不重复其内容。状态词汇：`LOCAL_PASS` / `CLUSTER_PENDING` /
`DATA_PENDING` / `MODEL_PENDING`。

## 1. 进展快照（2026-10-17 核对）

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
  云实例**，SuperShip 的 RTX3090 实例不用于本题。集群接入仍 `CLUSTER_PENDING`
  （详见 §3 前提与 §4；客服反馈材料 `../doc/GPU_LOGIN_SUPPORT.md`，资源现状
  `../doc/RESOURCE_HANDOFF.md`）。
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

## 3. 比赛 RTX4090 集群必须重验的步骤（CLUSTER_PENDING 清单）

前提：本题按官网要求使用**并行科技 RTX4090 集群**——经并行科技云桌面 SSH 接入、
Slurm sbatch 提交、单节点最多 2 GPU（`docs/official.txt`；2026-10-17 复核快照
`../doc/records/cluster_route_20261017/`）。**不创建独立 RTX4090 云实例**；用户当前
只创建了 SuperShip 的 RTX3090 实例，不得当作本题环境，不得用它做本题计算。

集群接入现状（`CLUSTER_PENDING`）：云桌面 SSH 连接即 `connection closed`、本机 CLI
三条线路 `Permission 'loginCluster' denied`、云桌面快传错误码 12213，均自 2026-10-16
起未解决、本轮未重试；平台"超算账号管理"当前可见绑定 **sczf074 / BSCC-N16R4**，但
是否对应比赛指定的 4090 账号/分区待客服/主办方核对。客服反馈材料见
`../doc/GPU_LOGIN_SUPPORT.md`（用户计划 2026-10-18 自行发送；不代发、不重试登录、
不读取密码/证书）；两题资源现状索引见 `../doc/RESOURCE_HANDOFF.md`。

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

- [ ] 比赛 RTX4090 集群接入可用（云桌面 SSH 恢复且账号/分区经客服/主办方确认——
      含 sczf074 / BSCC-N16R4 归属核对；`module load` CUDA 12.4、Miniforge3
      24.11、juno conda 环境，`pip install -e .` 成功，环境构建按官方允许的方式
      在规定节点上进行）
- [ ] `data/` 八类文件就位（§2；one/0/1/2 的 raw + 5 共享 + 各 id 参考结果）
- [ ] 目标节点 `make install` 重编 + `cpp/test_sjrng.py` 位级对照通过（或据此
      禁用 SJRNG 并记录）
- [ ] one 基线（默认路径）`--compare` 三项 exit code 全 0
- [ ] 每个启用中的开关单独在 one 上过 `--compare`
- [ ] 0/1/2 各自 Slurm 作业：Pipeline/两项 Comparison/Final exit code 全 0，
      TOTAL 与 time.csv `real_s` 一致，作业编号与产物同次运行对应
- [ ] 提交物齐全：time.csv、`output/sparsify|PE|tvE/<id>.*` 与 `.log`、
      `.compare.txt`、`superjuno-<JOBID>.out/.err`（0/1/2 三点，one 免提交）
- [ ] report.md 写明瓶颈、方法、配置、各点作业编号/正确性/耗时与加速比
      （只引用作业 TOTAL 与 time.csv，不引用本地微基准）

以上任何一项未完成时，相应结论只能标 `CLUSTER_PENDING` / `DATA_PENDING`，不得写成
已通过。
