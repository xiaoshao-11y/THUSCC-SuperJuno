# SuperJUNO 后续实施交接（2026-10-17，Codex 下达）

> 环境入口已由 `ZCODE_CLUSTER_ROUTE.md` 修正：SuperJUNO 使用官网要求的RTX4090集群、云桌面SSH和Slurm；不再等待创建独立RTX4090云实例。恢复的客服材料见 `../doc/GPU_LOGIN_SUPPORT.md`，当前集群接入仍待处理。

用户指定后续实施由本机zcode负责，Codex在用户发消息时查进度和正确性。你只修改SuperJUNO/；不改公共doc/、根文件、其他题目。不创建定时任务，不远端启动agent，不读取或上传私钥。

先阅读 ../AGENTS.md、docs/official.txt、上游README和test/README、ZCODE_TASK.md、README.md、IMPLEMENTATION.md。

## 当前事实与约束

- 用户目前只创建了SuperShip的RTX3090云实例；本题使用比赛RTX4090集群，不复用该实例。云桌面SSH故障及绑定账号是否具备4090分区权限，待客服或主办方确认。
- 原始上游commit为82f25ea67f10ede8881f4dd4656ad78cc40cfd5f。scripts/run.py、test/比较器、计时和参考结果不得修改。
- 固定seed=0；先验Lucy2000次、PE4000步和保存规则、重建2000步/20EM不能降低；不得改变随机量归属。
- SUPERJUNO_SJRNG和SUPERJUNO_BETA_INV候选默认关闭；GPU融合草稿未接入。未经目标环境原始随机流与官方PE/tvE比较通过不得默认启用。
- 已通过的本地证据：18随机库+4公式+3安装/接入+52本地假调度控制；Codex23项独立检查。它们不等于官方PE/tvE或GPU计时。
- 官方网页快照明确正式提交经Slurm、单节点最多2GPU；新云实例指南只说明实例创建与SSH，未明确废除这些要求。不能自装假Slurm或捏造作业编号，也不能自行将无Slurm直接运行视为正式提交。

## 本轮完成有用的后续准备

1. 只读核对当前本题输入/参考结果是否实际存在；列出官方run.py检查的六类输入及one/0/1/2所需路径，避免把参考输出作为计算输入。不要搜索整个主目录或读取用户密钥。未发现只写本项目未取得，不猜测主办方未发布。
2. 核实默认候选仍关闭、原始runner和比较器未改，列清在真实RTX4090环境必须重新验证的步骤与命令（目标系统重编C++库、对当场NumPy版本位级对照、one完整官方compare、0/1/2完整比较/计时）。不为新文档写镜像实现的测试。
3. 整理本题README/IMPLEMENTATION过期监督脚本路径和重复操作说明；当前独立脚本位于../doc/verify_juno_supervisor.py，报告在../doc/。保留真实日志、许可证、上游源码和正式运行规则。
4. 写 docs/CONTINUATION.md，准确记录当前进展、可继续的本地任务、RTX4090/输入/调度规则条件及正式验收清单。已有代码若无必要缺陷不要加入未经官方验证的新优化。
5. 如果确有本地实现缺陷，先记录触发条件，再按原始随机映射和数学等效性最小修复，并运行直接相关测试；否则不改数值代码。

不连接SuperShip实例做本题计算；不运行任何远端计算；不触碰HPL/SuperMSA。输出本轮真实完成与未验证条件后结束，后续由Codex按用户消息检查。
