# SuperJUNO 环境入口纠正（2026-10-17）

用户明确纠正：SuperShip使用单独云服务器实例；SuperJUNO按官网使用RTX4090集群，通过并行科技云桌面连接并用Slurm提交。不要继续指导创建RTX4090云实例。

你只修改SuperJUNO/内文档，不改任何代码、测试、upstream、公共doc或原始日志。先阅读../AGENTS.md、ZCODE_TASK.md、上游README、docs/official.txt、README、IMPLEMENTATION、docs/CONTINUATION；本轮官网新快照只读见../doc/records/cluster_route_20261017/SuperJUNO.txt、cluster.txt。

1. 修正docs/CONTINUATION.md、README.md、IMPLEMENTATION.md中“等待RTX4090实例”“实例/集群”混用的当前指引，统一为比赛RTX4090集群/分区、云桌面SSH、Slurm。原本正确的算法/验证记录不改。
2. 明确当前平台可见sczf074 / BSCC-N16R4，但是否对应比赛4090资源待客服/主办方核对；云桌面SSH connection closed、CLI loginCluster拒绝、快传12213仍未解决，CLUSTER_PENDING。用户计划2026-10-18反馈，反馈材料路径../doc/GPU_LOGIN_SUPPORT.md；不要代发或重试登录，不读取密码或证书。
3. 保持默认优化关闭、固定随机量与全部迭代、官方runner/比较器不动，先one再0/1/2；集群登录节点只允许查询/文件管理，实际构建、数值测试与计算按规定在sbatch分配的计算资源上执行。C++目标系统重编和NumPy位级对照也不能在登录节点直接跑。
4. 给README和CONTINUATION加到公共资源/客服材料的相对链接，避免复制过多排障细节。
5. 只做文档链接与改动范围核查，不运行make/test、不连接远端、不新增优化。完成后报告。

Codex已修正根工作约定，接下来会核验你的文档与公共恢复材料。
