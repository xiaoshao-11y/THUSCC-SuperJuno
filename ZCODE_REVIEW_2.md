# Codex提交脚本复查

只修本题，首轮尚在实施时可以合并；若已完成则作为后续轮次执行。上一份ZCODE_REVIEW_1.md（含构建补充）仍须全部核对完成。

1. 已复现slurm/submit.sh无法做首次干净提交：archive_id最后一句`[ "$moved" = 1 ] && echo ...`在moved=0时返回1；函数在set -e脚本中被直接调用，整个脚本随即退出，还未执行sbatch。Codex用temp仓库+本地sbatch桩复现exit1，记录在supervisor_submission_findings.json；没有真实提交。请显式return0并补干净目录one提交的控制流测试。
2. 防覆盖还需考虑正在排队/运行的同id：当前只在运行前检查文件，提交包装器会先mv现有产物，可能归档一个仍在写的作业或允许同id双提交。请用明确的每case提交/运行锁或Slurm状态检查防止这种情况，duplicate ids参数（例如one one）也应提前拒绝；锁/归档的失败要可恢复且不影响不同id。先检查后归档，保留既有日志内容；归档目录不要仅秒级timestamp重名。实现可保持简单，清楚说明状态边界。
3. run.slurm启动前检查是Slurm、单节点、GPU请求1或2；实际参数one/0/1/2须在任何输出路径创建前验证，不能让`../../x`拼入路径。仍只调用官方run.py --compare，不改其计时/比较逻辑。
4. 补本地Shell控制流回归（sbatch/squeue等均为显式测试桩）：首次one可提交、已有历史产物会归档、同id重复/活跃时拒绝、GPUs>2拒绝、不合法id拒绝；不得写成实际Slurm/GPU通过。实际GPU流程仍待集群。

Codex已验证修正连续数组后的C++随机候选：根目录verify_juno_supervisor.py 16组按位对照通过；最终C++库再有改动后仍须重跑。这个结果不代替PE/tvE官方比较。

实现过程中提醒：squeue -j 历史已结束的jobid可能返回invalid job id/非0，不能因此永远阻止历史产物归档。可用`squeue -h -u "$USER" -o %A`查询当前用户全部活跃JobID，再与本id提交记录求交；查询失败才拒绝。回归包含“历史job已不在队列→允许归档重跑”和“当前job在队列→拒绝”。
