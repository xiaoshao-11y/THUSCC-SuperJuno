# 最后一个提交细节与许可收尾

前轮已成功exit0。独立监督已确认make test=18 RNG+4公式+3安装+40本地Shell控制；根verify_juno_supervisor.py=16随机+7调度器桩检查通过。只处理下面两件事，其他代码不再更改，不创建agent，不新增优化：

1. 修slurm/submit.sh::active_check及对应小测试：不要把全部历史jobid传给`squeue -j`（已经完成并从Slurm控制器清理的job可能返回Invalid job id，导致永久不能归档重跑）。先`squeue -h -u "$USER" -o %A`拿当前用户所有活跃ID，再与submissions.tsv该case的历史ID逐一求交；sbatch --parsable若有`;cluster`后缀比较时去掉。查询失败仍保守拒绝，已不存在队列的历史ID允许归档。更新文档描述。测试桩应支持新的查询，必须含已完成清理的job允许重跑和活跃job拒绝，不连真实集群。
2. 已逐句移植NumPy源码/常量表，README写"未再分发源码"不准确；本地找到已安装NumPy2.4.6的实际版权/许可证文件复制到例如third_party/NUMPY_LICENSE.txt，保留来源/version并在README引用。无需法律调研，只保留实际随源码使用的声明，不虚构作者或文本。

完成后跑bash slurm/test_submit_controlflow.sh、python3 ../verify_juno_supervisor.py，实际通过记录到IMPLEMENTATION.md。不再重复微基准，不声称官方PE/tvE或GPU通过。报告改动和测试后结束。
