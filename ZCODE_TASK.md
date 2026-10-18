# zcode 实施任务：SuperJUNO

只修改 SuperJUNO/。先读 docs/official.txt、upstream/README.md、scripts/run.py、test/README.md、docs/recon.tex，确认真正热点/接口再实施。upstream由监督者克隆官方GitHub仓库。

本机无CUDA、CuPy、HDF5库/数据和GPU。目标是可审查的C++优化候选与严格的接入/回退机制，不盲目重写随机算法，不宣称已通过GPU比较。

1. 记录上游commit，明确三阶段与不可变参数、随机量映射和官方比较接口。不得修改scripts/run.py计时逻辑、test/比较器、输入和参考结果。
2. 找一个真实可独立验证且不改变数学和随机量归属的热点，编写C++17或CUDA C++候选（如上游已有逐元素算子对应的融合/内存复用）；必须能指出准确上游源位置/公式。提供纯CPU小规模reference对照、边界测试和实际构建。若无可证明安全的热点，可以交付C++运行记录审计工具及具体优化设计，但必须明确这不是计算加速实现。
3. 若提供GPU优化候选，只以显式opt-in接入，默认保留上游计算；给出未来基线/候选完整三阶段官方比较命令。不能修改随机seed/迭代次数/活跃顶点/输出schema，不能用参考结果作为输入。禁止fast-math、改变随机数库、变更浮点归约次序。
4. 提供sbatch和提交脚本，单节点1 GPU（可参数化到最多2），启动官方 python scripts/run.py <id> --compare；one先验证，再0/1/2分别运行且防覆盖，日志目录提交前创建。完整保留同次运行CSV、输出、compare报告和stdout/stderr。
5. Makefile、README.md、IMPLEMENTATION.md明确哪些本地验证通过、哪些待集群/数据。不要把CPU微测试当官方PE/tvE通过或加速比。不要创建额外agent。
