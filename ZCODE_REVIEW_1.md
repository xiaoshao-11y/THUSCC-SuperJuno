# Codex 独立审查：随机映射候选有实测差异

请在首轮结束前处理，先读supervisor_draft_findings.json。只修改本题，不创建agent；不扩大算法重写范围。

1. 已独立复现wrapper的关键映射错误。监督者从未修改的`git show HEAD:src/analysis/fsmp/prepare.py`提取真正prepare_random作reference，使用requester dtype=[('eid',np.int64),('ch',np.int64)]、4条(0,0),(1,5),(2,10),(2**32,10000)、seed=0、TRIALS=100。四个数组均从[0,1]开始不一致。
   原因：结构化数组field是有stride的视图；`eids.astype(np.int64, copy=False)`在已是int64时仍非连续，直接把ctypes指针交给C++会把channel/下一个field读作EventID。chs同理。必须`np.ascontiguousarray(...,dtype=np.int64)`并验证整数/范围，不允许静默截断。覆盖32/64bit fields、切片view、reordered requester、多线程的按位对照。不要只测dtype导致astype自动拷贝的情况。
2. C++ `const int64_t total=n_streams*trials`在检查前已可能signed overflow。先用除法比较和SIZE_MAX/元素字节数上界，再乘法；无效维度必须在访问输出之前拒绝。C ABI不要让C++异常越过ctypes边界；线程创建失败时先join已经启动的线程，再返回明确错误。
3. 源码注释声称NumPy Generator提供跨版本bitstream保证，这个结论不成立（PCG64底层整数流承诺与Generator的分布/浮点实现不是一回事）。删去过强保证，记录实际NumPy版本及用于移植的源码版本/许可。默认继续原始NumPy；候选必须显式opt-in，未知集群版本需要在目标环境重新按位对照。不得默认启用或把CPU测试当PE/tvE官方通过。
4. 保留标准4000 steps和原始seed/StreamID/每流抽取顺序，np.log仍沿用NumPy。补测试：真正上游reference（可AST提取，避免CuPy依赖）、TRIALS=0/1/4000、空requester、多种大ID、随机顺序、1/多线程、选出稀有normal慢路径的大样本。float64按位比较，不能用宽松allclose掩盖随机量映射差异。
5. 上游run.py、test/比较器和时间计量不得改动。README/IMPLEMENTATION明确：本地按位对照≠官方HDF5/Parquet PE/tvE比较；完整三阶段/4090性能仍待数据与集群。所有候选接入默认关闭、可回退，sbatch通过官方入口。

请修复后make test，更新报告中的本地通过项和待验证项。

构建补充：Makefile目前构建使用std::thread的共享库未显式`-pthread`，本机新glibc可能掩盖问题，目标集群可能是旧发行版。请在编译及链接显式加-pthread，README要求在目标节点重新构建.so（不能把本机GCC16/glibc生成的.so直接当集群可移植产物）。
