# NumPy 随机流兼容性核对（Codex，2026-10-16）

已核对 NumPy 官方文档：Generator 不承诺跨版本兼容；改进分布实现时生成结果可能改变。PCG64 的固定种子整数流兼容承诺不能推广到 Generator 的所有浮点分布结果。因此本地按位相等仅适用于实际测试的环境与调用，不应表述为所有 NumPy 版本通用保证。

官方来源：

- https://numpy.org/doc/stable/reference/random/generator.html （No Compatibility Guarantee）
- https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html （Compatibility Guarantee）

当前移植来源与本地测试版本为 NumPy 2.4.6；目标集群应重新构建 C++ 共享库、逐位对照实际 NumPy、再运行官方 PE/tvE 比较。默认保留上游路径。
