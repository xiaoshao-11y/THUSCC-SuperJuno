# 最终修复轮：完成现有实现的可验收交付

用户要求zcode实现、Codex监督。首轮代码已完成大部，现只修复，不新增优化方向；请直接实施到结束，不创建agent。先读根AGENTS、本目录IMPLEMENTATION.md，然后逐项完成ZCODE_REVIEW_1.md和ZCODE_REVIEW_2.md（包括文件末尾的构建补充）；这些文件记录了监督者真实复现及明确的验收条件。

重点：C++随机库ABI溢出/异常安全、连续整数field转换及版本保证纠正、显式-pthread；第一次干净submit.sh应成功、同id活跃提交不得归档/覆盖、输入和资源检查；用本地控制流桩验证而不访问任何真实集群。

现有C++随机候选16组独立对照已通过，保留它与原NumPy映射。默认优化开关关闭；run.py/test/不可修改。现有融合核只是未接入草稿，不再扩展。集群数据、CuPy/CUDA/正式PE/tvE结果都没有，报告据实标注。

完成make test（适度规模即可，不要反复跑5000-stream微基准来声称流水线加速）；重跑根目录python3 verify_juno_supervisor.py。必要时让测试依赖明确NumPy版本，目标集群仍需重新build/test。补齐README.md、更新IMPLEMENTATION.md消除未经证明的跨版本保证并记录审查修复及真实测试结果；保留NumPy来源/许可；给出不混淆本层目录和upstream目录的部署步骤。

最终回答文件清单、实际测试、仍待官方资源验证事项；无需等用户批准。
