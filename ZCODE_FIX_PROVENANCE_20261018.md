# SuperJUNO 当前测试回归修复（2026-10-18）

用户已授权按计划实施。只修改 `SuperJUNO/`，不要修改公共 `doc/`、HPL/、SuperMSA/、SuperShip/。不要连接远端、不要提交集群作业、不要读取密码或证书。

先阅读 `../AGENTS.md`、`ZCODE_TASK.md`、`ZCODE_CONTINUE.md`、`docs/CONTINUATION.md`、`README.md`、`IMPLEMENTATION.md`、`Makefile`、`cpp/test_sjrng.py`。

## 当前阻塞证据

2026-10-18 Codex 实测：

- `make -C SuperJUNO test` 退出码 2；18 个 RNG 测试中 4 个错误。
- `make supervisor-check` 也在 SuperJUNO 验证入口退出码 2。
- 错误都来自 `cpp/test_sjrng.py` / `doc/verify_juno_supervisor.py` 执行：
  `git -C SuperJUNO/upstream show 82f25ea67f10ede8881f4dd4656ad78cc40cfd5f:src/analysis/fsmp/prepare.py`
  但当前 `upstream/` 没有自己的 `.git`，外层仓库 HEAD 是备份提交，找不到该对象。
- 真实的官方历史仓库保存在 `upstream/official_git_backup/`，其中 refs/remotes/origin/main 指向 `82f25ea67f10ede8881f4dd4656ad78cc40cfd5f`，并且从该仓库执行 `git show <commit>:src/analysis/fsmp/prepare.py` 成功，得到约 6.9KB 的官方源文件。

这不是当前 C++ RNG 算法已证明错误，而是官方上游 commit 溯源环境回归；此前“23/23 通过”在恢复前提后需要重新生成。

## 修复要求

1. **优先恢复 `git -C upstream show 82f25...:<path>` 的工作方式，而不是修改监督器、参考结果、官方 runner 或比较器。** 可以使用 Git 的合法 worktree/gitdir 配置，把 `upstream/official_git_backup` 作为 `upstream/` 的工作树仓库；也可以采用其他不改验证脚本、能让上述命令按原样成功的等效方案。不要把官方提交内容手抄到测试代码，不要改 `cpp/test_sjrng.py` 的参考提取逻辑。
2. 恢复后核对：
   - `git -C upstream show 82f25ea67f10ede8881f4dd4656ad78cc40cfd5f:src/analysis/fsmp/prepare.py` 成功；
   - 当前 `upstream/src/analysis/fsmp/prepare.py`、`move.py` 的显式 opt-in 修改、候选 `.so` 与其他工作文件仍保留；
   - `scripts/run.py`、`test/`、参考比较器和计时逻辑没有改动；
   - 上游来源/commit 记录仍准确。
3. 运行并原样记录：
   - `make test`（预期 18 RNG + 4 公式 + 3 安装 + 52 本地 Slurm 控制全部通过）；
   - `python3 ../doc/verify_juno_supervisor.py`（预期 23/23）；
   - 不运行真实集群、不要声称官方 PE/tvE 或 GPU 已通过。
4. 将实际命令、退出码、修复方式和任何剩余问题写入 `docs/CONTINUATION.md` 或 `IMPLEMENTATION.md`，并保存原始日志。不要把预期结果写成实际结果。
5. 如发现只能修改公共验证器才能解决，停止并报告原因，不要自行绕过。

完成后输出：修改的文件、Git 溯源修复方式、三项测试实际结果、未验证条件。不要顺便做新的数值优化。
