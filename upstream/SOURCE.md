# upstream 来源与备份说明

## 官方来源

- 官方仓库：https://github.com/thu-scc/SuperJUNO
- 本地克隆 commit：`82f25ea`（preliminary），分支 `main`，为浅克隆（grafted），
  与 `origin/main` 一致，无本地独有提交、无 stash（commit 亦记录于仓库根
  `upstream_commit.txt`）
- 为把官方源码连同本地改动一并纳入备份仓库，官方克隆的 `.git` 目录于 2026-10-18
  整体移至 `upstream/official_git_backup/`（只移动、未删除、未改动任何内容）。
- 2026-10-18 起 git 功能已按 gitfile 方式恢复：`upstream/.git` 是单行
  `gitdir: official_git_backup` 的 gitfile（相对路径，Git 对 gitfile 的标准
  工作树发现机制）——`upstream/` 为工作树、`official_git_backup/` 为其 git 目录。
  官方对象库、refs、config、index 均未改动；`official_git_backup/info/exclude`
  排除了该目录自身，避免它出现在 `git status`。因此以下命令直接可用
  （`cpp/test_sjrng.py` 与监督脚本即依赖第一条 show）：

  ```
  git -C upstream status && git -C upstream diff
  git -C upstream show 82f25ea67f10ede8881f4dd4656ad78cc40cfd5f:src/analysis/fsmp/prepare.py
  ```

- 如需改回常规布局（`.git` 为实体目录）：
  `rm upstream/.git && mv upstream/official_git_backup upstream/.git`。
- 灾备路径（未执行，仅备查）：备份仓库根 `.gitignore` 排除了
  `upstream/official_git_backup/`，从 GitHub 备份重新克隆时该目录不存在、上述
  gitfile 会悬空；可用官方仓库重建，再按上面的 gitfile 方式接回：
  `git clone https://github.com/thu-scc/SuperJUNO.git tmp &&
  git -C tmp checkout 82f25ea67f10ede8881f4dd4656ad78cc40cfd5f &&
  mv tmp/.git upstream/official_git_backup && rm -rf tmp`
  （重建后 HEAD 为 detached `82f25ea`，对本仓库 status/diff/show 用途与原浅克隆
  等效。）

## 备份时点 upstream 内的本地改动（相对官方 82f25ea）

- 修改：`src/analysis/fsmp/move.py`、`src/analysis/fsmp/prepare.py`
- 新增：`src/sjrng.py`（本地 RNG 封装）、`src/_sjrng_core.so`（cpp/ 编译产物，
  可由 Makefile 重建；为完整备份运行态一并入库）、本说明文件 `SOURCE.md`
  （2026-10-18 新增；`upstream/.git` gitfile 本身 Git 永不列出）

## 未入库内容

- `__pycache__/`、`build/` 等构建缓存（可重建）
- 官方 `.gitignore` 规则中的 `raw/`、`PE/`、`data`、`output` 等数据/输出目录
  （备份时点不存在这些目录）
