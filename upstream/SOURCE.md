# upstream 来源与备份说明

## 官方来源

- 官方仓库：https://github.com/thu-scc/SuperJUNO
- 本地克隆 commit：`82f25ea`（preliminary），分支 `main`，为浅克隆（grafted），
  与 `origin/main` 一致，无本地独有提交、无 stash（commit 亦记录于仓库根
  `upstream_commit.txt`）
- 为把官方源码连同本地改动一并纳入备份仓库，已将官方克隆的 `.git` 目录整体移至
  `upstream/official_git_backup/`（只移动、未删除、未改动任何内容）。
  如需恢复其 git 功能（并查看本地改动相对官方的 diff）：

  ```
  mv upstream/official_git_backup upstream/.git
  git -C upstream status && git -C upstream diff
  ```

## 备份时点 upstream 内的本地改动（相对官方 82f25ea）

- 修改：`src/analysis/fsmp/move.py`、`src/analysis/fsmp/prepare.py`
- 新增：`src/sjrng.py`（本地 RNG 封装）、`src/_sjrng_core.so`（cpp/ 编译产物，
  可由 Makefile 重建；为完整备份运行态一并入库）

## 未入库内容

- `__pycache__/`、`build/` 等构建缓存（可重建）
- 官方 `.gitignore` 规则中的 `raw/`、`PE/`、`data`、`output` 等数据/输出目录
  （备份时点不存在这些目录）
