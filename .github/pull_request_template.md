<!-- PSuM 目前由核心团队主导，暂不接受未经讨论的外部 PR。
     请先开 Issue 讨论；经确认后提交的变更适用本模板。
     PSuM is currently maintained by the core team; unsolicited external PRs are
     not accepted yet. Open an issue first; this template applies to changes
     confirmed there. -->

### 关联 Issue / Linked issue
<!-- Closes #NN —— 未关联 Issue 的 PR 请先说明背景 / state the background if not linked -->

### 变更说明 / What and why
<!-- 改了什么，为什么改；一个 PR 一件事 / one logical change per PR -->

### 变更类型 / Change type
- [ ] 新功能 / feature
- [ ] 缺陷修复 / bug fix
- [ ] 文档 / docs
- [ ] 重构（不改行为）/ refactor (no behavior change)
- [ ] 其他 / other

### 自查清单 / Checklist
- [ ] 遵循 docs/coding-style.md（文件路径=命名空间、蛇形命名、头文件保护、include 顺序）/ follows docs/coding-style.md
- [ ] `./test/runCheck.sh` 冒烟测试通过；涉及慢层目标时已用 `TIMEOUT=20 RUNCHECK_STRICT=1` 或直接运行复核 / default (fast smoke) runCheck passes; long targets re-verified with `TIMEOUT=20 RUNCHECK_STRICT=1` if touched
- [ ] 新增功能附有测试（test/<module>/ 按 makefile 约定）或示例 / new features ship tests or examples
- [ ] **改动 docs/ 下中文文档时，已同步对应 *_en.md** / Chinese doc changes are mirrored in the matching *_en.md
- [ ] 未触及 config.mk（个人配置走 config.mk.local）/ config.mk untouched (use config.mk.local)
- [ ] 提交信息清晰，一条提交一件事 / clear commit messages, one thing per commit
