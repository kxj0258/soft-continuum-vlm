# 进度记录

## 2026-07-01

- 已将已有工作区提交为 `3247ce1 Define Feagine 4D action development baseline`。
- 用户批准按完整路线依次实施，并要求自动进行 Git 版本管理。
- 创建开发分支 `codex/roadmap-implementation`，后续采用窄步骤原子提交。
- 未运行任何测试、验证、lint、format、build、install 或仿真命令。
- M1 已增加依赖无关的 4D action space、范围校验、裁剪、任务空间缩放、公开导出、配置约定和目标单元测试。
- M1 测试未运行，验收状态保持待验证。
- M1 已提交为 `b2f7054 Add Feagine 4D action contract`。
- M2 已增加离线 PCC 工作空间采样、有限值/高度过滤、PCA 椭球拟合、左右安全任务区、NPY/JSON/PNG 输出脚本和目标测试。
- M2 测试与工作空间脚本均未运行；碰撞和穿模过滤仍等待真实 MuJoCo 后端复核。
- M2 已提交为 `e812d12 Add PCC workspace modeling pipeline`。
- M3 已增加统一 `IkSolver`/`IkResult`、迭代 PCC IK、单步微分 IK、缩小目标重试、最终保持策略、配置、设计文档和目标测试。
- M3 测试未运行，验收状态保持待验证。
- M3 已提交为 `2a622cb Add unified Feagine IK solvers`。
- M4 已增加 `FeagineActionAdapter`、语义低层命令、runtime alias、线性夹爪映射、任务阶段姿态控制、结构化转换结果、配置、设计文档和目标测试。
- M4 测试未运行，验收状态保持待验证。
- M4 已提交为 `95c857f Add Feagine 4D action adapter`。
- M5 已增加 MetaWorld 风格任务抽象、Reach Left/Right/3D、任务注册、4D backend wrapper、IK/距离指标、deterministic Reach expert、配置、设计文档和目标测试。
- M5 测试未运行，验收状态保持待验证。
- M5 已提交为 `20418bc Add MetaWorld-style Feagine reach tasks`。
- M6a 已增加 Push Left-to-Right、Right-to-Left、Contact Push、接触归因、力/穿透奖励、状态阶段、4D deterministic Push expert、配置、设计文档和目标测试。
- M6a 测试未运行，验收状态保持待验证。
- M6a 已提交为 `cca8dff Add MetaWorld-style Feagine push tasks`。
- M6b 已增加三种 Pick-Place 任务、完整阶段状态机、严格 grasp/lift/place/retract 判定、自动姿态 context、4D deterministic expert、配置、设计文档和目标测试。
- M6b 测试未运行，验收状态保持待验证。

## 历史进度摘要

- 2026-06-23：完成既有 Stage 1-3，包括状态抽取、场景绑定、PCC IK、任务阶段专家、安全投影、数据采集、policy、rollout、evaluation 和图表导出。
- 2026-06-23：当时完整 `pytest` 结果为 `68 passed, 4 skipped`，并成功运行 mock rollout、demo collection、policy evaluation 和 figure export。
- 2026-06-24：完成 README、PLANS、experiments、data/outputs 等 Markdown 中文化和 Windows 单行命令整理。
