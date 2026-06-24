# 发现记录

## 仓库状态

- 当前分支是 `master`，跟踪 `origin/master`。
- 已保留用户已有改动：`configs/env/feagine_mujoco_a03_type_2.yaml` 中 `env.max_episode_steps` 为 `2000000`。
- `FeagineMujocoEnv` 已升级为完整结构化 observation。
- `MockContinuumEnv` 和 `FeagineMujocoEnv` 的 schema 已对齐到 `validate_observation()` 需要的字段。
- `PccIkController` 已不再是纯零动作 placeholder。
- `SafetyProjector` 支持 `drop_blocked_fields`、`hold_current` 和 `scale_down`。
- `collect_scripted_demos.py` 已改用 `TaskPhaseExpert`，并保存 phase、target_state、raw_action、safe_action、safety 和 task_metrics。
- 已添加 policy 层、统一 rollout、`evaluate_policies.py`、AdapterPolicy、VLM planner IK policy、train/val adapter metrics、评估配置和图表输出。

## 验证结果

- 完整 `pytest` 结果：`68 passed, 4 skipped`。
- mock task-phase rollout 已生成 `outputs/rollouts/mock_obstacle_avoid_pick_expert.json`。
- mock demonstration 已生成 `data/demos/obstacle_avoid_pick_task_phase_debug.npz`。
- mock policy evaluation 已生成 `outputs/metrics/mock_policy_eval.json` 和 `outputs/metrics/mock_policy_eval.csv`。
- figure export 已生成 `success_rate_by_task.png`、`contact_force_by_policy.png`、`penetration_by_policy.png`、`safety_clip_count_by_policy.png`、`summary_table.csv` 和 `summary_table.md`。
- 真实 Feagine 命令在当前 shell 中清楚失败，原因是缺少 `pyfeagine_sim_core`、`feagine_mujoco` 或 `mujoco`。

## 附件顺序

1. 真实 Feagine/MuJoCo 状态抽取和场景绑定。
2. PCC IK、任务阶段专家和接触安全 action projection。
3. Feagine expert data、adapter training、policy evaluation 和 paper metrics。

## 文档修订要求

- README 和其他 Markdown 文档使用中文说明。
- Windows 环境中的命令示例使用单行形式。
- 文档中不使用续行符。
