# 实验 007：任务阶段专家

## 目的

使用结构化 observation，通过任务阶段、PCC IK 和接触安全投影，为四个论文任务生成确定性专家动作。

## 阶段状态机

- `pick_red_object`：`approach_above_target`、`approach_target`、`close_gripper`、`lift`、`done`
- `obstacle_avoid_pick`：`move_to_pre_avoid_waypoint`、`arc_around_obstacle`、`approach_target`、`close_gripper`、`lift`、`done`
- `contact_push`：`approach_push_object`、`make_safe_contact`、`push_toward_region`、`retract`、`done`
- `rotate_and_place`：`approach_object`、`close_gripper`、`rotate_grasper`、`move_to_target_pose`、`release`、`done`

## mock 调试命令

```powershell
python scripts/run_task_phase_expert.py --task obstacle_avoid_pick --mock-env --max-steps 80 --output outputs/rollouts/mock_obstacle_avoid_pick_expert.json
```

真实论文候选实验请在具备 Feagine/MuJoCo runtime 的机器上使用 `--env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml`。

## 注意

该 expert 是确定性 baseline，不是最终学习策略。每个 step 会记录 phase、target_state、raw_action、safe_action、safety、contact summary、reward、success 和 task metrics。
