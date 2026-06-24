# soft-continuum-vlm

本仓库是机器人论文项目 **Embodiment-Aware Vision-Language Manipulation for Contact-Safe Soft Continuum Arms** 的研究代码。

中文题目方向：**面向接触安全操作的软体连续体机械臂视觉语言具身适配方法**。

这不是普通软件 demo。当前目标是把 Feagine/MuJoCo 真实仿真入口、结构化状态抽取、PCC IK、任务阶段专家、专家数据采集、Soft Embodiment Adapter 训练、策略评估和论文图表导出串成可复现的最小实验闭环。VLM、OpenVLA、Octo 等大型模型在明确集成里程碑之前都保持 deterministic stub，不下载大型权重。

## 论文实验主命令

mock 命令只用于 CI 和调试。候选论文实验必须显式使用 `--env feagine_mujoco`，并传入 `configs/env/feagine_mujoco_a03_type_2.yaml`。当前 VLM planner 是离线确定性 baseline，不调用网络，也不下载 OpenVLA/OCTO 或其他大型模型权重。

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/a03_type_2_scene.json

python scripts/run_task_phase_expert.py --task obstacle_avoid_pick --env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml --max-steps 120 --output outputs/rollouts/feagine_obstacle_avoid_pick_expert.json

python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 20 --max-steps 150 --output data/demos/feagine_obstacle_avoid_pick_expert.npz --env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml --seed 0 --save-config-snapshot

python scripts/train_adapter.py --demo data/demos/feagine_obstacle_avoid_pick_expert.npz --epochs 50 --batch-size 32 --train-ratio 0.8 --val-ratio 0.2 --seed 0 --output outputs/checkpoints/adapter_best.pt --metrics-output outputs/metrics/adapter_training.json

python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert adapter vlm_planner_ik --adapter-checkpoint outputs/checkpoints/adapter_best.pt --env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml --num-episodes 10 --max-steps 150 --seed 0 --output outputs/metrics/feagine_policy_eval.json --csv-output outputs/metrics/feagine_policy_eval.csv

python scripts/export_paper_figures.py --metrics outputs/metrics/feagine_policy_eval.json --output-dir outputs/figures
```

所有论文图表都必须由保存的 JSON/CSV 指标生成，不允许手工填写结果。

## 当前进度

- Milestone 0：项目骨架和 Feagine 安装验证已完成。
- Milestone 1：Feagine MuJoCo wrapper 已支持 reset/step、human viewer 与 headless 覆盖。
- Milestone 2：四个任务已有确定性评估逻辑。
- Milestone 3：PCC IK、任务阶段专家和 SafetyProjector 已形成可测试控制路径。
- Milestone 4：`TaskPhaseExpert + MockContinuumEnv` 可生成 `.npz` demo 和 metadata。
- Milestone 5：Soft Embodiment Adapter 支持训练、验证集划分、best checkpoint 和 metrics 输出。
- Milestone 6：Deterministic VLM planner stub 可把中英文语言转成结构化子目标和安全约束。
- Milestone 8 到 11：mock policy evaluation、真实 Feagine 场景绑定、任务阶段专家 rollout、论文图表导出都已有命令入口。

尚未完成：

- 真实视觉检测或真实 VLM planner。
- 基于可用 Feagine/MuJoCo runtime 的批量真实专家数据采集。
- 使用真实 Feagine 数据训练 adapter 并做论文级统计。
- OpenVLA/OCTO 类 baseline 的真实接入。

## 目录结构

```text
configs/      环境、任务、方法和评估配置。
data/         本地生成的数据目录，demo 文件默认被 Git 忽略。
scripts/      安装、验证、demo、数据采集、训练、评估和图表入口。
src/          可导入的 Python 包。
tests/        不依赖 MuJoCo 图形窗口的单元测试。
experiments/  实验记录和复现实验说明。
outputs/      checkpoint、metrics、figures、rollouts 等生成物目录。
```

Feagine 必须保留在本仓库外部。默认相对位置是：

- `../feagine_simulation`
- `../feagine-simulation`

如果 Feagine 目录在别处，请设置 `FEAGINE_SIM_ROOT`。

## 快速验证

推荐先激活环境并安装本仓库：

```powershell
conda activate feagine_vlm
pip install -e .
```

验证 Feagine 安装：

```powershell
python scripts/verify_feagine_install.py
```

运行单元测试：

```powershell
pytest
```

运行 MuJoCo human viewer demo：

```powershell
python scripts/run_demo_env.py
```

无窗口运行同一个 demo：

```powershell
python scripts/run_demo_env.py --headless
```

MuJoCo human viewer 的亮度和默认视角由 `configs/env/feagine_mujoco_a03_type_2.yaml` 控制：

```yaml
env:
  render_mode: human
  visual_preset: debug_bright
  viewer_camera:
    lookat: [0.0, 0.0, 0.35]
    distance: 1.0
    azimuth: 120
    elevation: -20
```

这些字段只影响运行时显示效果，例如 headlight 和 viewer camera，不会修改 Feagine 安装包、MJCF 文件或仿真动力学。

## 真实 Feagine/MuJoCo 场景绑定

`MockContinuumEnv` 只用于 CI 和调试。真实论文实验必须使用 `FeagineMujocoEnv`。

真实环境通过 `SceneRegistry` 将 MuJoCo body/geom 绑定到任务对象。候选 `body_names` 和 `geom_name_patterns` 写在 `configs/env/feagine_mujoco_a03_type_2.yaml`。如果对象无法在真实 MJCF 中解析，observation 会保留该对象并写入 `available: false` 和 `missing_reason`，然后根据 inspection 脚本输出修正 YAML。

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/a03_type_2_scene.json
```

结构化 Feagine observation 包含：

- `rgb`
- `depth`
- `language`
- `proprioception`
- `robot_state`
- `objects`
- `contact`

## PCC IK 与任务阶段专家

控制闭环只依赖结构化 observation：

```text
observation + task
  -> TaskPhaseExpert
  -> PccIkController
  -> SafetyProjector(mode="hold_current")
  -> Feagine action
```

`PccIkController` 在给定可达 waypoint 时不再返回纯零动作。当前运动学是近似 PCC/section-angle 模型，使用数值 Jacobian；后续可以把 `continuum_kinematics.py` 中的近似 FK/Jacobian 替换为 `pyfeagine_sim_core` 的精确实现，而不改变控制器接口。

mock 调试命令：

```powershell
python scripts/run_task_phase_expert.py --task obstacle_avoid_pick --mock-env --max-steps 80 --output outputs/rollouts/mock_obstacle_avoid_pick_expert.json

python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 3 --max-steps 80 --output data/demos/obstacle_avoid_pick_task_phase_debug.npz --mock-env --seed 0
```

## Observation Schema

任务评估、mock 环境、数据采集和 adapter 训练都读取结构化 observation。后续视觉模型只负责估计这些字段。

```python
observation = {
    "rgb": ...,
    "depth": ...,
    "language": "...",
    "proprioception": ...,
    "robot_state": {
        "tip_pose": {
            "position": [x, y, z],
            "orientation": [w, x, y, z],
        },
        "section_angles": [...],
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    },
    "objects": {
        "red_object": {
            "pose": {
                "position": [x, y, z],
                "orientation": [w, x, y, z],
            },
            "grasped": False,
        }
    },
    "contact": {
        "max_force": 0.0,
        "max_penetration": 0.0,
        "contacts": [
            {
                "geom1": "...",
                "geom2": "...",
                "position": [x, y, z],
                "normal": [nx, ny, nz],
                "force": [fx, fy, fz],
                "distance": 0.0,
            }
        ],
    },
}
```

四个任务的最小评估规则：

- `pick_red_object`：目标物体已抓取，且高度超过 lift 阈值。
- `obstacle_avoid_pick`：目标已抓取，且与 obstacle 的接触力不超过阈值。
- `contact_push`：目标物体进入目标区域，且最大接触力不超过安全阈值。
- `rotate_and_place`：目标物体位置误差和姿态误差均低于阈值。

## Action Schema

项目内部 action 字段直接对齐 Feagine MuJoCo runtime：

```python
action = {
    "section_angles": [...],
    "grip_command": 0.0,
    "grasper_rotation": 0.0,
}
```

字段名必须使用 Feagine 的 `grasper_rotation`，不是 `gripper_rotation`。`SafetyProjector` 会裁剪 `section_angles`、`grip_command` 和 `grasper_rotation`；接触力或穿透深度超过限制时，真实控制推荐使用 `hold_current` 模式冻结到当前状态。

## Dataset Schema

`scripts/collect_scripted_demos.py` 生成 `.npz` demo 文件，并保存同名 `.json` metadata。每个 step 包含：

- `proprioception`
- `contact`
- `language`
- `language_feature`
- `morphology`
- `action`
- `action_vector`
- `reward`
- `done`
- `success`
- `task_name`
- `phase`
- `episode_id`
- `step_id`
- `target_state`
- `raw_action`
- `safe_action`
- `safety`
- `task_metrics`

metadata 包含 git commit、命令行、seed、task/method/env 配置、action schema 和 observation schema version。

## Deterministic VLM Planner

当前 planner 是 deterministic stub，不调用真实 VLM，也不下载权重。它支持中文和英文关键词规则：

- 颜色：红、蓝、绿、黄、黑，`red`、`blue`、`green`、`yellow`、`black`
- 动作：抓取、绕过、推动、旋转、放置，`grasp`、`avoid`、`push`、`rotate`、`place`
- 安全约束：轻轻、不要碰、`gentle`、`gently`、`avoid`

输出包含 `target_object`、`avoid_objects`、`approach_side`、`grasp_mode`、`contact_force_limit`、`requires_rotation`、`requires_push`、`subgoals` 和 `language_constraints`。后续真实 VLM 只需要替换 `BasePlanner.plan()` 实现。

## 评估与图表

mock baseline 评估：

```powershell
python scripts/evaluate_baselines.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --baselines scripted_expert adapter vlm_planner_ik --num-episodes 3 --max-steps 60 --mock-env --output outputs/metrics/baseline_debug.json --csv-output outputs/metrics/baseline_debug.csv

python scripts/export_paper_figures.py --metrics outputs/metrics/baseline_debug.json --output-dir outputs/figures
```

policy 评估：

```powershell
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert vlm_planner_ik --mock-env --num-episodes 2 --max-steps 80 --seed 0 --output outputs/metrics/mock_policy_eval.json --csv-output outputs/metrics/mock_policy_eval.csv
```

图表导出包括：

- `success_rate_by_task.png`
- `contact_force_by_policy.png`
- `penetration_by_policy.png`
- `safety_clip_count_by_policy.png`
- `summary_table.csv`
- `summary_table.md`

mock-env 指标只用于 pipeline 验证和论文草稿排版，不能作为最终论文结论。

## 开发约束

- 不要复制或修改 `../feagine_simulation` 或 `../feagine-simulation`。
- Feagine 只能通过安装脚本、相对路径或 `FEAGINE_SIM_ROOT` 引用。
- 所有路径必须支持相对项目位置。
- 测试不能依赖 MuJoCo 图形窗口。
- 如果 MuJoCo 或 Feagine 不可用，相关测试必须 graceful skip 或使用 fake/mock runtime。
- 不下载 OpenVLA、Octo 或大型模型权重。
- 每次代码改动后运行相关测试。
