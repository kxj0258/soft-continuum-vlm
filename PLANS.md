# PLANS.md

本路线图用于保证仓库始终服务于论文项目 **Embodiment-Aware Vision-Language Manipulation for Contact-Safe Soft Continuum Arms**。VLM 和 VLA 集成在明确里程碑之前都保持确定性 stub。

## 全局约束

- 不复制、不移动、不修改 `../feagine_simulation` 或 `../feagine-simulation`。
- Feagine 只能通过安装脚本、相对路径或 `FEAGINE_SIM_ROOT` 引用。
- action schema 固定为 `section_angles`、`grip_command`、`grasper_rotation`。
- 禁止引入 `gripper_rotation`。
- 不下载 OpenVLA、Octo、VLM/VLA 或大型模型权重。
- MuJoCo 或 Feagine 不可用时，相关测试必须 graceful skip 或使用 fake/mock runtime。
- mock-env 只用于 CI 和调试，不能作为论文实验结论。
- 所有命令示例面向当前 Windows 开发环境，写成单行，不使用续行符。

## Milestone 0：项目骨架和 Feagine 安装验证

目标：创建研究仓库、文档、配置、单元测试和 Feagine 验证脚本，且不复制或修改同级 Feagine 发行目录。

输入：项目 brief、同级 `../feagine_simulation` 或 `../feagine-simulation`，可选 `FEAGINE_SIM_ROOT`。

输出：可导入 Python 包、YAML 配置、安装脚本、验证脚本和不依赖 Feagine 的单元测试。

验证命令：

```powershell
python scripts/verify_feagine_install.py
pytest
```

风险与 fallback：本机 Feagine 目录名可能不同。处理方式是设置 `FEAGINE_SIM_ROOT`，项目代码只使用相对路径或环境变量。

## Milestone 1：Feagine MuJoCo 环境封装

目标：为 `a03_type_2` 封装 Feagine MuJoCo runtime，提供稳定 reset/step/render/close 接口，并暴露 observation、contact、robot state 和 viewer/headless 行为。

实现说明：`FeagineMujocoEnv` 在 `reset()` 中加载已安装 runtime，构造 `mujoco.MjData` 和 `FeagineMjcfRobot`。action 直接映射到 Feagine 控制方法：`section_angles` 调用 `robot.drive_section_angles(...)`，`grip_command` 调用 `robot.set_grip_command(...)`，`grasper_rotation` 调用 `robot.drive_grasper_rotation(...)`。真实 wrapper 不再使用旧的 delta action。

验证命令：

```powershell
python scripts/verify_feagine_install.py
python scripts/run_demo_env.py --headless
pytest
```

风险与 fallback：runtime 中 `a03_type_2` 可能映射到 legacy 目录 `a03`。仅当 `a03/preset.yaml` 声明 `preset_id: a03_type_2` 时允许 fallback。

## Milestone 2：四个任务定义

目标：围绕软体连续体机械臂和旋转双指 grasper 定义四个任务。

任务：

- `pick_red_object`
- `obstacle_avoid_pick`
- `contact_push`
- `rotate_and_place`

输出：确定性 task class，包含 language、target object、success criteria 和 evaluation metrics。

验证命令：

```powershell
pytest tests/test_tasks.py tests/test_configs.py
```

风险与 fallback：真实 object state 可能尚不可用。处理方式是通过结构化 observation 和 mock/fake fixture 先稳定 schema。

## Milestone 3：PCC IK、任务阶段专家和 SafetyProjector

目标：实现安全 scripted baseline，组合近似 PCC IK、任务阶段 waypoint 和 contact-aware action projection。

输入：结构化 observation、scene state、task spec、robot state 和 safety limits。

输出：Feagine action dict，字段固定为 `section_angles`、`grip_command`、`grasper_rotation`。

验证命令：

```powershell
pytest tests/test_continuum_kinematics.py tests/test_pcc_ik_controller.py tests/test_task_phase_expert.py tests/test_safety_projector.py tests/test_safety_projector_modes.py
```

风险与 fallback：软臂精确运动学尚未完全确定。当前使用明确注释的 section-angle 近似和数值 Jacobian，后续可替换为 `pyfeagine_sim_core` PCC FK/Jacobian。

## Milestone 4：专家数据采集

目标：用 `TaskPhaseExpert` 为四个任务采集确定性 demonstration。

输出：可重放 `.npz` 数据、同名 JSON metadata、action schema、observation schema version、seed、命令行和配置快照。

mock 调试命令：

```powershell
python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 3 --max-steps 80 --output data/demos/obstacle_avoid_pick_task_phase_debug.npz --mock-env --seed 0
```

真实候选命令：

```powershell
python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 20 --max-steps 150 --output data/demos/feagine_obstacle_avoid_pick_expert.npz --env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml --seed 0 --save-config-snapshot
```

风险与 fallback：MuJoCo 不可用时，真实命令必须清楚失败，不能 silent fallback 到 mock。只有显式 `--mock-env` 时才使用 mock。

## Milestone 5：Soft Embodiment Adapter 训练

目标：训练小型 adapter，将确定性语言特征、proprioception、contact state 和 morphology 映射到 continuum action。

输出：adapter checkpoint、train/val metrics、best checkpoint 和 action decoder 兼容性测试。

验证命令：

```powershell
pytest tests/test_adapter_shapes.py tests/test_train_adapter_smoke.py tests/test_action_decoder.py
```

风险与 fallback：轻量环境中可能没有 torch。处理方式是通过 `pytest.importorskip("torch")` 跳过 adapter 专项测试，其他确定性 pipeline 仍可运行。

## Milestone 6：Deterministic VLM Planner 集成

目标：提供确定性 planner interface，把语言和 scene state 转换成 symbolic subgoals，供 IK policy 使用。

输出：`target_object`、`avoid_objects`、`approach_side`、`grasp_mode`、`contact_force_limit`、`requires_rotation`、`requires_push` 和 `subgoals`。

验证命令：

```powershell
pytest tests/test_vlm_planner.py tests/test_vlm_planner_ik_policy.py
```

风险与 fallback：真实 VLM 行为非确定性。当前保留 deterministic baseline，真实模型只在明确集成里程碑中接入。

## Milestone 7：OpenVLA/OCTO 类 baseline 接口预留

目标：预留 VLA baseline 接口，但默认不安装、不下载、不运行大型模型。

输出：disabled-by-default baseline adapter 和明确安装说明。

验证要求：离线环境运行测试时不得触发任何大型模型下载。

## Milestone 8：实验评估和论文图表导出

目标：评估 scripted、adapter、VLM-planner 和预留 VLA baseline，并导出可复现论文资产。

mock 验证命令：

```powershell
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert vlm_planner_ik --mock-env --num-episodes 2 --max-steps 80 --seed 0 --output outputs/metrics/mock_policy_eval.json --csv-output outputs/metrics/mock_policy_eval.csv
python scripts/export_paper_figures.py --metrics outputs/metrics/mock_policy_eval.json --output-dir outputs/figures
```

风险与 fallback：完整评估可能慢或依赖硬件。处理方式是保留小规模 smoke test，并把真实 simulator-heavy run 设为显式命令。

## Milestone 9：真实 Feagine 场景状态绑定

目标：把真实 MuJoCo body、geom、site、contact 和 robot state 绑定到任务、专家控制器和数据集使用的结构化 observation。

输出：`FeagineMujocoEnv` observation 包含 `rgb`、`depth`、`language`、`proprioception`、`robot_state`、`objects`、`contact`。

验证命令：

```powershell
pytest tests/test_mujoco_state_fake.py tests/test_scene_registry.py tests/test_feagine_observation_schema.py
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/a03_type_2_scene.json
```

风险与 fallback：真实 MJCF 名称可能与默认 YAML 不一致。处理方式是在 observation 中保留 `available: false` 和 `missing_reason`，并打印 body/geom 摘要供人工修正。

## Milestone 10：任务阶段专家真实控制闭环

目标：用结构化 observation 形成真实 Feagine/MuJoCo rollout 控制闭环。

控制链路：

```text
observation + task
  -> TaskPhaseExpert
  -> PccIkController
  -> SafetyProjector
  -> Feagine action
```

验证命令：

```powershell
python scripts/run_task_phase_expert.py --task obstacle_avoid_pick --mock-env --max-steps 80 --output outputs/rollouts/mock_obstacle_avoid_pick_expert.json
```

真实候选命令：

```powershell
python scripts/run_task_phase_expert.py --task obstacle_avoid_pick --env feagine_mujoco --config configs/env/feagine_mujoco_a03_type_2.yaml --max-steps 120 --output outputs/rollouts/feagine_obstacle_avoid_pick_expert.json
```

## Milestone 11：论文实验闭环

目标：从 Feagine expert data 到 adapter training、rollout evaluation、deterministic VLM planner baseline 和 paper metrics，形成最小完整闭环。

输出：

- metadata-rich demonstration
- adapter checkpoint
- rollout JSON/CSV
- metrics summary
- 从 metrics 生成的 PNG/CSV/Markdown 图表

完整验证命令：

```powershell
pytest
python scripts/evaluate_policies.py --tasks pick_red_object obstacle_avoid_pick contact_push rotate_and_place --policies task_phase_expert vlm_planner_ik --mock-env --num-episodes 2 --max-steps 80 --seed 0 --output outputs/metrics/mock_policy_eval.json --csv-output outputs/metrics/mock_policy_eval.csv
python scripts/export_paper_figures.py --metrics outputs/metrics/mock_policy_eval.json --output-dir outputs/figures
```
