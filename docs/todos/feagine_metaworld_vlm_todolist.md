# Feagine-MetaWorld-VLM Long-Term TODO

## 0. 项目开发基线与总体原则

### 0.1 环境与运行基线

- 固定使用项目依赖环境：

```powershell
conda activate feagine_vlm
```

- 以下命令是建议用户手动运行的基础检查命令，不是 Codex 默认自动执行项：

```powershell
pip install -e .
pytest
python scripts/verify_feagine_install.py
```

- 以下命令是建议用户在涉及 MuJoCo 或 Feagine 场景修改后手动运行的检查命令，不是 Codex 默认自动执行项：

```powershell
python scripts/inspect_feagine_scene.py --config configs/env/feagine_mujoco_a03_type_2.yaml --steps 5 --print-bodies --print-geoms --print-sites --output outputs/scene_inspection/feagine_scene.json
```

### 0.2 新接口原则

- 废弃旧的顶层 action 接口，不再维护旧的 `section_angles` 顶层控制入口。
- 新的顶层 action 统一采用 MetaWorld 风格：

```text
action = [dx, dy, dz, gripper_control]
```

- `dx, dy, dz` 表示末端在任务空间中的增量位移。
- `gripper_control` 表示高层夹爪开合控制。
- Feagine 夹爪的旋转自由度 `grasper_rotation` 不直接暴露为 RL/VLM/VLA 顶层 action。
- `grasper_rotation` 后续由底层控制器、任务阶段控制器、姿态对齐控制器或 `FeagineActionAdapter` 自动生成。
- 所有 RL、VLM、VLA 和 scripted expert 后续都应统一使用这个 4 维顶层 action 接口。
- 底层再将 4 维 action 转换为：

```text
section_angles
grasper_rotation
gripper_open_close
```

### 0.3 Codex 执行约束

- Codex 默认只修改代码和文档，不自动运行测试、验证、安装、构建、格式化、lint、资产同步、训练、评估或仿真命令。
- 只有当用户在当前任务中明确要求“运行测试”“运行验证”“执行 pytest”“同步资产”等操作时，Codex 才允许执行对应命令。
- 每次完成后，Codex 只需要列出建议用户手动运行的验证命令。
- 不得在没有实际运行的情况下声称测试通过。

### 0.4 当前阶段范围

- 当前阶段只建立开发基线、接口原则、验证命令、设计文档和轻量配置占位。
- 当前阶段不实现完整 `FeagineActionAdapter`、完整 IK solver 或任务环境重构。
- 当前阶段允许保留现有低层运行接口；后续重构时再将顶层 4D action 接入到现有低层命令链路。
