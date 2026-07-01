# Feagine 4D Action Interface Design

## Motivation

为对齐 MetaWorld 风格任务接口，并为后续 RL、VLM 和 VLA 扩展预留统一入口，Feagine 的未来顶层动作接口统一为 4 维动作：

```text
[dx, dy, dz, gripper_control]
```

当前仓库中的运行时控制仍以低层 Feagine 命令为主，因此本设计文档定义的是后续顶层接口基线，而不是立即替换现有执行路径。

## Top-level Action

```text
dx, dy, dz:
  task-space end-effector delta motion

gripper_control:
  high-level gripper open/close command
```

- `dx, dy, dz` 表示任务空间中的末端增量位移，而不是 section-level 关节角目标。
- `gripper_control` 只表达高层开合意图，不直接承担夹爪旋转控制。
- 所有未来 task、RL policy、VLM planner、VLA policy 和 scripted expert 都应以这个 4D 接口作为统一输出约定。

## Low-level Feagine Command

4D action 不直接发送给 Feagine 运行时，而是先经过底层 adapter 转换为：

```text
section_angles
grasper_rotation
gripper_open_close
```

在当前 Python 包装器和测试中，夹爪开合字段常见名称仍是 `grip_command`。本设计文档中的 `gripper_open_close` 是更清晰的语义名；在真正接入时，两者应视为同一条低层开合控制通道。

## Grasper Rotation

Feagine 夹爪本身具有旋转自由度，但该自由度不作为顶层 action 直接暴露。

后续推荐由以下模块之一自动决定：

```text
task phase controller
orientation alignment controller
FeagineActionAdapter
```

这样可以让顶层策略先专注于任务空间运动与抓取开合，而将姿态细节保留给更接近执行层的模块。

## Integration Path

推荐的后续接入路径如下：

```text
top-level policy / task logic
  -> [dx, dy, dz, gripper_control]
  -> FeagineActionAdapter
  -> {section_angles, grasper_rotation, gripper_open_close}
  -> FeagineMujocoEnv.step(...)
```

当前阶段不要求实现完整 adapter、IK 或任务重构；这里只固定接口边界，避免后续多套顶层动作协议并存。

## Deprecation Policy

未来不再维护旧的 `section_angles` 顶层 action 接口。

- `section_angles` 可以继续作为底层 Feagine 控制命令存在。
- 现有代码中的 `section_angles`、`grasper_rotation`、`grip_command` 相关实现当前先保留。
- 后续重构时，顶层 task/policy API 应逐步迁移到统一的 4D action 接口。
