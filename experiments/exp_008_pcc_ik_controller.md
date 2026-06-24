# 实验 008：近似 PCC IK 控制器

## 目的

用可测试的确定性控制器替换零动作 placeholder，将目标 tip waypoint 转换为 Feagine `section_angles`。

## 当前模型

`continuum_kinematics.py` 使用明确标注的 section-angle 近似：x/y 位移与加权 section bend 成正比，z 位移随弯曲幅度略微下降。这是 Feagine 控制开发阶段的可解释近似，不是最终标定后的机器人运动学。

## 替换路径

当 `pyfeagine_sim_core` 暴露论文所需的精确 PCC forward kinematics 和 Jacobian 后，替换 `section_angles_to_tip_delta()` 与 `numeric_jacobian_tip_delta()`，同时保持 `PccIkController.compute_action()` 和 action schema 不变。

## 验证命令

```powershell
pytest tests/test_continuum_kinematics.py tests/test_pcc_ik_controller.py
```
