# 实验 004：确定性 VLM Planner 调试

## 命令

```powershell
python scripts/run_vlm_planner_demo.py --task obstacle_avoid_pick --language "绕过黑色障碍物，轻轻抓住蓝色圆柱" --mock-env --max-steps 60 --output outputs/rollouts/vlm_planner_debug.json
```

## 目的

验证 deterministic VLM planner stub。它把中文或英文语言转换为结构化目标对象、避障约束、轻柔接触限制和子目标。

## 当前限制

这不是真实 VLM，而是用于 pipeline 测试的确定性规则解析器。mock-env 结果不能作为最终论文证据。

## 下一步

在保持 `BasePlanner` 接口和输出 schema 不变的前提下，用真实 VLM-backed planner 替换 `DeterministicVLMPlanner.plan()`。
