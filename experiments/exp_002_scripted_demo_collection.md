# 实验 002：脚本化专家数据采集

## 命令

```powershell
python scripts/collect_scripted_demos.py --task obstacle_avoid_pick --num-episodes 4 --max-steps 50 --output data/demos/debug_obstacle_avoid_pick.npz --mock-env --seed 0
```

## 生成文件

- `data/demos/debug_obstacle_avoid_pick.npz`
- `data/demos/debug_obstacle_avoid_pick.json`

## 数据字段

每个 step 保存 proprioception、contact、language、确定性 language feature、morphology、JSON action、flatten 后的 action vector、reward、done、success、task name、phase、episode id 和 step id。

## mock 物理限制

mock 环境是确定性、无窗口的测试环境，只近似 tip motion、grasping、pushing 和 grasper rotation。它适合验证 pipeline，不适合作为物理实验结论。

## 下一步

真实 Feagine rollout policy 和场景状态抽取稳定后，把 `--mock-env` 替换为 `--env feagine_mujoco`，并传入真实环境配置。
