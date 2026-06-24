# 实验 003：Adapter 训练

## 命令

```powershell
python scripts/train_adapter.py --demo data/demos/debug_obstacle_avoid_pick.npz --epochs 3 --batch-size 8 --output outputs/checkpoints/adapter_debug.pt --metrics-output outputs/metrics/adapter_debug.json
```

## 生成文件

- `outputs/checkpoints/adapter_debug.pt`
- `outputs/metrics/adapter_debug.json`

## 数据 schema

Adapter 读取确定性 language feature、proprioception、contact state 和 morphology vector，预测 flatten 后的 Feagine action vector。action schema 固定为 `section_angles`、`grip_command` 和 `grasper_rotation`。

## 当前限制

语言编码器是确定性 hash stub，示例数据来自 mock rollout。该结果只能证明训练 pipeline 可运行，不能证明真实 VLM 或真实机器人性能。

## 下一步

真实场景对象和接触状态抽取稳定后，用同一 `.npz` schema 训练 Feagine MuJoCo 专家数据。
