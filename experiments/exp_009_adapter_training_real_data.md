# 实验 009：基于专家数据训练 Adapter

## 目的

用保存的专家 demonstration 训练 `SoftEmbodimentAdapter`，支持 train/val split、best checkpoint 保存和带 metadata 的 metrics。

## 命令

```powershell
python scripts/train_adapter.py --demo data/demos/feagine_obstacle_avoid_pick_expert.npz --epochs 50 --batch-size 32 --train-ratio 0.8 --val-ratio 0.2 --seed 0 --output outputs/checkpoints/adapter_best.pt --metrics-output outputs/metrics/adapter_training.json
```

## 注意

checkpoint 会记录 action schema 和 input dimensions。如果当前环境没有 torch，adapter 相关测试会 skip；确定性专家和 VLM-planner policy 的 smoke test 仍可在无 torch 环境中运行。
