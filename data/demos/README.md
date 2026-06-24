# data/demos/

`scripts/collect_scripted_demos.py` 会把专家 demonstration 数据写入本目录。

预期生成文件：

- `*.npz`：压缩数组，包含 observation 特征、action、reward、done、success、phase 和专家信息字段。
- `*.json`：同名 metadata，包含命令行、seed、schema、环境类型和配置快照。

mock-env 数据只用于 CI 和调试；真实论文数据必须由 `TaskPhaseExpert + FeagineMujocoEnv` 采集。
