# AGENTS.md

本仓库是机器人论文研究代码，不是通用软件 demo。

- 不要把 `feagine-simulation/` 复制到本仓库。
- 不要把 `feagine_simulation/` 复制到本仓库。
- 不要修改 `../feagine-simulation` 或 `../feagine_simulation` 下的文件。
- 只能通过安装脚本、相对路径或 `FEAGINE_SIM_ROOT` 引用 Feagine。
- 源代码、配置和测试中的路径必须能从项目相对位置运行。
- 优先选择可测试、可复现、可扩展的实现，不要只做快速视觉 demo。
- 每次代码改动后运行相关单元测试。
- 添加复杂行为前先更新 `PLANS.md`。
- 在明确集成里程碑前，VLM 和 VLA 模块必须保持 deterministic。
- setup、测试和 demo 过程中不要下载 OpenVLA、Octo 或大型模型权重。
- 如果 MuJoCo 或 Feagine 不可用，测试必须 graceful skip。
- 所有 TODO 注释和 deferred engineering notes 都必须说明预期输入、输出和集成路径。
- 当前 Windows 环境中的文档命令示例必须写成单行，不使用续行符。
