# GR00T Tactile

触觉 ResNet-18 编码器及 GR00T 动作上下文融合，使用同一套根目录 `.venv`。
环境、训练与触觉布局说明见 [项目 README](../README.md)。

- `configs/train.yaml`：正式训练配置。
- `configs/smoke.yaml`：batch size 1、3 步训练验证。
- `gr00t_tactile/data/`：触觉热力图、有效掩码及布局。
- `gr00t_tactile/model/`：ResNet-18 与触觉 token 融合。
- `gr00t_tactile/training/`：YAML 入口与训练 pipeline。
- `gr00t_tactile/inference/`：checkpoint 注册与在线触觉适配。
