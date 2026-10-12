# GR00T Tactile

基于 NVIDIA Isaac-GR00T N1.7 的上身触觉微调分支。
112 路触觉读数生成 128×128 热力图，经 ResNet-18 编码为 16 个 tokens，
在 VLM 输出之后接入动作 DiT。默认冻结视觉和语言骨干，训练触觉分支与动作头。

## 环境

以下命令在仓库根目录执行，使用 Python 3.12 和 CUDA 12.8：

```bash
uv sync --python 3.12
source gr00t_tactile/activate_env.sh
```

激活脚本选择本机匹配的 NVIDIA 驱动库及 `.venv/ffmpeg-runtime` 中的 FFmpeg 7。

## 模型缓存

先在 Hugging Face 获得 [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) 的访问权限：

```bash
hf auth login
hf download nvidia/GR00T-N1.7-3B
hf download nvidia/Cosmos-Reason2-2B
```

## 训练

默认数据为 `gr00t_tactile/dataset/2026-09-24-17-17-20`，机器人为 `unitree_g1_sonic`。
更换数据时修改 YAML 的 `data.dataset_path`；本机默认 batch size 为 1。

```bash
# 3 步验证：加载数据、反向传播、保存 checkpoint
gr00t-tactile-finetune --config gr00t_tactile/configs/smoke.yaml

# 正式训练：步数和学习率在 YAML 中设置
gr00t-tactile-finetune --config gr00t_tactile/configs/train.yaml
```

缓存完整后，可在命令前加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
已验证 3 步离线训练、权重重载和真实数据推理；长期收敛尚需评估。
验证 checkpoint 位于 `outputs/gr00t_tactile_smoke/checkpoint-3`，报告在同目录上级的 `validation/`。

## 触觉布局与 jq_skin

已有数据训练从 `meta/modality.json` 读取触觉字段，不依赖 `jq_skin`。
默认按通道数固定分区排列，并把布局保存到 checkpoint，训练与推理使用相同布局。
实时传感器适配或导出身体布局时，从同级源码安装 SDK：

```bash
uv pip install --python .venv/bin/python -e ../jq-skin
python -c "from jq_skin.layout import BODY_PART_LAYOUTS, BodyPart; print(BODY_PART_LAYOUTS[BodyPart.UPPER_BODY].channel_count)"
```

导出与数据通道匹配的身体布局：

```bash
python -m gr00t_tactile.data.export_layout \
  --dataset gr00t_tactile/dataset/2026-09-24-17-17-20 \
  --output gr00t_tactile/configs/upper_body_layout.json
```

新训练配置设 `data.tactile_layout_path: configs/upper_body_layout.json` 即可使用。
当前只处理当帧触觉；模型重载需先调用 `register_tactile_checkpoint_classes()`。
原始代码来自 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)，许可证见 [LICENSE](LICENSE)。
