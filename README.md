# GR00T Tactile

Tactile-conditioned fine-tuning for [Isaac-GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T).
A ResNet-18 encodes a 128×128 tactile map into 16 spatial tokens, appended to the
vision-language context consumed by the action DiT. By default, the vision and
language backbone is frozen while the tactile encoder and action head are trained.

## Setup

Requires Python 3.12, CUDA 12.8, an NVIDIA GPU, and FFmpeg 4–7 for video decoding.
The tactile package targets the root environment; separate Jetson/Spark stacks are not supported.
Run all commands from the repository root. Request access to
[Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) before downloading.

```bash
uv sync --python 3.12
source .venv/bin/activate
hf auth login
hf download nvidia/GR00T-N1.7-3B
hf download nvidia/Cosmos-Reason2-2B
```

For project-local CUDA/FFmpeg libraries, use `source gr00t_tactile/activate_env.sh`.

## Training

Datasets are not included. Supply a LeRobot dataset compatible with `unitree_g1_sonic`,
including `tactile_*` state fields in `meta/modality.json`. Set `data.dataset_path`
in the configurations below; adjust batch size and training steps for your GPU.

```bash
# Three-step check, batch size 1
gr00t-tactile-finetune --config gr00t_tactile/configs/smoke.yaml

# Fine-tuning
gr00t-tactile-finetune --config gr00t_tactile/configs/train.yaml
```

With both models cached, prefix either command with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for offline training.
The smoke checkpoint is saved to `outputs/gr00t_tactile_smoke/checkpoint-3`.

## Scope and validation

The baseline uses the current tactile frame and a fixed packed layout saved with
the processor. Training from recorded data does not require `jq_skin`; live sensor
adaptation and anatomical layout export do. See the [package guide](gr00t_tactile/README.md).

Three-step training, checkpoint reload, and real-data inference were verified on a
single 48 GB GPU. Long-run convergence and hardware deployment remain unverified.
Run regression tests with `uv run --extra dev pytest tests/gr00t_tactile -q`.

## License

Based on NVIDIA Isaac-GR00T. Code is licensed under [Apache-2.0](LICENSE);
pretrained models and datasets retain their respective licenses.
