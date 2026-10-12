# GR00T Tactile

First baseline for tactile-conditioned GR00T N1.7:

1. Tactile zones are rasterized into one fixed 128x128 image using a pinned layout.
2. ResNet-18 converts the image into 16 spatial tokens.
3. The original vision/language tokens first pass through GR00T's VLLN and VL self-attention.
4. Tactile tokens are then appended as image-context tokens and consumed by AlternateVLDiT's
   masked cross-attention.

```text
gr00t_tactile/
├── configs/
│   ├── train.yaml
│   └── sonic_tactile_modality.py
├── gr00t_tactile/
│   ├── data/
│   │   ├── tactile_image.py
│   │   ├── gr00t_tactile_processor.py
│   │   └── export_layout.py
│   ├── model/
│   │   ├── tactile_encoder.py
│   │   └── gr00t_tactile.py
│   ├── training/
│   │   ├── finetune.py
│   │   └── pipeline.py
│   └── inference/
│       ├── tactile_observation.py
│       └── registration.py
```

Responsibilities:

- `data/tactile_image.py`: fixed-range normalization and layout rasterization.
- `data/gr00t_tactile_processor.py`: split tactile zones from GR00T's flat robot state.
- `data/export_layout.py`: export a fixed anatomical layout from the collection environment.
- `model/tactile_encoder.py`: ResNet-18 spatial token encoder.
- `model/gr00t_tactile.py`: post-VLM context injection.
- `training/pipeline.py`: custom GR00T model/processor loading.
- `training/finetune.py`: YAML training entry point.
- `inference/tactile_observation.py`: build online observations with tactile input.
- `inference/registration.py`: register custom classes before loading a checkpoint.

Robot-side sensor acquisition remains in `gear_sonic/tactile/`.

The bundled `configs/train.yaml` points to the migrated four-episode dataset.
Set `data.dataset_path` to another dataset before training on different data. To use the anatomical layout,
run this from the repository root in the collection environment where `jq_skin`
is installed, then set `data.tactile_layout_path` to
`configs/upper_body_layout.json`:

```bash
PYTHONPATH=gr00t_tactile python -m gr00t_tactile.data.export_layout \
  --dataset gr00t_tactile/dataset/2026-09-24-17-17-20 \
  --output gr00t_tactile/configs/upper_body_layout.json
```

Set up the shared GR00T N1.7 and tactile environment from the Isaac-GR00T root:

```bash
uv sync --python 3.12
source gr00t_tactile/activate_env.sh
python -c "import gr00t, gr00t_tactile; print('GR00T tactile installed')"
```

Train from the repository root:

```bash
source gr00t_tactile/activate_env.sh
gr00t-tactile-finetune --config gr00t_tactile/configs/train.yaml
```

The shared `.venv` uses Python 3.12, Torch 2.9.0, torchvision 0.24.0, and CUDA 12.8.
`activate_env.sh` also selects the NVIDIA 570 userspace libraries matching this workstation's
loaded kernel driver and exposes the project-local FFmpeg 7 runtime required by TorchCodec.
Without `data.tactile_layout_path`, training deterministically packs zones using
their recorded channel counts. In either mode, the processor validates the layout
against the dataset and saves the exact layout in `processor_config.json`.
Inference loads that saved layout, independent of whether `jq_skin` is installed.
Tactile checkpoints serialize `model_type: Gr00tN1d7Tactile`. The identifier
matches the earlier tactile checkpoint format, so this naming correction does
not require checkpoint migration.
The packed layout is not an anatomical map. Live sensor acquisition still
requires `jq_skin` through `gear_sonic/tactile/`. Checkpoints saved before
this layout was pinned need a validated processor layout before inference.

This baseline uses the current tactile frame only. Temporal tactile memory is the next model
extension and does not change the dataset schema introduced here.

The tactile package is installed as an editable local dependency by the root
`pyproject.toml`; its standalone project does not fetch a second GR00T checkout.
Model loading requires access to the gated `nvidia/Cosmos-Reason2-2B` on Hugging Face.
