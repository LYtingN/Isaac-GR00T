# Tactile package

See the [project README](../README.md) for installation and training.
Implementation lives in `gr00t_tactile/data/`, `model/`, `training/`, and `inference/`.

## Optional sensor layout

Install `jq_skin` from your SDK source checkout to use the live observation adapter
or export an anatomical layout. Recorded-data training uses dataset metadata.

```bash
uv pip install --python .venv/bin/python /path/to/jq-skin
python -m gr00t_tactile.data.export_layout \
  --dataset /path/to/lerobot-dataset \
  --output gr00t_tactile/configs/upper_body_layout.json
```

Set `data.tactile_layout_path: configs/upper_body_layout.json` for a new training run.
The exporter checks zone channel counts against the dataset; inference reuses the
layout saved in the checkpoint. Before loading a tactile model, call
`gr00t_tactile.inference.registration.register_tactile_checkpoint_classes()`.
