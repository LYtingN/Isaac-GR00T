"""Launch the ResNet + cross-attention tactile baseline from a YAML config."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run
import yaml

from ..model.gr00t_tactile import Gr00tN1d7TactileConfig
from . import pipeline as _pipeline  # noqa: F401  (registers the custom pipeline)


def _load_python_config(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gr00t_tactile_modality_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(settings: dict, config_path: Path):
    model_settings = settings.get("model", {})
    data_settings = settings.get("data", {})
    training_settings = settings.get("training", {})

    modality_path = Path(
        data_settings.get("modality_config_path", "configs/sonic_tactile_modality.py")
    )
    if not modality_path.is_absolute():
        modality_path = config_path.parent.parent / modality_path
    dataset_path = data_settings.get("dataset_path")
    if not dataset_path:
        raise ValueError("Set data.dataset_path in the training config")
    dataset_paths = [path for path in str(dataset_path).split(os.pathsep) if path]
    modality_module = _load_python_config(modality_path.resolve())
    tactile_parts = data_settings.get("tactile_parts", ["upper_body"])
    tactile_layout_path = data_settings.get("tactile_layout_path")
    tactile_layout = None
    if tactile_layout_path:
        layout_path = Path(tactile_layout_path)
        if not layout_path.is_absolute():
            layout_path = config_path.parent.parent / layout_path
        tactile_layout = json.loads(layout_path.read_text())
    tactile_keys = data_settings.get("tactile_keys") or _read_tactile_keys(dataset_paths[0])
    if tactile_keys:
        modality_module.register_tactile_keys(tactile_keys)
    else:
        modality_module.register_tactile_parts(tactile_parts)
    embodiment = data_settings.get("embodiment_tag", "unitree_g1_sonic")
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment,
                    }
                ],
            }
        }
    )

    config.model = Gr00tN1d7TactileConfig(
        tune_llm=bool(model_settings.get("tune_llm", False)),
        tune_visual=bool(model_settings.get("tune_visual", False)),
        tune_projector=bool(model_settings.get("tune_projector", True)),
        tune_diffusion_model=bool(model_settings.get("tune_diffusion_model", True)),
        use_relative_action=True,
        load_bf16=False,
        reproject_vision=False,
        tactile_input_channels=1,
        tactile_image_size=model_settings.get("tactile_image_size", [128, 128]),
        tactile_parts=tactile_parts,
        tactile_value_min=float(data_settings.get("tactile_value_min", 0.0)),
        tactile_value_max=float(data_settings.get("tactile_value_max", 255.0)),
        tactile_resnet_pretrained=bool(model_settings.get("resnet_pretrained", False)),
        tactile_token_type=model_settings.get("tactile_token_type", "image"),
        tactile_layout=tactile_layout,
    )
    config.training.start_from_checkpoint = model_settings.get(
        "base_model_path", "nvidia/GR00T-N1.7-3B"
    )
    config.training.output_dir = training_settings.get("output_dir", "outputs/gr00t_tactile")
    config.training.max_steps = int(training_settings.get("max_steps", 20_000))
    config.training.global_batch_size = int(training_settings.get("global_batch_size", 32))
    config.training.learning_rate = float(training_settings.get("learning_rate", 1e-4))
    config.training.num_gpus = int(training_settings.get("num_gpus", 1))
    config.training.dataloader_num_workers = int(training_settings.get("num_workers", 2))
    config.training.use_wandb = bool(training_settings.get("use_wandb", False))
    config.training.save_steps = int(training_settings.get("save_steps", 1000))
    config.training.save_only_model = bool(training_settings.get("save_only_model", False))
    config.training.logging_steps = int(training_settings.get("logging_steps", 10))
    config.data.shard_size = int(data_settings.get("shard_size", 1024))
    config.data.episode_sampling_rate = float(data_settings.get("episode_sampling_rate", 0.1))
    return config


def _read_tactile_keys(dataset_path: str) -> list[str]:
    """Read logical tactile state keys from a collected LeRobot dataset."""

    modality_path = Path(dataset_path) / "meta" / "modality.json"
    if not modality_path.exists():
        return []
    modality = json.loads(modality_path.read_text())
    return sorted(key for key in modality.get("state", {}) if key.startswith("tactile_"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    run(build_config(settings, args.config.resolve()))


if __name__ == "__main__":
    main()
