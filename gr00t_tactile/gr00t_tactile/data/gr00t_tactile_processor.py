"""GR00T processor adapter that separates tactile data from robot state."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
import torch
from transformers import AutoProcessor

from .tactile_image import TactileImageProcessor, TactileLayout


class Gr00tN1d7TactileProcessor(Gr00tN1d7Processor):
    """Remove ``tactile_*`` fields from proprioception and rasterize them."""

    def __init__(
        self,
        *args,
        tactile_parts: list[str] | tuple[str, ...] = ("upper_body",),
        tactile_image_size: list[int] | tuple[int, int] = (128, 128),
        tactile_value_min: float = 0.0,
        tactile_value_max: float = 255.0,
        tactile_state_prefix: str = "tactile_",
        tactile_layout: dict | None = None,
        **kwargs,
    ) -> None:
        modality_configs = kwargs.get("modality_configs")
        if modality_configs is None and args:
            modality_configs = args[0]
        if modality_configs is None:
            raise ValueError("modality_configs is required")

        self.source_modality_configs = deepcopy(modality_configs)
        stripped_configs = self._without_tactile_state(modality_configs, tactile_state_prefix)
        if args:
            args = (stripped_configs, *args[1:])
        else:
            kwargs["modality_configs"] = stripped_configs

        self.tactile_parts = tuple(part.lower() for part in tactile_parts)
        self.tactile_image_size = tuple(int(v) for v in tactile_image_size)
        self.tactile_value_min = float(tactile_value_min)
        self.tactile_value_max = float(tactile_value_max)
        self.tactile_state_prefix = tactile_state_prefix
        self._tactile_layout_data = tactile_layout
        self.configure_tactile(
            self.tactile_parts,
            self.tactile_image_size,
            self.tactile_value_min,
            self.tactile_value_max,
        )
        super().__init__(*args, **kwargs)

    def configure_tactile(
        self,
        parts: list[str] | tuple[str, ...],
        image_size: list[int] | tuple[int, int],
        value_min: float,
        value_max: float,
        tactile_layout: dict | None = None,
    ) -> None:
        """Set the preprocessing contract shared by training and inference."""

        self.tactile_parts = tuple(part.lower() for part in parts)
        self.tactile_image_size = tuple(int(value) for value in image_size)
        self.tactile_value_min = float(value_min)
        self.tactile_value_max = float(value_max)
        if tactile_layout is not None:
            self._tactile_layout_data = tactile_layout
        layout = (
            TactileLayout.from_dict(self._tactile_layout_data)
            if self._tactile_layout_data is not None
            else None
        )
        if layout is not None and (layout.height, layout.width) != self.tactile_image_size:
            raise ValueError("Pinned tactile layout size does not match tactile_image_size")
        self.tactile_processor = self._make_processor(layout) if layout is not None else None

    def _make_processor(self, layout: TactileLayout) -> TactileImageProcessor:
        return TactileImageProcessor(
            layout, value_min=self.tactile_value_min, value_max=self.tactile_value_max
        )

    def _ensure_tactile_processor(self, readings: dict) -> TactileImageProcessor:
        if self.tactile_processor is None:
            layout = TactileLayout.packed_from_readings(readings, self.tactile_image_size)
            self.tactile_processor = self._make_processor(layout)
            self._tactile_layout_data = {
                "source": {"kind": "packed_from_dataset"},
                "layout": asdict(layout),
            }
        return self.tactile_processor

    def prepare_layout_from_dataset_metadata(self, dataset_path: str | Path) -> None:
        """Resolve a fallback layout and check every pinned zone against a dataset."""

        metadata_path = Path(dataset_path) / "meta" / "modality.json"
        metadata = json.loads(metadata_path.read_text())
        tactile_fields = {
            key: spec["end"] - spec["start"]
            for key, spec in metadata.get("state", {}).items()
            if key.startswith(self.tactile_state_prefix)
        }
        if not tactile_fields:
            raise ValueError(f"No tactile state fields in {metadata_path}")
        readings = {
            key: np.zeros(count, dtype=np.float32) for key, count in tactile_fields.items()
        }
        layout = self._ensure_tactile_processor(readings).layout
        expected = {zone.key: int(np.prod(zone.grid_shape)) for zone in layout.zones}
        if tactile_fields != expected:
            raise ValueError(
                f"Tactile layout does not match {metadata_path}: "
                f"dataset={tactile_fields}, layout={expected}"
            )

    @staticmethod
    def _without_tactile_state(modality_configs: dict, prefix: str) -> dict:
        result = deepcopy(modality_configs)
        for embodiment_config in result.values():
            state_config = embodiment_config.get("state")
            if state_config is None:
                continue
            if isinstance(state_config, dict):
                keys = state_config["modality_keys"]
                state_config["modality_keys"] = [key for key in keys if not key.startswith(prefix)]
            else:
                keys = state_config.modality_keys
                embodiment_config["state"] = ModalityConfig(
                    delta_indices=list(state_config.delta_indices),
                    modality_keys=[key for key in keys if not key.startswith(prefix)],
                    sin_cos_embedding_keys=state_config.sin_cos_embedding_keys,
                    mean_std_embedding_keys=state_config.mean_std_embedding_keys,
                    action_configs=state_config.action_configs,
                )
        return result

    def __call__(self, messages: list[dict[str, Any]]):
        content = messages[0]["content"]
        processor = self._ensure_tactile_processor(content.states)
        tactile_values, tactile_mask = processor(content.states)
        transformed = super().__call__(messages)
        transformed["tactile_values"] = tactile_values
        transformed["tactile_mask"] = tactile_mask
        return transformed

    def process_observation(self, observation: dict[str, Any], embodiment_tag):
        if self.tactile_processor is None:
            raise ValueError(
                "No tactile layout is pinned in this processor checkpoint; "
                "re-save it with a validated layout before inference"
            )
        transformed = super().process_observation(observation, embodiment_tag)
        batch_size = transformed["state"].shape[0]
        values, masks = [], []
        for index in range(batch_size):
            sample = {}
            for key in self.tactile_processor.zone_keys:
                value = observation.get(f"state.{key}")
                if value is not None:
                    sample[key] = value[index]
            sample_values, sample_mask = self.tactile_processor(sample)
            values.append(sample_values)
            masks.append(sample_mask)
        transformed["tactile_values"] = torch.from_numpy(np.stack(values))
        transformed["tactile_mask"] = torch.from_numpy(np.stack(masks))
        return transformed

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        if self.tactile_processor is None:
            raise ValueError("Resolve the tactile layout before saving the processor")
        paths = super().save_pretrained(save_directory)
        config_path = Path(save_directory) / "processor_config.json"
        config = json.loads(config_path.read_text())
        config["processor_class"] = self.__class__.__name__
        config["processor_kwargs"]["modality_configs"] = to_json_serializable(
            self.source_modality_configs
        )
        config["processor_kwargs"].update(
            {
                "tactile_parts": list(self.tactile_parts),
                "tactile_image_size": list(self.tactile_image_size),
                "tactile_value_min": self.tactile_value_min,
                "tactile_value_max": self.tactile_value_max,
                "tactile_state_prefix": self.tactile_state_prefix,
                "tactile_layout": self._tactile_layout_data
                or asdict(self.tactile_processor.layout),
            }
        )
        config_path.write_text(json.dumps(config, indent=2))
        return paths


AutoProcessor.register("Gr00tN1d7Tactile", Gr00tN1d7TactileProcessor)
