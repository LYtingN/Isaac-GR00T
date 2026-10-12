"""Isaac-GR00T training pipeline for the tactile model."""

import json
import logging
from pathlib import Path

from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline, convert_tensors_to_lists
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import run_or_wait_on_rank0

from ..data.gr00t_tactile_processor import Gr00tN1d7TactileProcessor
from ..model.gr00t_tactile import Gr00tN1d7Tactile, Gr00tN1d7TactileConfig


class Gr00tN1d7TactilePipeline(Gr00tN1d7Pipeline):
    model_class = Gr00tN1d7Tactile
    processor_class = Gr00tN1d7TactileProcessor

    def _create_model(self):
        checkpoint = self.config.training.start_from_checkpoint
        skip_weights = getattr(self.config.training, "skip_weight_loading", False)
        if checkpoint is not None and not skip_weights:
            checkpoint_config = Gr00tN1d7TactileConfig.from_pretrained(
                checkpoint, **self.transformers_loading_kwargs
            )
            # Training flags and tactile settings come from this run, while all
            # architecture details not listed here remain checkpoint-derived.
            override_fields = (
                "tune_llm",
                "tune_visual",
                "tune_projector",
                "tune_diffusion_model",
                "tune_vlln",
                "state_dropout_prob",
                "backbone_trainable_params_fp32",
                "load_bf16",
                "tactile_input_channels",
                "tactile_image_size",
                "tactile_parts",
                "tactile_value_min",
                "tactile_value_max",
                "tactile_resnet_pretrained",
                "tactile_token_type",
            )
            for field in override_fields:
                setattr(checkpoint_config, field, getattr(self.model_config, field))
            if self.model_config.tactile_layout is not None:
                checkpoint_config.tactile_layout = self.model_config.tactile_layout
            model, loading_info = self.model_class.from_pretrained(
                checkpoint,
                config=checkpoint_config,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )
            missing = [
                key
                for key in loading_info.get("missing_keys", [])
                if not key.startswith("tactile_encoder.")
            ]
            unexpected = loading_info.get("unexpected_keys", [])
            mismatched = loading_info.get("mismatched_keys", [])
            if missing or unexpected or mismatched:
                raise RuntimeError(
                    "Checkpoint mismatch outside the new tactile encoder: "
                    f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
                )
            logging.info(
                "Initialized tactile encoder; loaded all GR00T weights from %s", checkpoint
            )
        else:
            model = self.model_class(
                self.model_config,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                (self.save_cfg_dir / "final_model_config.json").write_text(
                    model.config.to_filtered_json()
                )
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        logging.info("Model parameters: %s total, %s trainable", f"{total:,}", f"{trainable:,}")
        return model

    def _create_dataset(self, save_cfg_dir):
        checkpoint = self.config.training.start_from_checkpoint
        if checkpoint is None:
            raise ValueError("Tactile fine-tuning requires a base_model_path")

        processor = self.processor_class.from_pretrained(
            checkpoint,
            active_embodiment_tags=[spec.embodiment_tag for spec in self.config.data.datasets],
            transformers_loading_kwargs=self.transformers_loading_kwargs,
            modality_configs=self.config.data.modality_configs,
            use_percentiles=self.model_config.use_percentiles,
            image_crop_size=self.model_config.image_crop_size,
            image_target_size=self.model_config.image_target_size,
            random_rotation_angle=self.model_config.random_rotation_angle,
            color_jitter_params=self.model_config.color_jitter_params,
            model_name=self.model_config.model_name,
            model_type=self.model_config.backbone_model_type,
            formalize_language=self.model_config.formalize_language,
            max_action_horizon=self.model_config.action_horizon,
            use_albumentations=self.model_config.use_albumentations_transforms,
            extra_augmentation_config=self.model_config.extra_augmentation_config,
            shortest_image_edge=self.model_config.shortest_image_edge,
            crop_fraction=self.model_config.crop_fraction,
            letter_box_transform=self.model_config.letter_box_transform,
            use_relative_action=self.model_config.use_relative_action,
            exclude_state=self.model_config.exclude_state,
            state_dropout_prob=self.model_config.state_dropout_prob,
            use_mean_std=self.model_config.use_mean_std,
            **self.transformers_loading_kwargs,
        )
        processor.configure_tactile(
            self.model_config.tactile_parts,
            self.model_config.tactile_image_size,
            self.model_config.tactile_value_min,
            self.model_config.tactile_value_max,
            self.model_config.tactile_layout,
        )
        for dataset_spec in self.config.data.datasets:
            for dataset_path in dataset_spec.dataset_paths:
                processor.prepare_layout_from_dataset_metadata(Path(dataset_path))
        self.processor = processor
        train_dataset, eval_dataset = DatasetFactory(config=self.config).build(processor=processor)
        with run_or_wait_on_rank0(label="dataset_statistics.json write") as is_rank0:
            if is_rank0:
                statistics = convert_tensors_to_lists(train_dataset.get_dataset_statistics())
                (save_cfg_dir / "dataset_statistics.json").write_text(
                    json.dumps(statistics, indent=2)
                )
        return train_dataset, eval_dataset


register_model(Gr00tN1d7TactileConfig, Gr00tN1d7TactilePipeline)
