"""Regression checks for loading tactile checkpoints and missing live sensors."""

from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t_tactile.data.tactile_image import TactileImageProcessor, TactileLayout
from gr00t_tactile.inference.tactile_observation import add_tactile_state
from gr00t_tactile.model.gr00t_tactile import Gr00tN1d7Tactile, Gr00tN1d7TactileConfig
import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "processor_config"


def test_offline_vlm_processor_uses_resolved_snapshot(tmp_path):
    snapshot = tmp_path / "snapshots" / "revision" / "processor"
    loading_kwargs = {"local_files_only": True, "revision": "revision", "subfolder": "processor"}
    with (
        patch.object(
            processing, "cached_file", return_value=str(snapshot / "config.json")
        ) as cached,
        patch.object(processing.Qwen3VLProcessor, "from_pretrained") as load,
    ):
        processing.build_processor("nvidia/Cosmos-Reason2-2B", loading_kwargs)
    cached.assert_called_once_with("nvidia/Cosmos-Reason2-2B", "config.json", **loading_kwargs)
    load.assert_called_once_with(str(snapshot), local_files_only=True, revision="revision")
    assert loading_kwargs["subfolder"] == "processor"


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(7, 64)

    def prepare_input(self, inputs):
        return BatchFeature(data={"state": inputs["state"]})

    def forward(self, inputs):
        features = self.projection(inputs["state"]).repeat(1, 8, 1)
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        image_mask = mask.clone()
        image_mask[:, :4] = False
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": mask,
                "image_mask": image_mask,
            }
        )


@pytest.fixture
def local_model_loading(load_hf_model_weights):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with (
        load_hf_model_weights(),
        patch(
            "gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls",
            return_value=lambda **kwargs: SmallBackbone(),
        ),
        patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"),
    ):
        yield
    torch.set_num_threads(previous_threads)


def small_config():
    return Gr00tN1d7TactileConfig(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        max_seq_len=32,
        use_alternate_vl_dit=True,
        use_flash_attention=False,
        load_bf16=False,
        tune_llm=False,
        tune_visual=False,
        state_dropout_prob=0.0,
        vl_self_attention_cfg={"num_layers": 0},
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )


def test_base_checkpoint_initializes_tactile_and_backpropagates(tmp_path, local_model_loading):
    config = small_config()
    model = Gr00tN1d7Tactile(config)
    config.save_pretrained(tmp_path)
    save_file(
        {key: value for key, value in model.state_dict().items() if not key.startswith("tactile_")},
        tmp_path / "model.safetensors",
    )
    del model
    loaded = Gr00tN1d7Tactile.from_pretrained(tmp_path, local_files_only=True)
    for embedding in (
        loaded.tactile_encoder.modality_embedding,
        loaded.tactile_encoder.position_embedding,
    ):
        assert torch.isfinite(embedding).all()
        assert 0.005 < embedding.detach().std() < 0.05
        assert embedding.detach().abs().max() < 0.2

    loaded.train()
    loss = loaded(
        {
            "state": torch.randn(2, 1, 7),
            "action": torch.randn(2, 4, 7),
            "action_mask": torch.ones(2, 4, 7),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
            "tactile_values": torch.rand(2, 1, 128, 128),
            "tactile_mask": torch.ones(2, 1, 128, 128),
        }
    )["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in loaded.tactile_encoder.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert loaded.tactile_encoder.position_embedding.grad.abs().sum() > 0


def test_tactile_checkpoint_preserves_trained_embeddings(tmp_path, local_model_loading):
    model = Gr00tN1d7Tactile(small_config())
    with torch.no_grad():
        model.tactile_encoder.modality_embedding.fill_(0.125)
        model.tactile_encoder.position_embedding.fill_(-0.25)
    model.config.save_pretrained(tmp_path)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    loaded = Gr00tN1d7Tactile.from_pretrained(tmp_path, local_files_only=True)
    assert torch.equal(
        loaded.tactile_encoder.modality_embedding, model.tactile_encoder.modality_embedding
    )
    assert torch.equal(
        loaded.tactile_encoder.position_embedding, model.tactile_encoder.position_embedding
    )


@pytest.fixture
def mixed_processor_checkpoint(tmp_path):
    for source in FIXTURE_DIR.iterdir():
        if source.is_file():
            shutil.copy2(source, tmp_path / source.name)
    config_path = tmp_path / "processor_config.json"
    config = json.loads(config_path.read_text())
    configs = config["processor_kwargs"]["modality_configs"]
    configs["unused_robot"] = deepcopy(configs["libero_sim"])
    configs["unused_robot"]["action"]["delta_indices"] = list(range(50))
    config_path.write_text(json.dumps(config))
    stats_path = tmp_path / "statistics.json"
    stats = json.loads(stats_path.read_text())
    stats["unused_robot"] = deepcopy(stats["libero_sim"])
    stats_path.write_text(json.dumps(stats))
    return tmp_path


def test_active_embodiment_scopes_configs_and_statistics(mixed_processor_checkpoint):
    with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"):
        processor = Gr00tN1d7Processor.from_pretrained(
            mixed_processor_checkpoint,
            active_embodiment_tags=["libero_sim"],
            max_action_horizon=40,
        )
    assert set(processor.modality_configs) == {"libero_sim"}
    assert set(processor.state_action_processor.statistics) == {"libero_sim"}


@pytest.mark.parametrize("active_tags", [None, ["unused_robot"]])
def test_scope_keeps_active_horizon_validation(mixed_processor_checkpoint, active_tags):
    with (
        patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"),
        pytest.raises(ValueError, match="unused_robot=50"),
    ):
        Gr00tN1d7Processor.from_pretrained(
            mixed_processor_checkpoint, active_embodiment_tags=active_tags, max_action_horizon=40
        )


@pytest.mark.parametrize("active_tags", [[], ["missing_robot"]])
def test_invalid_active_embodiments_fail(mixed_processor_checkpoint, active_tags):
    with pytest.raises(ValueError, match="active"):
        Gr00tN1d7Processor.from_pretrained(
            mixed_processor_checkpoint, active_embodiment_tags=active_tags
        )


@pytest.mark.parametrize(
    "message,expected_valid_pixels", [(None, 0), ({"upper_body": {"chest": [0] * 4}}, 32)]
)
def test_live_missing_zones_are_masked(monkeypatch, message, expected_valid_pixels):
    layout_module = ModuleType("jq_skin.layout")
    body_part = Enum("BodyPart", ["UPPER_BODY"])
    layout_module.BodyPart = body_part
    layout_module.BODY_PART_LAYOUTS = {
        body_part.UPPER_BODY: SimpleNamespace(
            zones=[SimpleNamespace(name=name, channel_count=4) for name in ("chest", "back")]
        )
    }
    monkeypatch.setitem(sys.modules, "jq_skin", ModuleType("jq_skin"))
    monkeypatch.setitem(sys.modules, "jq_skin.layout", layout_module)
    observation = add_tactile_state({}, message)
    readings = {key: value[0] for key, value in observation.items()}
    layout = TactileLayout.packed_from_readings(readings, output_size=(8, 8))
    values, mask = TactileImageProcessor(layout)(readings)
    assert np.isfinite(values).all()
    assert np.count_nonzero(values) == 0
    assert mask.sum() == expected_valid_pixels
