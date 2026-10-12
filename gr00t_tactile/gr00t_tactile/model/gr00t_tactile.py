"""GR00T N1.7 with post-VLM tactile context-token injection."""

from dataclasses import dataclass
from typing import Any

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from ..data.tactile_image import TactileLayout
from .tactile_encoder import TactileEncoderConfig, TactileResNet18Encoder


@dataclass
class Gr00tN1d7TactileConfig(Gr00tN1d7Config):
    """N1.7 configuration extended with the ResNet tactile baseline."""

    model_type: str = "Gr00tN1d7Tactile"
    tactile_input_channels: int = 1
    tactile_image_size: tuple[int, int] = (128, 128)
    tactile_parts: tuple[str, ...] = ("upper_body",)
    tactile_value_min: float = 0.0
    tactile_value_max: float = 255.0
    tactile_resnet_pretrained: bool = False
    tactile_token_type: str = "image"
    tactile_layout: dict | None = None

    def __init__(
        self,
        tactile_input_channels: int = 1,
        tactile_image_size: tuple[int, int] | list[int] = (128, 128),
        tactile_parts: tuple[str, ...] | list[str] = ("upper_body",),
        tactile_value_min: float = 0.0,
        tactile_value_max: float = 255.0,
        tactile_resnet_pretrained: bool = False,
        tactile_token_type: str = "image",
        tactile_layout: dict | None = None,
        **kwargs,
    ) -> None:
        kwargs["model_type"] = self.model_type
        super().__init__(**kwargs)
        self.tactile_input_channels = tactile_input_channels
        self.tactile_image_size = tuple(tactile_image_size)
        self.tactile_parts = tuple(tactile_parts)
        self.tactile_value_min = float(tactile_value_min)
        self.tactile_value_max = float(tactile_value_max)
        self.tactile_resnet_pretrained = tactile_resnet_pretrained
        self.tactile_token_type = tactile_token_type
        self.tactile_layout = tactile_layout
        if len(self.tactile_image_size) != 2 or min(self.tactile_image_size) <= 0:
            raise ValueError("tactile_image_size must contain two positive integers")
        if tactile_layout is not None:
            layout = TactileLayout.from_dict(tactile_layout)
            if (layout.height, layout.width) != self.tactile_image_size:
                raise ValueError("Pinned tactile layout size does not match tactile_image_size")
        if not self.tactile_parts:
            raise ValueError("tactile_parts must not be empty")
        if self.tactile_value_max <= self.tactile_value_min:
            raise ValueError("tactile_value_max must be greater than tactile_value_min")
        if tactile_token_type not in {"image", "non_image"}:
            raise ValueError("tactile_token_type must be 'image' or 'non_image'")
        if not self.use_alternate_vl_dit:
            raise ValueError(
                "The tactile baseline requires use_alternate_vl_dit=True so its token mask "
                "is honored by GR00T cross-attention"
            )


class Gr00tN1d7TactileActionHead(Gr00tN1d7ActionHead):
    """Allow the model wrapper to preprocess VL context exactly once."""

    _PREPROCESSED_KEY = "_tactile_vl_context_preprocessed"

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        if backbone_output.pop(self._PREPROCESSED_KEY, False):
            return backbone_output
        return super().process_backbone_output(backbone_output)


class Gr00tN1d7Tactile(Gr00tN1d7):
    """Append ResNet tactile tokens to the context consumed by the action DiT."""

    config_class = Gr00tN1d7TactileConfig

    def __init__(
        self,
        config: Gr00tN1d7TactileConfig,
        transformers_loading_kwargs: dict | None = None,
    ) -> None:
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}
        super().__init__(config, transformers_loading_kwargs=transformers_loading_kwargs)
        # This subclass adds behavior only and has the same module/state layout.
        # Reusing the constructed instance avoids allocating the large DiT twice.
        self.action_head.__class__ = Gr00tN1d7TactileActionHead
        self.tactile_encoder = TactileResNet18Encoder(
            TactileEncoderConfig(
                input_channels=config.tactile_input_channels,
                context_dim=config.backbone_embedding_dim,
                image_size=config.tactile_image_size,
                pretrained=config.tactile_resnet_pretrained,
            )
        )

    def _prepare_input_with_tactile(
        self, inputs: dict
    ) -> tuple[BatchFeature, BatchFeature, torch.Tensor | None, torch.Tensor | None]:
        # GR00T's prepare_input may pop ``vlm_content``; isolate that mutation.
        model_inputs = dict(inputs)
        tactile_values = model_inputs.pop("tactile_values", None)
        tactile_mask = model_inputs.pop("tactile_mask", None)
        backbone_inputs, action_inputs = super().prepare_input(model_inputs)
        return backbone_inputs, action_inputs, tactile_values, tactile_mask

    def _append_tactile_context(
        self,
        backbone_output: BatchFeature,
        tactile_values: torch.Tensor | None,
        tactile_mask: torch.Tensor | None,
    ) -> BatchFeature:
        vl_tokens = backbone_output["backbone_features"]
        batch_size = vl_tokens.shape[0]
        encoder_parameter = next(self.tactile_encoder.parameters())

        if tactile_values is None:
            height, width = self.config.tactile_image_size
            tactile_values = torch.zeros(
                batch_size,
                self.config.tactile_input_channels,
                height,
                width,
                device=encoder_parameter.device,
                dtype=encoder_parameter.dtype,
            )
            tactile_mask = torch.zeros(
                batch_size, 1, height, width, device=encoder_parameter.device
            )
        else:
            tactile_values = torch.as_tensor(
                tactile_values,
                device=encoder_parameter.device,
                dtype=encoder_parameter.dtype,
            )
            if tactile_mask is not None:
                tactile_mask = torch.as_tensor(tactile_mask, device=encoder_parameter.device)

        tactile_tokens, tactile_token_mask = self.tactile_encoder(
            tactile_values, tactile_mask
        )
        tactile_tokens = tactile_tokens.to(device=vl_tokens.device, dtype=vl_tokens.dtype)
        tactile_token_mask = tactile_token_mask.to(device=vl_tokens.device)

        fused = BatchFeature(data=dict(backbone_output))
        fused["backbone_features"] = torch.cat((vl_tokens, tactile_tokens), dim=1)
        backbone_mask = backbone_output["backbone_attention_mask"]
        fused["backbone_attention_mask"] = torch.cat(
            (backbone_mask, tactile_token_mask.to(dtype=backbone_mask.dtype)), dim=1
        )

        # AlternateVLDiT routes image and non-image tokens through alternating
        # cross-attention blocks. Tactile is spatial, so the baseline groups it
        # with image tokens by default.
        if "image_mask" in backbone_output:
            image_mask = backbone_output["image_mask"]
            tactile_is_image = self.config.tactile_token_type == "image"
            tactile_type_mask = torch.full(
                tactile_token_mask.shape,
                tactile_is_image,
                dtype=image_mask.dtype,
                device=image_mask.device,
            )
            fused["image_mask"] = torch.cat((image_mask, tactile_type_mask), dim=1)
        return fused

    def _prepare_action_context(
        self,
        backbone_output: BatchFeature,
        tactile_values: torch.Tensor | None,
        tactile_mask: torch.Tensor | None,
    ) -> BatchFeature:
        # The checkpoint uses four VL self-attention layers without a padding
        # mask. Process original VL tokens first so missing tactile tokens cannot
        # leak into valid visual/language tokens before the DiT mask is applied.
        backbone_output = self.action_head.process_backbone_output(backbone_output)
        backbone_output = self._append_tactile_context(
            backbone_output, tactile_values, tactile_mask
        )
        backbone_output[self.action_head._PREPROCESSED_KEY] = True
        return backbone_output

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs, tactile_values, tactile_mask = (
            self._prepare_input_with_tactile(inputs)
        )
        backbone_output = self.backbone(backbone_inputs)
        backbone_output = self._prepare_action_context(
            backbone_output, tactile_values, tactile_mask
        )
        return self.action_head(backbone_output, action_inputs)

    def get_action(
        self, inputs: dict, options: dict[str, Any] | None = None
    ) -> BatchFeature:
        backbone_inputs, action_inputs, tactile_values, tactile_mask = (
            self._prepare_input_with_tactile(inputs)
        )
        backbone_output = self.backbone(backbone_inputs)
        backbone_output = self._prepare_action_context(
            backbone_output, tactile_values, tactile_mask
        )
        return self.action_head.get_action(backbone_output, action_inputs, options)


AutoConfig.register(Gr00tN1d7TactileConfig.model_type, Gr00tN1d7TactileConfig)
AutoModel.register(Gr00tN1d7TactileConfig, Gr00tN1d7Tactile)
