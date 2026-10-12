"""ResNet tactile encoder used by the first GR00T tactile baseline."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class TactileEncoderConfig:
    input_channels: int = 1
    context_dim: int = 2048
    image_size: tuple[int, int] = (128, 128)
    pretrained: bool = False


class TactileResNet18Encoder(nn.Module):
    """Encode a tactile heatmap into spatial tokens for GR00T cross-attention.

    The classifier and global pooling are removed. Each cell in the final
    feature map becomes one token, retaining coarse contact location. A
    128x128 input produces 16 tokens (a 4x4 feature map).
    """

    def __init__(self, config: TactileEncoderConfig) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if config.pretrained else None
        backbone = resnet18(weights=weights)

        if config.input_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                config.input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if weights is not None:
                with torch.no_grad():
                    rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(rgb_mean.repeat(1, config.input_channels, 1, 1))
            backbone.conv1 = new_conv

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.projection = nn.Linear(512, config.context_dim)
        self.layer_norm = nn.LayerNorm(config.context_dim)
        self.modality_embedding = nn.Parameter(torch.empty(1, 1, config.context_dim))
        feature_height = (config.image_size[0] + 31) // 32
        feature_width = (config.image_size[1] + 31) // 32
        self.feature_shape = (feature_height, feature_width)
        self.position_embedding = nn.Parameter(
            torch.empty(1, feature_height * feature_width, config.context_dim)
        )
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        tactile_values: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tactile_values.ndim != 4:
            raise ValueError(
                f"tactile_values must have shape [B, C, H, W], got {tuple(tactile_values.shape)}"
            )
        if tactile_values.shape[1] != self.features[0].in_channels:
            raise ValueError(
                f"Expected {self.features[0].in_channels} tactile channels, "
                f"got {tactile_values.shape[1]}"
            )
        feature_map = self.features(tactile_values)
        if feature_map.shape[-2:] != self.feature_shape:
            raise ValueError(
                f"Expected tactile feature map {self.feature_shape}, "
                f"got {tuple(feature_map.shape[-2:])}"
            )
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = (
            self.layer_norm(self.projection(tokens))
            + self.modality_embedding
            + self.position_embedding
        )

        if tactile_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if (
                tactile_mask.ndim != 4
                or tactile_mask.shape[:2] != (tactile_values.shape[0], 1)
                or tactile_mask.shape[-2:] != tactile_values.shape[-2:]
            ):
                raise ValueError("tactile_mask must have shape [B, 1, H, W]")
            pooled_mask = F.adaptive_max_pool2d(
                tactile_mask.to(dtype=torch.float32), feature_map.shape[-2:]
            )
            token_mask = pooled_mask.flatten(1) > 0
        return tokens, token_mask
