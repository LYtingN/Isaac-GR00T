"""Register SONIC state fields plus jq_skin tactile zones for GR00T loading."""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ModalityConfig


EMBODIMENT = "unitree_g1_sonic"
TACTILE_PARTS = ("upper_body",)


def register_tactile_keys(tactile_keys) -> None:
    """Add explicit dataset tactile keys to the SONIC loader configuration."""

    current = deepcopy(MODALITY_CONFIGS[EMBODIMENT])
    state = current["state"]
    current["state"] = ModalityConfig(
        delta_indices=list(state.delta_indices),
        modality_keys=[
            *[key for key in state.modality_keys if not key.startswith("tactile_")],
            *sorted(set(tactile_keys)),
        ],
        sin_cos_embedding_keys=state.sin_cos_embedding_keys,
        mean_std_embedding_keys=state.mean_std_embedding_keys,
    )
    MODALITY_CONFIGS[EMBODIMENT] = current


def register_tactile_parts(parts=TACTILE_PARTS) -> None:
    """Add the requested jq_skin zones to the SONIC loader configuration."""

    from jq_skin.layout import BODY_PART_LAYOUTS, BodyPart

    tactile_keys = []
    for part_name in parts:
        canonical_part = part_name.lower()
        part = BodyPart[canonical_part.upper()]
        tactile_keys.extend(
            f"tactile_{canonical_part}_{zone.name}" for zone in BODY_PART_LAYOUTS[part].zones
        )

    register_tactile_keys(tactile_keys)
