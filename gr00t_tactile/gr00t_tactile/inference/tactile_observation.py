"""Add a tactile-server message to a batched GR00T observation."""

from copy import deepcopy

import numpy as np


def add_tactile_state(
    observation: dict,
    tactile_message: dict | None,
    parts: tuple[str, ...] = ("upper_body",),
) -> dict:
    """Return an observation with ``state.tactile_*`` arrays of shape ``[1, 1, D]``."""

    from jq_skin.layout import BODY_PART_LAYOUTS, BodyPart

    result = deepcopy(observation)
    for part_name in parts:
        part = BodyPart[part_name.upper()]
        part_message = tactile_message.get(part_name) if tactile_message else None
        for zone in BODY_PART_LAYOUTS[part].zones:
            values = part_message.get(zone.name) if part_message else None
            if values is None:
                vector = np.zeros(zone.channel_count, dtype=np.float32)
            else:
                vector = np.asarray(values, dtype=np.float32).reshape(-1)
                if vector.size != zone.channel_count:
                    raise ValueError(
                        f"{part_name}.{zone.name} has {vector.size} values; "
                        f"expected {zone.channel_count}"
                    )
            result[f"state.tactile_{part_name}_{zone.name}"] = vector[None, None]
    return result
