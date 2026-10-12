"""Export one fixed tactile layout from the collection environment's jq_skin."""

import argparse
from dataclasses import asdict
from importlib.metadata import distribution
import json
from pathlib import Path

import numpy as np

from .tactile_image import TactileLayout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", default=["upper_body"])
    parser.add_argument("--size", nargs=2, type=int, default=[128, 128])
    args = parser.parse_args()

    layout = TactileLayout.from_jq_skin(args.parts, tuple(args.size))
    metadata = json.loads((args.dataset / "meta" / "modality.json").read_text())
    recorded = {
        key: spec["end"] - spec["start"]
        for key, spec in metadata["state"].items()
        if key.startswith("tactile_")
    }
    expected = {zone.key: int(np.prod(zone.grid_shape)) for zone in layout.zones}
    if recorded != expected:
        raise ValueError(f"Dataset tactile channels do not match jq_skin: {recorded} != {expected}")

    package = distribution("jq-skin")
    source = package.read_text("direct_url.json")
    payload = {
        "source": {
            "package": "jq-skin",
            "version": package.version,
            "direct_url": json.loads(source) if source else None,
        },
        "layout": asdict(layout),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
