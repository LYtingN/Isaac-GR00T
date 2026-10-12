"""Register tactile model and processor classes for checkpoint loading."""


def register_tactile_checkpoint_classes() -> None:
    """Import custom Hugging Face classes before loading a tactile checkpoint."""

    from ..data import gr00t_tactile_processor  # noqa: F401
    from ..model import gr00t_tactile  # noqa: F401
