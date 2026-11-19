"""Small configuration objects shared by the data-hook modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Counts and seed used to build the Phase 1 manifest.

    ``class_limit`` is only used by tests that generate a tiny fake dataset.
    """

    seed: int = 2026
    train_per_class: int = 100
    relearn_per_class: int = 20
    val_per_split_per_class: int = 25
    class_limit: int | None = None
