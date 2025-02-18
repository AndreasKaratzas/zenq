from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SplineConfig:
    n_segments: int = 8
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    x_range: Tuple[float, float] = (-2.0, 2.0)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 1000
    n_points: int = 200
    val_split: float = 0.2
    n_workers: int = 4
    device: str = "cuda"
    log_dir: str = "logs"
