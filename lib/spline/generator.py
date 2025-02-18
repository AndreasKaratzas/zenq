from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataConfig:
    num_points: int = 200
    batch_size: int = 32
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    num_workers: int = 4
    pin_memory: bool = True


class SplineDataset(Dataset):
    """Dataset for spline fitting."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.FloatTensor(x.reshape(-1, 1))
        self.y = torch.FloatTensor(y.reshape(-1, 1))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SplineDataModule(pl.LightningDataModule):
    """DataModule for spline fitting."""

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.datasets: dict = {}

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for spline fitting."""
        x = np.linspace(-2, 2, self.config.num_points)
        y = np.sin(2 * x) + 0.5 * x**2
        y += np.random.normal(0, 0.1, x.shape)
        return x, y

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if not self.datasets:
            x, y = self.generate_data()

            # Create split indices
            indices = np.random.permutation(self.config.num_points)
            split_points = np.cumsum(
                [
                    int(split * self.config.num_points)
                    for split in self.config.train_val_test_split
                ]
            )

            # Split data
            splits = np.split(indices, split_points[:-1])
            stages = ["train", "val", "test"]

            # Create datasets
            self.datasets = {
                stage: SplineDataset(x[idx], y[idx])
                for stage, idx in zip(stages, splits)
            }

    def _create_dataloader(self, stage: str) -> DataLoader:
        """Create dataloader for a specific stage."""
        shuffle = stage == "train"
        return DataLoader(
            self.datasets[stage],
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader("test")
