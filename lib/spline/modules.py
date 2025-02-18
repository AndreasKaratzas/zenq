from dataclasses import dataclass
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class SplineConfig:
    num_segments: int = 8
    enforce_continuity: bool = True
    learning_rate: float = 1e-3
    x_range: Tuple[float, float] = (-2.0, 2.0)


class CubicSpline(pl.LightningModule):
    """Neural cubic spline implementation using PyTorch Lightning."""

    def __init__(self, config: SplineConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Initialize knot points
        self.register_buffer(
            "knots",
            torch.linspace(
                config.x_range[0], config.x_range[1], config.num_segments + 1
            ),
        )

        # Initialize learnable parameters
        if config.enforce_continuity:
            self._init_continuity_params()
        else:
            self._init_direct_params()

    def _init_continuity_params(self) -> None:
        """Initialize parameters for continuity-enforced spline."""
        self.control_points = nn.Parameter(
            torch.randn(self.config.num_segments + 1, 1) * 0.1
        )
        self.derivatives = nn.Parameter(
            torch.randn(self.config.num_segments + 1, 1) * 0.1
        )

    def _init_direct_params(self) -> None:
        """Initialize parameters for direct coefficient learning."""
        self.coefficients = nn.Parameter(torch.randn(self.config.num_segments, 4) * 0.1)

    def compute_coefficients(self) -> torch.Tensor:
        """Compute spline coefficients based on control points and derivatives."""
        if not self.config.enforce_continuity:
            return self.coefficients

        coefficients = torch.zeros(self.config.num_segments, 4, device=self.device)

        for i in range(self.config.num_segments):
            x0, x1 = self.knots[i : i + 2]
            h = x1 - x0

            y0, y1 = self.control_points[i : i + 2]
            dy0, dy1 = self.derivatives[i : i + 2]

            # Hermite spline coefficients
            coefficients[i] = torch.cat(
                [
                    y0,  # a
                    dy0,  # b
                    (3 * (y1 - y0) / h - 2 * dy0 - dy1) / h,  # c
                    (2 * (y0 - y1) / h + dy0 + dy1) / h**2,  # d
                ]
            )

        return coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the spline model."""
        x = x if x.dim() > 1 else x.unsqueeze(-1)

        # Get coefficients and compute segment indices
        coeffs = self.compute_coefficients()
        segment_idx = torch.clamp(
            torch.sum(x >= self.knots.unsqueeze(0), dim=1) - 1,
            0,
            self.config.num_segments - 1,
        )

        # Compute local coordinates
        x0 = self.knots[segment_idx]
        x_local = x - x0.unsqueeze(-1)

        # Compute powers matrix
        x_powers = torch.cat(
            [torch.ones_like(x_local), x_local, x_local.pow(2), x_local.pow(3)], dim=-1
        )

        # Apply coefficients
        batch_coeffs = coeffs[segment_idx]
        return torch.sum(batch_coeffs * x_powers, dim=-1, keepdim=True)

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Generic step for training, validation, and testing."""
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate, weight_decay=0.01
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
