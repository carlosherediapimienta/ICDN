"""Construction of the per-product spline basis from training prices."""

import numpy as np
import torch

from .multi_cubic import MultiCubicSplineBasis

# Guards against unstable normalization when a product barely changes price.
_MIN_SCALE = 0.2


class SplineBuilder:
    """Places spline knots at price quantiles observed during training.

    Knots follow the empirical distribution of each product's log-price, so the
    basis is dense where prices actually move instead of being spread evenly
    over an arbitrary range.
    """

    def build_for_product(
        self,
        prices: np.ndarray,
        n_knots: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95,
    ) -> dict:
        """Returns the knots, shift and scale of a single product."""
        x = np.asarray(prices, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("no finite prices available to build the spline basis")

        knots = np.quantile(x, np.linspace(q_min, q_max, n_knots))
        return {
            "knots": torch.tensor(knots, dtype=torch.float32),
            "shift": float(x.mean()),
            "scale": float(max(x.std(ddof=0), _MIN_SCALE)),
        }

    def build_basis(
        self,
        prices: np.ndarray,
        n_knots: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95,
    ) -> MultiCubicSplineBasis:
        """Builds the shared basis from a (n_observations, n_products) matrix."""
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 2:
            raise ValueError(f"prices must have shape (N, n), got {prices.shape}")

        configs = [
            self.build_for_product(prices[:, i], n_knots=n_knots, q_min=q_min, q_max=q_max)
            for i in range(prices.shape[1])
        ]
        return MultiCubicSplineBasis(
            knots=torch.stack([c["knots"] for c in configs], dim=0),
            shift=torch.tensor([c["shift"] for c in configs], dtype=torch.float32),
            scale=torch.tensor([c["scale"] for c in configs], dtype=torch.float32),
        )
