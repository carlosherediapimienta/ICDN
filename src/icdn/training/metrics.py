"""Batched inference helpers and masked regression metrics."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.neighbors import ProductMetadata


@torch.no_grad()
def predict_demand(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    meta: ProductMetadata | None = None,
) -> np.ndarray:
    """Predicted log-demand with shape (n_observations, n_products)."""
    model.eval()
    chunks = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y_hat, _ = model(batch, meta=meta)
        chunks.append(y_hat.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def predict_elasticities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    meta: ProductMetadata | None = None,
) -> np.ndarray:
    """Elasticity matrices with shape (n_observations, n_products, n_products).

    Entry ``[t, i, j]`` is the response of product ``i``'s demand to a change
    in product ``j``'s price, so the diagonal holds own-price elasticities.
    """
    model.eval()
    chunks = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, _, aux = model(batch, return_parts=True, compute_E=True, meta=meta)
        chunks.append(aux["E"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def collect_targets(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Observed log-demand and its mask, in loader order."""
    demands, masks = [], []
    for batch in loader:
        demands.append(batch["demands"].numpy())
        masks.append(batch["obs_mask"].numpy())
    return np.concatenate(demands, axis=0), np.concatenate(masks, axis=0)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """MAE, RMSE and R2 computed only over observed product-period cells."""
    selected = mask.astype(bool)
    if not selected.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n_obs": 0}

    truth = y_true[selected]
    pred = y_pred[selected]
    residual = truth - pred

    ss_res = float((residual**2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())

    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual**2).mean())),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_obs": int(selected.sum()),
    }
