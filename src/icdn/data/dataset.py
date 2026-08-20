"""Torch dataset and loaders over the wide panel."""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .panel import STORE_INDEX, PanelLayout


class MultiProductDataset(Dataset):
    """One sample per store and period, holding every product at once.

    All columns are stacked into seven tensors upfront so each batch triggers a
    handful of host-to-device copies instead of hundreds of scalar transfers.
    """

    def __init__(self, wide: pd.DataFrame, layout: PanelLayout, period_col: str):
        n = layout.n_products
        self.layout = layout
        self._len = len(wide)

        self.ids = torch.stack(
            [
                torch.tensor(wide[STORE_INDEX].to_numpy(), dtype=torch.long),
                torch.tensor(pd.factorize(wide[period_col])[0], dtype=torch.long),
            ],
            dim=1,
        )

        self.shared_feats = _stack_columns(wide, layout.shared_features)
        self.prices = _stack_columns(wide, [f"log_price_{i}" for i in range(n)])
        self.demands = _stack_columns(wide, [f"log_demand_{i}" for i in range(n)])
        self.obs_mask = _stack_columns(wide, [f"obs_mask_{i}" for i in range(n)])

        self.product_feats = torch.zeros(self._len, n, len(layout.product_features))
        for j, feature in enumerate(layout.product_features):
            self.product_feats[:, :, j] = _stack_columns(
                wide, [f"{feature}_{i}" for i in range(n)]
            )

        # Brand and style are static per product position, so a single row is
        # broadcast to the whole dataset.
        brand = layout.brand_codes or [0] * n
        style = layout.style_codes or [0] * n
        self.product_cat = torch.tensor([brand, style], dtype=torch.long).T

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "ids": self.ids[idx],
            "shared_feats": self.shared_feats[idx],
            "prices": self.prices[idx],
            "demands": self.demands[idx],
            "obs_mask": self.obs_mask[idx],
            "product_feats": self.product_feats[idx],
            "product_cat": self.product_cat,
        }


class DataLoaderFactory:
    """Creates loaders with consistent worker and memory settings."""

    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def evaluate(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def _stack_columns(wide: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    missing = [c for c in columns if c not in wide.columns]
    if missing:
        raise KeyError(f"the wide panel is missing columns {missing[:5]}")
    return torch.tensor(wide[columns].to_numpy(dtype="float32"), dtype=torch.float32)
