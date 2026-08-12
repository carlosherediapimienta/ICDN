"""Chronological splitting and block resampling for panel data."""

import numpy as np
import pandas as pd


class TemporalSplitter:
    """Chronological train/validation splits, never shuffled.

    Validation periods always come after the training ones, so the reported
    error measures genuine forecasting ability rather than interpolation.
    """

    def __init__(self, period_col: str = "week_id"):
        self.period_col = period_col

    def single_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        periods = sorted(df[self.period_col].unique())
        threshold_idx = min(int(len(periods) * train_frac), len(periods) - 1)
        threshold = periods[max(threshold_idx, 1)]
        train = df[df[self.period_col] < threshold].copy()
        val = df[df[self.period_col] >= threshold].copy()
        return train, val

    def expanding_splits(
        self,
        df: pd.DataFrame,
        n_folds: int,
        min_train_frac: float = 0.5,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Expanding window folds: train on the past, validate on the future."""
        periods = sorted(df[self.period_col].unique())
        n_periods = len(periods)

        min_train = max(1, int(n_periods * min_train_frac))
        val_size = max(1, (n_periods - min_train) // max(n_folds, 1))

        folds = []
        for i in range(n_folds):
            train_end = min_train + i * val_size
            val_end = min(train_end + val_size, n_periods)
            if val_end <= train_end:
                break

            train = df[df[self.period_col].isin(periods[:train_end])].copy()
            val = df[df[self.period_col].isin(periods[train_end:val_end])].copy()
            if len(val) == 0:
                continue
            folds.append((train, val))

        return folds


class BlockBootstrapSampler:
    """Resamples contiguous blocks of periods with replacement.

    Sampling whole blocks instead of individual periods preserves the
    short-run autocorrelation of demand, which keeps the resulting confidence
    intervals honest.
    """

    def __init__(
        self,
        period_col: str = "week_id",
        block_size: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self.period_col = period_col
        self.block_size = block_size
        self.rng = rng or np.random.default_rng()

    def sample(self, df: pd.DataFrame, periods: list | np.ndarray) -> pd.DataFrame:
        periods = sorted(np.asarray(periods).tolist())
        n_periods = len(periods)

        if n_periods < self.block_size:
            return df[df[self.period_col].isin(periods)].copy()

        starts = list(range(0, n_periods - self.block_size + 1, self.block_size)) or [0]
        chosen = self.rng.choice(len(starts), size=len(starts), replace=True)

        pieces = []
        for idx in chosen:
            start = starts[idx]
            block = periods[start : start + self.block_size]
            pieces.append(df[df[self.period_col].isin(block)].copy())

        return pd.concat(pieces, ignore_index=True)
