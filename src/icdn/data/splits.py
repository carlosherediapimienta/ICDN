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
        if not 0.0 < train_frac < 1.0:
            raise ValueError(f"train_frac must be in (0,1)")
        periods = self._unique_periods(df)
        if len(periods) < 2:
            raise ValueError(
                f"Need at least 2 distinct {self.period_col} values for a temporal split; "
                f"got {len(periods)}"
            )
        # At least 1 period in training and 1 in validation
        threshold_idx = int(len(periods) * train_frac)
        threshold_idx = min(max(threshold_idx, 1), len(periods) - 1)
        threshold = periods[threshold_idx]
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
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if not 0.0 < min_train_frac < 1.0:
            raise ValueError("min_train_frac must be in (0, 1)")

        periods = self._unique_periods(df)
        n_periods = len(periods)
        min_train = max(1, int(n_periods * min_train_frac))
        if min_train >= n_periods:
            raise ValueError(
                f"min_train_frac={min_train_frac} leaves no periods for validation "
                f"({n_periods} distinct {self.period_col} values)"
            )

        holdout = n_periods - min_train
        if holdout < n_folds:
            raise ValueError(
                f"not enough periods for {n_folds} expanding folds: "
                f"{n_periods} distinct {self.period_col} values leave {holdout} "
                f"validation periods after min_train_frac={min_train_frac} "
                f"(need at least {n_folds}). Lower n_folds or provide more history."
            )
        val_size = max(1, holdout // n_folds)

        folds = []
        for i in range(n_folds):
            train_end = min_train + i * val_size
            if train_end >= n_periods:
                break
            if i == n_folds - 1:
                val_end = n_periods
            else:
                val_end = min(train_end + val_size, n_periods)
            if val_end <= train_end:
                break

            train = df[df[self.period_col].isin(periods[:train_end])].copy()
            val = df[df[self.period_col].isin(periods[train_end:val_end])].copy()
            if len(val) == 0:
                continue
            folds.append((train, val))

        if len(folds) != n_folds:
            raise ValueError(
                f"expanding_splits produced {len(folds)} folds, expected {n_folds}. "
                "Not enough periods for the requested protocol."
            )
        return folds

    # --- Helpers
    def _unique_periods(self, df: pd.DataFrame) -> list:
        col = df[self.period_col]
        if col.isna().any():
            raise ValueError(f"{self.period_col} contains null values")
        periods = sorted(col.unique().tolist())
        if not periods:
            raise ValueError(f"{self.period_col} has no values")
        return periods


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
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.period_col = period_col
        self.block_size = block_size
        self.rng = rng or np.random.default_rng()

    def sample(self, df: pd.DataFrame, periods: list | np.ndarray) -> pd.DataFrame:
        periods = sorted(np.asarray(periods).tolist())
        n_periods = len(periods)

        if n_periods < self.block_size:
            return df[df[self.period_col].isin(periods)].copy()

        length = self.block_size
        starts = list(range(0, n_periods, length))  # non-overlapping partition
        n_blocks = int(np.ceil(n_periods / length))
        chosen = self.rng.choice(len(starts), size=n_blocks, replace=True)

        pieces = []
        next_id = 0
        gap = 1
        kept = 0
        for idx in chosen:
            if kept >= n_periods:
                break
            start = int(starts[idx])
            block_periods = periods[start : start + length]
            take = min(len(block_periods), n_periods - kept)
            block_periods = block_periods[:take]
            chunk = df[df[self.period_col].isin(block_periods)].copy()
            origin = int(block_periods[0])
            remap = {old: next_id + int(old) - origin for old in block_periods}
            chunk[self.period_col] = chunk[self.period_col].map(remap)
            pieces.append(chunk)
            kept += take
            span = int(block_periods[-1]) - origin + 1
            next_id += span + gap

        return pd.concat(pieces, ignore_index=True)