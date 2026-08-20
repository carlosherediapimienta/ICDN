"""Feature engineering over a long panel, from raw prices to model inputs.

Everything the model consumes is derived here so callers only ever supply
identifiers, prices, units and a promotional flag. Every feature is strictly
historical: lags and rolling statistics exclude the current period, so no
future information leaks into training.
"""

import numpy as np
import pandas as pd

from ..config import ICDNConfig

LOG_PRICE = "log_price"
LOG_DEMAND = "log_demand"
PERIOD_RANK = "period_rank"
PROMO_INTENSITY = "promo_intensity"


class FeatureBuilder:
    """Expands a long panel into the shared and per-product model features.

    After ``run`` the generated column names are available in
    ``shared_features`` (constant across products within a store-period) and
    ``product_features`` (varying by product).
    """

    def __init__(self, config: ICDNConfig):
        self.config = config
        self.schema = config.schema
        self.shared_features: list[str] = []
        self.product_features: list[str] = []
        self._period_rank_map: dict | None = None
        self._max_train_rank: int = 0
        self._product_first_rank: dict | None = None
        self._store_product_first_rank: pd.Series | None = None

    def _validate_panel(self, df: pd.DataFrame) -> None:
        schema = self.schema
        schema.validate(df.columns)
        store, product, period = schema.store, schema.product, schema.period
        keys = [store, product, period]
        duplicated = df.duplicated(keys, keep=False)
        if duplicated.any():
            n = int(duplicated.sum())
            raise ValueError(
                f"{n} duplicate (store, product, period) rows. "
                "Aggregate units (sum) and prices (e.g. quantity-weighted) "
                "before calling fit(); a silent mean after log() is not valid."
            )

        price = pd.to_numeric(df[schema.price], errors="coerce")
        units = pd.to_numeric(df[schema.units], errors="coerce")
        if not np.isfinite(price).all() or not np.isfinite(units).all():
            raise ValueError("price and units must be finite")
        if (price <= 0).any() or (units <= 0).any():
            raise ValueError(
                "price must be strictly positive and units strictly positive. "
                "Pass levels, not logs."
            )

        if schema.size is not None and schema.size in df.columns:
            size = pd.to_numeric(df[schema.size], errors="coerce")
            if (size < 0).any():
                raise ValueError(f"{schema.size} must be non-negative")

    def run(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._validate_panel(panel)
        schema = self.schema

        df = panel.copy()
        store, product, period = schema.store, schema.product, schema.period
        df = df.sort_values([store, product, period]).reset_index(drop=True)

        self.shared_features = []
        self.product_features = []

        df = self._add_targets(df)
        df = self._add_calendar(df)
        df = self._add_lifecycle(df)
        df = self._add_lags_and_rollings(df)
        df = self._add_promo(df)
        df = self._add_competitive(df)
        df = self._add_static_attributes(df)

        return df

    def fit(self, panel: pd.DataFrame) -> "FeatureBuilder":
        self._validate_panel(panel)
        schema = self.schema
        store, product, period = schema.store, schema.product, schema.period

        periods = np.sort(panel[period].dropna().unique())
        self._period_rank_map = {p: i + 1 for i, p in enumerate(periods)}
        self._max_train_rank = len(periods)

        # Save first-appearance ranks so val lifecycle features count from training start
        temp = panel[[store, product, period]].copy()
        temp[PERIOD_RANK] = temp[period].map(self._period_rank_map)
        self._product_first_rank = temp.groupby(product)[PERIOD_RANK].min().to_dict()
        self._store_product_first_rank = temp.groupby([store, product])[PERIOD_RANK].min()
        return self

    def fit_transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        self.fit(panel)
        return self.run(panel)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self._period_rank_map is None:
            raise RuntimeError("FeatureBuilder.transform called before fit_transform")
        # Extend the rank map for unseen (val/test) periods, continuing from training
        unseen = sorted(
            p for p in panel[self.schema.period].dropna().unique()
            if p not in self._period_rank_map
        )
        extended_map = dict(self._period_rank_map)
        for i, p in enumerate(unseen):
            extended_map[p] = self._max_train_rank + i + 1
        # Swap map for this call only; do not mutate fit state
        saved = self._period_rank_map
        self._period_rank_map = extended_map
        result = self.run(panel)
        self._period_rank_map = saved
        return result

    # ── Targets ─────────────────────────────────────────────────────────────

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        schema = self.schema
        price = pd.to_numeric(df[schema.price], errors="coerce")
        units = pd.to_numeric(df[schema.units], errors="coerce")
        if (price <= 0).any() or (units <= 0).any():
            raise ValueError(
                "price must be strictly positive and units strictly positive."
                "Pass levels, not logs: price p > 0 and units > 0."
            )
        df[LOG_PRICE] = np.log(price)
        df[LOG_DEMAND] = np.log(units)
        return df

    # ── Calendar ────────────────────────────────────────────────────────────

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._period_rank_map is not None:
            rank = self._period_rank_map
        else:
            periods = np.sort(df[self.schema.period].dropna().unique())
            rank = {p: i + 1 for i, p in enumerate(periods)}
        df[PERIOD_RANK] = df[self.schema.period].map(rank)
        self.shared_features.append(PERIOD_RANK)

        for p in self.config.seasonal_periods:
            df[f"sin_{p}"] = np.sin(2 * np.pi * df[PERIOD_RANK] / p)
            df[f"cos_{p}"] = np.cos(2 * np.pi * df[PERIOD_RANK] / p)
            self.shared_features += [f"sin_{p}", f"cos_{p}"]
        return df

    def _add_lifecycle(self, df: pd.DataFrame) -> pd.DataFrame:
        store, product = self.schema.store, self.schema.product
        if self._product_first_rank is not None:
            first_product = df[product].map(self._product_first_rank).fillna(df[PERIOD_RANK])
            sp_first = self._store_product_first_rank.reset_index(name="_first_sp")
            df = df.merge(sp_first, on=[store, product], how="left")
            df["_first_sp"] = df["_first_sp"].fillna(df[PERIOD_RANK])
            df["periods_seen_product"] = (df[PERIOD_RANK] - first_product).astype(float)
            df["periods_seen_store_product"] = (df[PERIOD_RANK] - df["_first_sp"]).astype(float)
            df = df.drop(columns=["_first_sp"])
        else:
            first_product = df.groupby(product)[PERIOD_RANK].transform("min")
            first_store_product = df.groupby([store, product])[PERIOD_RANK].transform("min")
            df["periods_seen_product"] = (df[PERIOD_RANK] - first_product).astype(float)
            df["periods_seen_store_product"] = (df[PERIOD_RANK] - first_store_product).astype(float)
        self.product_features += ["periods_seen_product", "periods_seen_store_product"]
        return df

    # ── History ─────────────────────────────────────────────────────────────

    def _week_grid(self, df: pd.DataFrame) -> list:
        """ Contiguous week index from the earliest period in play"""
        period = self.schema.period
        values = set(df[period].dropna().unique())
        if self._period_rank_map is not None:
            values |= set(self._period_rank_map)
        start, end = min(values), max(values)
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
            raise TypeError(
                f"{period} must be an integer week index to build gap-aware lags."
                "Map dates to a sequential week_id before fitting."
            )
        return list(range(int(start), int(end) + 1))
    
    def _join_calendar_feature(
        self,
        df: pd.DataFrame,
        grid: pd.DataFrame,
        col: str,
        miss: str,
    ) -> pd.DataFrame:
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        stacked = (
            grid.stack(future_stack=True)
            .rename(col)
            .rename_axis([store, product, period])
            .reset_index(drop=False)
        )
        df = df.merge(stacked, on=[store, product, period], how="left")
        df[miss] = (~np.isfinite(df[col])).astype(float)
        df[col] = df[col].fillna(0.0)
        self.product_features += [col, miss]
        return df

    def _add_lags_and_rollings(self, df: pd.DataFrame) -> pd.DataFrame:
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        grid = self._week_grid(df)

        demand = (
            df.pivot_table(
                index=[store, product],
                columns=period,
                values=LOG_DEMAND,
                aggfunc="mean",
            )
        .reindex(columns=grid)
        )

        for k in self.config.lags:
            col, miss = f"lag_{k}", f"miss_lag_{k}"
            df = self._join_calendar_feature(df, demand.shift(k, axis=1), col, miss)

        for window in self.config.rolling_windows:
            col, miss = f"roll_{window}", f"miss_roll_{window}"
            # shift(1) keeps the window strictly historical; rolling is over weeks, no rows.
            rolled = demand.shift(1, axis=1).T.rolling(window, min_periods=1).mean().T
            df = self._join_calendar_feature(df, rolled, col, miss)

        return df

    def _add_promo(self, df: pd.DataFrame) -> pd.DataFrame:
        promo = pd.to_numeric(df[self.schema.promo], errors="coerce").fillna(0.0)
        df["promo"] = promo.astype(float)
        self.product_features.append("promo")

        intensity = df.groupby([self.schema.store, self.schema.period])["promo"].transform("mean")
        df[PROMO_INTENSITY] = intensity
        self.shared_features.append(PROMO_INTENSITY)
        return df

    # ── Competitive context ─────────────────────────────────────────────────

    def _add_competitive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the competitive context of each product within its neighborhood.

        A neighborhood is the set of other products sold in the same store and
        period, restricted to the same category when one is configured. All
        aggregates exclude the product itself.
        """
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        category = self.schema.category

        group = [store, period] + ([category] if category else [])
        brand = self.schema.brand

        size = df.groupby(group, observed=True)[product].transform("count")
        n_neighbors = (size - 1).clip(lower=0)
        df["n_neighbors"] = n_neighbors.astype(float)
        self.product_features.append("n_neighbors")

        promo_sum = df.groupby(group, observed=True)["promo"].transform("sum")
        df["neighbor_promo_share"] = _share(promo_sum - df["promo"], n_neighbors)
        self.product_features.append("neighbor_promo_share")

        if brand is not None:
            group_brand = group + [brand]
            brand_size = df.groupby(group_brand, observed=True)[product].transform("count")
            n_same_brand = (brand_size - 1).clip(lower=0)
            brand_promo = df.groupby(group_brand, observed=True)["promo"].transform("sum")
            df["n_same_brand_neighbors"] = n_same_brand.astype(float)
            df["same_brand_promo_share"] = _share(brand_promo - df["promo"], n_same_brand)
            self.product_features += ["n_same_brand_neighbors", "same_brand_promo_share"]

        history_cols = [f"lag_{self.config.lags[0]}", f"roll_{self.config.rolling_windows[0]}"]
        for col in history_cols:
            mean_col = f"neighbor_{col}"
            miss_col = f"miss_neighbor_{col}"
            observed = 1.0 - df[f"miss_{col}"]
            values = df[col] * observed

            keys = [df[c] for c in group]
            total = values.groupby(keys).transform("sum")
            counts = observed.groupby(keys).transform("sum")

            neighbor_total = total - values
            neighbor_count = (counts - observed).clip(lower=0)
            mean = neighbor_total / neighbor_count.replace(0, np.nan)
            df[miss_col] = mean.isna().astype(float)
            df[mean_col] = mean.fillna(0.0)
            self.product_features += [mean_col, miss_col]

        is_new = (df["periods_seen_store_product"] <= self.config.new_product_periods).astype(float)
        new_sum = is_new.groupby([df[c] for c in group]).transform("sum")

        new_neighbors = (new_sum - is_new).clip(lower=0)
        df["n_new_neighbors"] = new_neighbors
        df["share_new_neighbors"] = _share(new_neighbors, n_neighbors)
        self.product_features += ["n_new_neighbors", "share_new_neighbors"]

        static_group = [store] + ([category] if category else [])
        df["assortment_size"] = (
            df.groupby(static_group, observed=True)[product].transform("nunique").astype(float)
        )
        self.product_features.append("assortment_size")
        return df

    def _add_static_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema.size is not None:
            df["product_size"] = pd.to_numeric(df[self.schema.size], errors="coerce").fillna(0.0)
            self.product_features.append("product_size")
        return df


def _share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0).astype(float)
