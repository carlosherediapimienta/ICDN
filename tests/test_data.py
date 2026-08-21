import pytest
import numpy as np
import pandas as pd
import torch
from icdn.data.features import LOG_PRICE

from icdn.data import (
    FeatureBuilder, MultiProductDataset, PanelBuilder, TemporalSplitter, BlockBootstrapSampler
)
from icdn.model.splines import SplineBuilder

def build_wide(panel, config):
    features = FeatureBuilder(config)
    long_df = features.run(panel)
    builder = PanelBuilder(config)
    wide = builder.fit_transform(long_df, features.shared_features, features.product_features)
    return builder.layout, wide

def test_block_bootstrap_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size"):
        BlockBootstrapSampler(block_size=0)

def test_expanding_splits_rejects_too_few_periods_for_the_requested_folds():
    df = pd.DataFrame({"week_id": [0, 1, 2, 3]})
    splitter = TemporalSplitter(period_col="week_id")
    with pytest.raises(ValueError, match="not enough periods"):
        splitter.expanding_splits(df, n_folds=5, min_train_frac=0.5)

def test_block_bootstrap_uses_non_overlapping_starts():
    """Overlapping moving blocks would offer n - L + 1 starts, not ceil(n / L)."""
    periods = list(range(10))
    df = pd.DataFrame({"week_id": periods, "x": periods})

    class _Rng:
        def choice(self, a, size=None, replace=True):
            assert a == 3  # starts 0, 4, 8 — overlapping would be 7
            return np.zeros(size, dtype=int)

    out = BlockBootstrapSampler(block_size=4, rng=_Rng()).sample(df, periods)
    # Three copies of [0,1,2,3] with a 1-week hole between blocks.
    assert set(out["week_id"]) == {0, 1, 2, 3, 5, 6, 7, 8, 10, 11}


def test_block_bootstrap_preserves_calendar_gaps():
    df = pd.DataFrame(
        {
            "week_id": [1, 3, 5, 7],
            "store_code": "S0",
            "x": [1, 2, 3, 4],
        }
    )

    class _Rng:
        def choice(self, a, size=None, replace=True):
            return np.zeros(size, dtype=int)  # always [1, 3]

    out = BlockBootstrapSampler(block_size=2, rng=_Rng()).sample(df, [1, 3, 5, 7])
    # Two copies of [1, 3] → [0, 2] and [4, 6], not compressed to consecutive ids.
    assert set(out["week_id"]) == {0, 2, 4, 6}

def test_min_products_drops_sparse_store_periods_from_the_wide_panel(config):
    config.min_products = 2
    config.n_products = None
    config.min_coverage = 0.0

    schema = config.schema
    rows = []
    for week, products in [(1, ["A", "B", "C"]), (2, ["A"])]:
        for sku in products:
            rows.append(
                {
                    schema.store: "S0",
                    schema.product: sku,
                    schema.period: week,
                    schema.price: 2.0,
                    schema.units: 10.0,
                    schema.promo: 0,
                    schema.category: "beverages",
                    schema.brand: "B0",
                    schema.style: "T0",
                    schema.size: 0.33,
                }
            )
    panel = pd.DataFrame(rows)

    features = FeatureBuilder(config)
    long_df = features.fit_transform(panel)
    builder = PanelBuilder(config)
    wide = builder.fit_transform(
        long_df, features.shared_features, features.product_features
    )

    assert set(wide[schema.period]) == {1}
    assert (wide[schema.store] == "S0").all()

    # The bug is in transform(original), not only in fit().
    again = builder.transform(long_df)
    assert set(again[schema.period]) == {1}

def test_spline_basis_uses_observed_prices_not_imputed_wide(panel, config):
    """Knots/shift/scale must come from observed train prices, not imputed fills."""
    schema = config.schema
    store, product, period = schema.store, schema.product, schema.period

    features = FeatureBuilder(config)
    full_long = features.fit_transform(panel)
    builder = PanelBuilder(config)
    builder.fit(full_long, features.shared_features, features.product_features)
    products = builder.layout.products
    assert len(products) >= 2

    target = products[0]
    hole_store = full_long[store].iloc[0]
    # Wipe one store for the target SKU → column-mean fallback pollutes that column.
    drop = (full_long[product] == target) & (full_long[store] == hole_store)
    train_long = full_long.loc[~drop].reset_index(drop=True)
    assert int(drop.sum()) > 10

    train_wide = builder.transform(train_long)
    n_knots = config.n_knots

    observed = (
        train_long.loc[train_long[product].isin(products)]
        .pivot_table(
            index=[period, store],
            columns=product,
            values=LOG_PRICE,
            aggfunc="mean",
            observed=True,
        )
        .reindex(columns=products)
        .to_numpy(dtype=np.float64)
    )
    assert np.isnan(observed).any()

    imputed = train_wide[[f"log_price_{i}" for i in range(len(products))]].to_numpy(
        dtype=np.float64
    )
    assert np.isfinite(imputed).all()
    assert np.isfinite(imputed).sum() > np.isfinite(observed).sum()

    spline = SplineBuilder()
    basis_obs = spline.build_basis(observed, n_knots=n_knots)
    basis_imp = spline.build_basis(imputed, n_knots=n_knots)

    # Mean-fallback cells change the empirical price distribution.
    assert not torch.allclose(basis_obs.shift, basis_imp.shift)
    assert not torch.allclose(basis_obs.knots, basis_imp.knots)

    from icdn import ICDNModel

    model = ICDNModel(config)
    model.layout = builder.layout
    model._panel_builder = builder
    built = model._build_model(train_long).price_splines

    torch.testing.assert_close(built.knots.cpu(), basis_obs.knots)
    torch.testing.assert_close(built.shift.cpu(), basis_obs.shift)
    torch.testing.assert_close(built.scale.cpu(), basis_obs.scale)


def test_validation_lags_use_train_history(panel, config):
    """Holdout features must see the training tail, as in fit() / _prepare()."""
    period = config.schema.period
    store, product = config.schema.store, config.schema.product
    train_raw, val_raw = TemporalSplitter(period_col=period).single_split(
        panel, train_frac=1.0 - config.validation_fraction
    )
    assert not val_raw.empty
    n_tail = max(list(config.lags) + list(config.rolling_windows))
    train_tail = (
        train_raw.sort_values([store, product, period])
        .groupby([store, product], group_keys=False)
        .tail(n_tail)
    )
    features = FeatureBuilder(config)
    features.fit_transform(train_raw)
    first_val = val_raw[period].min()
    tail = train_tail[train_tail[period] < first_val]
    val_extended = pd.concat([tail, val_raw], ignore_index=True) if not tail.empty else val_raw
    val_long = features.transform(val_extended)
    val_long = val_long[val_long[period].isin(val_raw[period].unique())]
    # Pick one series present in both sides of the split.
    keys = [store, product]
    shared = (
        train_raw[keys].drop_duplicates()
        .merge(val_raw[keys].drop_duplicates(), on=keys)
        .iloc[0]
    )
    s, p = shared[store], shared[product]
    last_train = (
        train_raw[(train_raw[store] == s) & (train_raw[product] == p)]
        .sort_values(period)
        .iloc[-1]
    )
    first_holdout = (
        val_long[(val_long[store] == s) & (val_long[product] == p)]
        .sort_values(period)
        .iloc[0]
    )
    expected_lag = float(np.log(last_train[config.schema.units]))
    assert first_holdout["miss_lag_1"] == 0.0
    assert first_holdout["lag_1"] == pytest.approx(expected_lag, rel=0, abs=1e-9)
    # Guard against the old bug: transforming val alone marks lag_1 as missing.
    cold = features.transform(val_raw)
    cold_first = (
        cold[(cold[store] == s) & (cold[product] == p)]
        .sort_values(period)
        .iloc[0]
    )
    assert cold_first["miss_lag_1"] == 1.0


def test_feature_builder_generates_history_without_leakage(panel, config):
    features = FeatureBuilder(config)
    out = features.run(panel)

    assert {"log_price", "log_demand", "period_rank"} <= set(out.columns)
    assert "lag_1" in features.product_features
    assert "promo_intensity" in features.shared_features

    # The first period of a series has no history, so its lag must be flagged.
    first = out.sort_values("week_id").groupby(["store_code", "product_code"]).head(1)
    assert (first["miss_lag_1"] == 1.0).all()


def test_feature_builder_requires_the_schema_columns(panel, config):
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureBuilder(config).run(panel.drop(columns=["price"]))


def test_panel_builder_produces_one_row_per_store_and_period(panel, config):
    layout, wide = build_wide(panel, config)

    assert layout.n_products == 4
    assert layout.n_stores == 3
    assert len(wide) == wide[["store_code", "week_id"]].drop_duplicates().shape[0]
    for i in range(layout.n_products):
        assert f"log_price_{i}" in wide.columns
        assert f"obs_mask_{i}" in wide.columns


def test_dataset_exposes_the_expected_tensor_shapes(panel, config):
    layout, wide = build_wide(panel, config)
    dataset = MultiProductDataset(wide, layout, period_col="week_id")
    sample = dataset[0]

    n = layout.n_products
    assert sample["prices"].shape == (n,)
    assert sample["product_feats"].shape == (n, len(layout.product_features))
    assert sample["product_cat"].shape == (n, 2)
    assert sample["shared_feats"].shape == (len(layout.shared_features),)


def test_temporal_split_keeps_validation_in_the_future(panel, config):
    _, wide = build_wide(panel, config)
    train, val = TemporalSplitter(period_col="week_id").single_split(wide, train_frac=0.8)

    assert train["week_id"].max() < val["week_id"].min()
    assert len(train) > len(val)


def test_expanding_folds_never_train_on_the_future(panel, config):
    _, wide = build_wide(panel, config)
    folds = TemporalSplitter(period_col="week_id").expanding_splits(wide, n_folds=3)

    assert len(folds) == 3
    for train, val in folds:
        assert train["week_id"].max() < val["week_id"].min()
