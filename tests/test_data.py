import pytest

import numpy as np
import pandas as pd
from icdn.data import FeatureBuilder, MultiProductDataset, PanelBuilder, TemporalSplitter


def build_wide(panel, config):
    features = FeatureBuilder(config)
    long_df = features.run(panel)
    builder = PanelBuilder(config)
    wide = builder.fit_transform(long_df, features.shared_features, features.product_features)
    return builder.layout, wide


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
