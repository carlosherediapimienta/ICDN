import pytest

from icdn.data import FeatureBuilder, MultiProductDataset, PanelBuilder, TemporalSplitter


def build_wide(panel, config):
    features = FeatureBuilder(config)
    long_df = features.run(panel)
    builder = PanelBuilder(config)
    wide = builder.fit_transform(long_df, features.shared_features, features.product_features)
    return builder.layout, wide


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
