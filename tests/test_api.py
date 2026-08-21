import pandas as pd
import pytest

from icdn import ICDNModel
from icdn.data import TemporalSplitter, PanelBuilder, FeatureBuilder

def test_warmup_validation_smoothing_uses_train_history(panel, config):
    """Warm-up val targets must roll across the train/val boundary."""
    config.smoothing_window = 3
    features = FeatureBuilder(config)
    long_df = features.run(panel)
    builder = PanelBuilder(config)
    wide = builder.fit_transform(long_df, features.shared_features, features.product_features)
    layout = builder.layout

    store, period = config.schema.store, config.schema.period
    train_wide, val_wide = TemporalSplitter(period_col=period).single_split(
        wide, train_frac=1.0 - config.validation_fraction
    )
    val_wide = val_wide.reset_index(drop=True)

    model = ICDNModel(config)
    model.layout = layout
    got = model._build_loaders(train_wide, val_wide)["warmup_val"].dataset.demands.numpy()

    isolated = model._smooth(val_wide.reset_index(drop=True))
    first = val_wide[period].idxmin() if False else val_wide[period].min()
    row = val_wide[val_wide[period] == first].index[0]
    s = val_wide.loc[row, store]
    i = 0

    hist = (
        train_wide[(train_wide[store] == s) & (train_wide[f"obs_mask_{i}"] > 0)]
        .sort_values(period)
        .tail(config.smoothing_window - 1)[f"log_demand_{i}"]
    )
    y0 = float(val_wide.loc[row, f"log_demand_{i}"])
    expected = float(pd.concat([hist, pd.Series([y0])], ignore_index=True).mean())

    assert isolated[f"log_demand_{i}"].iloc[row] == pytest.approx(y0)
    assert got[row, i] == pytest.approx(expected)
    assert got[row, i] != pytest.approx(y0)


def test_prepare_keeps_requested_store_product_period_support(panel, config):
    """The train tail must not reappear as current observations.
    Filtering the engineered panel by requested *periods* is not enough:
    another series' tail can share a period the caller asked for on a
    different store or product.
    """
    model = ICDNModel(config).fit(panel)
    schema = config.schema
    store, product, period = schema.store, schema.product, schema.period

    # Last training weeks are what _train_tail actually contains.
    early, late = 40, 45
    a, b = "P0", "P1"
    s0, s1 = "S0", "S1"

    # Case 1: same product, two stores, staggered periods.
    # Old code kept (S1, P0, 40) from S1's tail because 40 is a requested period.
    staggered = panel[
        ((panel[store] == s0) & (panel[product] == a) & (panel[period] == early))
        | ((panel[store] == s1) & (panel[product] == a) & (panel[period] == late))
    ]
    wide, _ = model._prepare(staggered)
    i_a = model.layout.products.index(a)
    observed = {
        (row[store], model.layout.products[i], row[period])
        for _, row in wide.iterrows()
        for i in range(model.layout.n_products)
        if row[f"obs_mask_{i}"] > 0
    }
    assert observed == {(s0, a, early), (s1, a, late)}
    assert (s1, early) not in set(zip(wide[store], wide[period], strict=True))

    # Case 2: same store, two products, staggered periods.
    # Old code set obs_mask of B at week 40 and counted B as a neighbor of A.
    same_store = panel[
        ((panel[store] == s0) & (panel[product] == a) & (panel[period] == early))
        | ((panel[store] == s0) & (panel[product] == b) & (panel[period] == late))
    ]
    wide, _ = model._prepare(same_store)
    i_b = model.layout.products.index(b)
    early_row = wide[(wide[store] == s0) & (wide[period] == early)].iloc[0]
    assert early_row[f"obs_mask_{i_a}"] == 1.0
    assert early_row[f"obs_mask_{i_b}"] == 0.0
    assert early_row[f"n_neighbors_{i_a}"] == 0.0

    late_row = wide[(wide[store] == s0) & (wide[period] == late)].iloc[0]
    assert late_row[f"obs_mask_{i_b}"] == 1.0
    assert late_row[f"miss_lag_1_{i_b}"] == 0.0  # tail still feeds history

    assert model.evaluate(same_store)["n_obs"] == 2

def test_score_warns_when_price_is_outside_spline_knots(panel, config):
    model = ICDNModel(config).fit(panel)
    extreme = panel.copy()
    extreme[config.schema.price] = extreme[config.schema.price] * 50
    with pytest.warns(UserWarning, match="outside the training spline knots"):
        model.score(extreme)

def test_evaluate_defaults_to_validation_split(panel, config):
    model = ICDNModel(config).fit(panel)
    period = config.schema.period
    _, val_raw = TemporalSplitter(period_col=period).single_split(
        panel, train_frac=1.0 - config.validation_fraction
    )
    default = model.evaluate()
    holdout = model.evaluate(val_raw)
    full = model.evaluate(panel)

    assert default == holdout
    assert full["n_obs"] > default["n_obs"]

def test_fit_reports_both_training_phases(panel, config):
    model = ICDNModel(config).fit(panel)

    assert model.is_fitted
    assert set(model.history) == {"warmup", "main"}
    assert len(model.products) == 4

def test_score_returns_one_row_per_store_period_and_product(panel, config):
    model = ICDNModel(config).fit(panel)
    scored = model.score()

    expected = [config.schema.store, config.schema.period, config.schema.product, "predicted_demand"]
    assert set(expected) <= set(scored.columns)
    assert scored["predicted_demand"].notna().all()
    assert scored[config.schema.product].nunique() == 4

def test_elasticities_omit_cross_when_use_cross_is_false(panel, config):
    config.use_cross = False
    model = ICDNModel(config).fit(panel)
    elasticities = model.elasticities()
    raw = model.elasticities(aggregate=False)

    assert set(elasticities["kind"]) == {"own"}
    assert (elasticities[config.schema.product] == elasticities["competitor"]).all()
    assert set(raw["kind"]) == {"own"}
    assert (raw[config.schema.product] == raw["competitor"]).all()

def test_elasticities_cover_own_and_cross_effects(panel, config):
    model = ICDNModel(config).fit(panel)
    elasticities = model.elasticities()

    assert set(elasticities["kind"]) == {"own", "cross"}
    own = elasticities[elasticities["kind"] == "own"]
    assert (own[config.schema.product] == own["competitor"]).all()
    assert {"temporal_q025", "temporal_q975", "n_obs"} <= set(elasticities.columns)


def test_raw_elasticities_keep_one_row_per_observation(panel, config):
    model = ICDNModel(config).fit(panel)
    raw = model.elasticities(aggregate=False)

    assert config.schema.period in raw.columns
    assert len(raw) > len(model.elasticities())


def test_evaluate_reports_masked_regression_metrics(panel, config):
    model = ICDNModel(config).fit(panel)
    metrics = model.evaluate()

    assert set(metrics) == {"mae", "rmse", "r2", "n_obs"}
    assert metrics["n_obs"] > 0


def test_saved_and_reloaded_model_scores_identically(panel, config, tmp_path):
    model = ICDNModel(config).fit(panel)
    before = model.score(panel)

    path = model.save(tmp_path / "model")
    restored = ICDNModel.load(path)
    after = restored.score(panel)

    pd.testing.assert_frame_equal(before, after)
    assert restored.products == model.products


def test_score_without_data_after_loading_is_explicit(panel, config, tmp_path):
    model = ICDNModel(config).fit(panel)
    restored = ICDNModel.load(model.save(tmp_path / "model"))

    with pytest.raises(ValueError, match="no panel supplied"):
        restored.score()


def test_using_the_model_before_fitting_fails_clearly(config):
    with pytest.raises(RuntimeError, match="not fitted"):
        ICDNModel(config).score()
