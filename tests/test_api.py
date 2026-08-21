import pandas as pd
import pytest

from icdn import ICDNModel
from icdn.data import TemporalSplitter

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
