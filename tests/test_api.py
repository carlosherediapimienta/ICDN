import pandas as pd
import pytest

from icdn import ICDNModel


def test_fit_reports_both_training_phases(panel, config):
    model = ICDNModel(config).fit(panel)

    assert model.is_fitted
    assert set(model.history) == {"warmup", "main"}
    assert len(model.products) == 4


def test_predict_returns_one_row_per_store_period_and_product(panel, config):
    model = ICDNModel(config).fit(panel)
    predictions = model.predict()

    expected = [config.schema.store, config.schema.period, config.schema.product, "predicted_demand"]
    assert set(expected) <= set(predictions.columns)
    assert predictions["predicted_demand"].notna().all()
    assert predictions[config.schema.product].nunique() == 4

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


def test_saved_and_reloaded_model_predicts_identically(panel, config, tmp_path):
    model = ICDNModel(config).fit(panel)
    before = model.predict(panel)

    path = model.save(tmp_path / "model")
    restored = ICDNModel.load(path)
    after = restored.predict(panel)

    pd.testing.assert_frame_equal(before, after)
    assert restored.products == model.products


def test_predict_without_data_after_loading_is_explicit(panel, config, tmp_path):
    model = ICDNModel(config).fit(panel)
    restored = ICDNModel.load(model.save(tmp_path / "model"))

    with pytest.raises(ValueError, match="no panel supplied"):
        restored.predict()


def test_using_the_model_before_fitting_fails_clearly(config):
    with pytest.raises(RuntimeError, match="not fitted"):
        ICDNModel(config).predict()
