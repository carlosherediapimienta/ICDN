import numpy as np
import pandas as pd
import pytest

from icdn import ICDNConfig
from icdn.data import FeatureBuilder


def test_config_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="k_neighbors"):
        ICDNConfig(k_neighbors=-1)
    with pytest.raises(ValueError, match="validation_fraction"):
        ICDNConfig(validation_fraction=0.0)
    with pytest.raises(ValueError, match="validation_fraction"):
        ICDNConfig(validation_fraction=1.0)
    with pytest.raises(ValueError, match="dropout"):
        ICDNConfig(dropout=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        ICDNConfig(batch_size=0)
    with pytest.raises(ValueError, match="epochs"):
        ICDNConfig(epochs=0)
    with pytest.raises(ValueError, match="n_knots"):
        ICDNConfig(n_knots=0)
    with pytest.raises(ValueError, match="lags"):
        ICDNConfig(lags=())
    with pytest.raises(ValueError, match="rolling_windows"):
        ICDNConfig(rolling_windows=())
    with pytest.raises(ValueError, match="own_elasticity_bounds"):
        ICDNConfig(own_elasticity_bounds=(0.0, -1.0))
    with pytest.raises(ValueError, match="cross_elasticity_bounds"):
        ICDNConfig(cross_elasticity_bounds=(1.0, -1.0))

@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_products": 1}, "n_products"),
        ({"lambda_smooth": -0.1}, "lambda_smooth"),
        ({"lambda_elast": -0.1}, "lambda_elast"),
        ({"huber_delta": 0.0}, "huber_delta"),
        ({"lr": 0.0}, "lr"),
        ({"warmup_lr": 0.0}, "warmup_lr"),
        ({"weight_decay": -1e-5}, "weight_decay"),
        ({"grad_clip": 0.0}, "grad_clip"),
        ({"smoothing_window": 0}, "smoothing_window"),
        ({"seasonal_periods": (52, 0)}, "seasonal"),
        ({"num_workers": -1}, "num_workers"),
        ({"hidden": (16, 0)}, "hidden"),
        ({"activation": "relu"}, "activation"),
        ({"own_elasticity_bounds": (-np.inf, 0.0)}, "own_elasticity_bounds"),
    ],
)
def test_config_rejects_invalid_values_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ICDNConfig(**kwargs)


def test_panel_rejects_configured_optional_column_if_missing(panel, config):
    with pytest.raises(ValueError, match="optional columns"):
        FeatureBuilder(config).run(panel.drop(columns=["brand"]))


def test_panel_rejects_null_identifiers_and_nonfinite_periods(panel, config):
    broken = panel.copy()
    broken.loc[broken.index[0], "store_code"] = None
    with pytest.raises(ValueError, match="null"):
        FeatureBuilder(config).run(broken)

    broken = panel.copy()
    broken["week_id"] = broken["week_id"].astype(float)
    broken.loc[broken.index[0], "week_id"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        FeatureBuilder(config).run(broken)

def test_panel_rejects_non_binary_promo(panel, config):
    for value in (2, -1, np.inf, "yes"):
        broken = panel.copy()
        broken["on_promo"] = broken["on_promo"].astype(object)
        broken.loc[broken.index[0], "on_promo"] = value
        with pytest.raises(ValueError, match="binary|on_promo"):
            FeatureBuilder(config).run(broken)
    broken = panel.copy()
    broken["on_promo"] = broken["on_promo"].astype(object)
    broken.loc[broken.index[0], "on_promo"] = "yes"
    with pytest.raises(ValueError, match="binary|on_promo"):
        FeatureBuilder(config).run(broken)


def test_panel_rejects_nonfinite_size(panel, config):
    broken = panel.copy()
    broken.loc[broken.index[0], "size"] = np.inf
    with pytest.raises(ValueError, match="size"):
        FeatureBuilder(config).run(broken)


def test_panel_rejects_inconsistent_sku_metadata(panel, config):
    broken = panel.copy()
    sku = broken["product_code"].iloc[0]
    other = broken.index[broken["product_code"] == sku][1]
    broken.loc[other, "brand"] = "OTHER"
    with pytest.raises(ValueError, match="inconsistent static metadata"):
        FeatureBuilder(config).run(broken)


def test_config_accepts_the_library_defaults():
    ICDNConfig()


def test_panel_rejects_duplicate_store_product_period(panel, config):
    duplicated = pd.concat([panel.iloc[[0]], panel], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        FeatureBuilder(config).run(duplicated)


def test_panel_rejects_non_finite_or_non_positive_price(panel, config):
    broken = panel.copy()
    broken.loc[broken.index[0], "price"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        FeatureBuilder(config).run(broken)

    broken = panel.copy()
    broken.loc[broken.index[0], "price"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        FeatureBuilder(config).run(broken)


def test_panel_rejects_negative_units_and_size(panel, config):
    broken = panel.copy()
    broken.loc[broken.index[0], "units"] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        FeatureBuilder(config).run(broken)

    broken = panel.copy()
    broken.loc[broken.index[0], "units"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        FeatureBuilder(config).run(broken)

    broken = panel.copy()
    broken.loc[broken.index[0], "size"] = -0.1
    with pytest.raises(ValueError, match="size"):
        FeatureBuilder(config).run(broken)


def test_panel_rejects_missing_schema_columns(panel, config):
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureBuilder(config).run(panel.drop(columns=["price"]))