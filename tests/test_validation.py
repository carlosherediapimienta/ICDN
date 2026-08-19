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
    with pytest.raises(ValueError, match="non-negative"):
        FeatureBuilder(config).run(broken)

    broken = panel.copy()
    broken.loc[broken.index[0], "size"] = -0.1
    with pytest.raises(ValueError, match="size"):
        FeatureBuilder(config).run(broken)


def test_panel_rejects_missing_schema_columns(panel, config):
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureBuilder(config).run(panel.drop(columns=["price"]))