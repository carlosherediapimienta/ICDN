import numpy as np
import pandas as pd
import pytest

from icdn import ICDNConfig, PanelSchema

N_STORES = 3
N_PRODUCTS = 4
N_WEEKS = 60


@pytest.fixture
def panel() -> pd.DataFrame:
    """Synthetic weekly panel with known own- and cross-price effects."""
    rng = np.random.default_rng(0)
    rows = []

    base_price = {p: 2.0 + 0.5 * p for p in range(N_PRODUCTS)}
    own_elasticity = -2.0
    cross_elasticity = 0.4

    for store in range(N_STORES):
        for week in range(N_WEEKS):
            prices = {
                p: base_price[p] * np.exp(rng.normal(0, 0.15)) for p in range(N_PRODUCTS)
            }
            for product, price in prices.items():
                competitor = (product + 1) % N_PRODUCTS
                log_demand = (
                    5.0
                    + 0.3 * store
                    + own_elasticity * np.log(price)
                    + cross_elasticity * np.log(prices[competitor])
                    + 0.2 * np.sin(2 * np.pi * week / 52)
                    + rng.normal(0, 0.05)
                )
                rows.append(
                    {
                        "store_code": f"S{store}",
                        "product_code": f"P{product}",
                        "week_id": week,
                        "price": price,
                        "units": np.exp(log_demand),
                        "on_promo": int(rng.random() < 0.2),
                        "category_code": "beverages",
                        "brand": f"B{product % 2}",
                        "style": f"T{product % 3}",
                        "size": 0.33 * (1 + product % 2),
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture
def config() -> ICDNConfig:
    """Fast configuration for tests: a couple of epochs, small batches."""
    return ICDNConfig(
        schema=PanelSchema(
            category="category_code",
            brand="brand",
            style="style",
            size="size",
        ),
        n_products=N_PRODUCTS,
        k_neighbors=2,
        hidden=(16, 8),
        batch_size=32,
        warmup_epochs=2,
        epochs=3,
        num_workers=0,
        verbose=False,
    )
