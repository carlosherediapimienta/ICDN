# ICDN

Demand and price-elasticity estimation for retail panels, powered by an
Integrable Context-Dependent Demand Network.

The model fits a single smooth log-demand surface conditioned on store,
calendar, promotional and competitive context, and reads own- and cross-price
elasticities off its derivatives. Elasticity estimation
therefore come from the same fitted object instead of separate regressions.

## Install

```bash
pip install icdn
```

## Quickstart

Your panel needs one row per store, product and period:

| store_code | product_code | week_id | price | units | on_promo |
|---|---|---|---|---|---|
| S1 | P1 | 1 | 2.49 | 130 | 0 |
| S1 | P2 | 1 | 3.10 | 88 | 1 |

```python
import pandas as pd
from icdn import ICDNModel, ICDNConfig

panel = pd.read_parquet("panel.parquet")

model = ICDNModel(ICDNConfig(n_products=5))
model.fit(panel)

elasticities = model.elasticities()
scored = model.score(panel)
model.save("artifacts/icdn")
```

`elasticities()` returns one row per store and product pair:

| store_code | product | competitor | kind | elasticity | temporal_q025 | temporal_q975 |
|---|---|---|---|---|---|---|
| S1 | P1 | P1 | own | -2.14 | -2.42 | -1.88 |
| S1 | P1 | P2 | cross | 0.37 | 0.19 | 0.55 |

Lags, seasonality, promotional pressure and competitive context are engineered
internally, so nothing beyond the six columns above is required.

## Mapping your columns

If your column names differ, describe them once:

```python
from icdn import ICDNConfig, PanelSchema

config = ICDNConfig(
    schema=PanelSchema(
        store="shop_id",
        product="sku",
        period="week",
        price="avg_price",
        units="qty",
        promo="promo_flag",
        category="category",   # optional
        brand="brand",         # optional
    ),
    n_products=10,
)
```

## Command line

```bash
icdn fit --data panel.parquet --config configs/default.yaml --out artifacts/model.icdn
icdn score --model artifacts/model.icdn --data panel.parquet --out scores.csv
icdn elasticities --model artifacts/model.icdn --data panel.parquet --out elasticities.csv
```

## Documentation

- [Getting started](docs/getting-started.md) covers the data contract, the
  configuration reference and how training works.

## Development

```bash
pip install icdn
```

## License

See [LICENSE](LICENSE).

## Relation to the paper

This library is an improved implementation of
[Heredia & Roncel (2026)](https://arxiv.org/abs/2605.22820), not a bit-for-bit
reproduction of the architecture used in the reported experiments.

The published model had no product-identity embedding: two SKUs with the same
covariates received the same token. This package adds a learned embedding of
size `n × d_product` (default `d_product=16`), indexed by the frozen panel
slot, so product heterogeneity is no longer carried only by covariates. That
correction is the right modelling choice; it also means published parameter
counts, scalability tables and ablations refer to the previous architecture.

The paper states that the number of parameters does not depend on `n` and
reports 64,285 parameters. With the product embedding, the count still does
not depend on `k` (the competitor set), but it grows as `n · d_product`:

| Products (`n`) | Parameters |
|---|---|
| 5 | 66,205 |
| 25 | 66,525 |
| 50 | 66,925 |
| 100 | 67,725 |
| 200 | 69,325 |

Treat this package as a new architectural revision. Reuse the paper's
hyperparameters as a starting point, not as a claim that `fit()` recovers the
published metrics. Empirical numbers, tuning and ablations would need to be
re-run to speak for this version.

## Citation
If you use this software in your research, please cite:
> Heredia, C., & Roncel, D. (2026). *Integrable Elasticity via Neural Demand Potentials*. arXiv:2605.22820.  
> https://arxiv.org/abs/2605.22820
```bibtex
@article{heredia2026integrable,
  title   = {Integrable Elasticity via Neural Demand Potentials},
  author  = {Heredia, Carlos and Roncel, Daniel},
  journal = {arXiv preprint arXiv:2605.22820},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.22820}
}