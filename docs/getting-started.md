# Getting started

## 1. The data contract

ICDN consumes a **long panel**: one row per store, product and period.

### Required columns

| Role | Default name | Notes |
|---|---|---|
| Store | `store_code` | Any identifier. Stores seen at training time are the only ones accepted later. |
| Product | `product_code` | Any identifier. |
| Period | `week_id` | Must sort chronologically. An integer week index is ideal. |
| Price | `price` | Strictly positive, on the same basis as `units`. |
| Units | `units` | Non-negative sales volume. |
| Promotion | `on_promo` | Binary flag. |

If your price and units are already logged, set
`PanelSchema(values_are_log=True)` and no transformation is applied.

### Optional columns

These improve how competitors are chosen. Omit any of them and the model falls
back to learned attention alone.

| Role | Effect |
|---|---|
| `category` | Products only compete inside the same category. |
| `brand` | Same-brand pairs are favoured as competitors. |
| `style` | Same sub-segment pairs are favoured. |
| `size` | Similar pack sizes are favoured. |

### What is derived for you

You never build features by hand. `fit` generates, per product:

- demand lags and their missing-value flags (default: 1, 2 and 4 periods),
- trailing rolling means (default: 4 and 13 periods),
- lifecycle counters since the product first appeared,
- competitive context: neighbour count, promotional share of neighbours,
  lagged neighbour demand, share of newly introduced neighbours, assortment
  size.

And per store-period: a period index, Fourier seasonality (default periods 52,
26 and 13) and promotional intensity.

Every historical feature excludes the current period, so no future information
reaches the model.

## 2. Choosing the products

The model works on a fixed set of `n_products` positions. When
`n_products` is set, ICDN greedily picks the products that share the densest
store-period overlap, then drops any product observed in fewer than
`min_coverage` of the store-period cells. Position `i` maps to
`model.products[i]` and stays stable across predictions and checkpoints.

Cross-price parameters grow with the square of the number of products, so
start around five to ten and grow from there.

## 3. How training works

`fit` runs two phases automatically:

1. **Warm-up.** Demand is smoothed with a trailing moving average, spline
   weights are frozen and the linear price coefficient starts at a negative
   prior. This anchors a stable downward-sloping demand curve.
2. **Main.** Splines are released and the raw series are fitted, letting the
   model capture non-linear price response and cross-price effects.

Afterwards the competitor graph is frozen from the average attention scores
over the training data, which makes inference deterministic and cheaper.

Both phases use early stopping on a chronological validation split, so the
epoch settings are upper bounds rather than exact durations.

## 4. Configuration reference

```python
from icdn import ICDNConfig

config = ICDNConfig(
    n_products=5,
    epochs=400,
    batch_size=256,
    lambda_smooth=0.0351,   # curvature penalty
    lambda_elast=0.0445,    # keeps elasticities in a plausible range
    own_elasticity_bounds=(-5.0, 0.0),
    cross_elasticity_bounds=(-1.0, 1.0),
    device="auto",
)
```

Defaults reproduce the configuration selected by the hyperparameter search of
the original study, so they are a reasonable starting point for weekly retail
data. Configurations can also be loaded from YAML with
`ICDNConfig.from_yaml("configs/default.yaml")`.

## 5. Reading the outputs

`elasticities()` reports how the demand of `product` responds to a 1% change in
the price of `competitor`:

- `kind = own` rows are own-price elasticities and should be negative.
- `kind = cross` rows are directional. The response of A to B's price is
  estimated independently from the response of B to A's price, so no symmetry
  is imposed.

Aggregated output summarises each store and product pair across periods with
its mean, standard deviation and a 95% interval. Pass `aggregate=False` for one
row per observation, which is what you want when studying how elasticity moves
with promotions or seasonality.

## 6. Saving and serving

```python
path = model.save("artifacts/icdn")
restored = ICDNModel.load(path)
restored.elasticities(new_panel)
```

A checkpoint holds the weights, the configuration, the panel layout, the
identifier encoders and the frozen competitor graph. It does not store your
data, so pass the panel explicitly after loading.
