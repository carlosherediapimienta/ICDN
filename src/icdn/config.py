"""User-facing configuration for the ICDN pipeline."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class PanelSchema:
    """Maps the columns of your panel to the roles the model expects.

    The panel is a long table with one row per store, product and period. Only
    the first six fields are required; the rest enrich competitor selection and
    can be left as None.

    Attributes:
        store: store or point-of-sale identifier.
        product: product identifier.
        period: period identifier, expected to be sortable (e.g. a week index).
        price: unit price in levels (strictly positive). Logged internally.
        units: units sold (non-negative). Transformed with log1p internally.
        promo: binary promotional flag.
        category: products only compete inside the same category.
        brand: brand identifier, biases competitor selection.
        style: sub-segment identifier, biases competitor selection.
        size: pack size, biases competitor selection toward similar formats.
    """

    store: str = "store_code"
    product: str = "product_code"
    period: str = "week_id"
    price: str = "price"
    units: str = "units"
    promo: str = "on_promo"
    category: str | None = None
    brand: str | None = None
    style: str | None = None
    size: str | None = None

    @property
    def required(self) -> list[str]:
        return [self.store, self.product, self.period, self.price, self.units, self.promo]

    @property
    def optional(self) -> dict[str, str]:
        candidates = {
            "category": self.category,
            "brand": self.brand,
            "style": self.style,
            "size": self.size,
        }
        return {role: col for role, col in candidates.items() if col is not None}

    def validate(self, columns) -> None:
        missing = [c for c in self.required if c not in columns]
        if missing:
            raise ValueError(
                f"the panel is missing required columns {missing}. "
                f"Adjust PanelSchema if your columns have different names."
            )


@dataclass
class ICDNConfig:
    """Everything needed to build and train an ICDN model.

    Defaults reproduce the configuration selected by the hyperparameter search
    of the original study, so a plain ``ICDNConfig()`` is a sensible starting
    point for weekly retail panels.
    """

    # ── Data ────────────────────────────────────────────────────────────────
    schema: PanelSchema = field(default_factory=PanelSchema)
    n_products: int | None = None
    lags: tuple[int, ...] = (1, 2, 4)
    rolling_windows: tuple[int, ...] = (4, 13)
    seasonal_periods: tuple[int, ...] = (52, 26, 13)
    min_coverage: float = 0.5
    min_products: int | None = None
    new_product_periods: int = 13

    # ── Architecture ────────────────────────────────────────────────────────
    n_knots: int = 3
    hidden: tuple[int, ...] = (256, 128, 64)
    activation: str = "gelu"
    dropout: float = 0.2547
    k_neighbors: int = 4
    d_store: int = 16
    d_brand: int = 8
    d_style: int = 8
    use_cross: bool = True
    same_category_first: bool = False

    # ── Objective ───────────────────────────────────────────────────────────
    huber_delta: float = 1.0
    lambda_smooth: float = 0.0351
    lambda_elast: float = 0.0445
    own_elasticity_bounds: tuple[float, float] = (-5.0, 0.0)
    cross_elasticity_bounds: tuple[float, float] = (-1.0, 1.0)

    # ── Training ────────────────────────────────────────────────────────────
    batch_size: int = 256
    warmup_epochs: int = 350
    epochs: int = 400
    warmup_lr: float = 1.6856e-3
    lr: float = 1.6246e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lr_patience: int = 30
    early_stopping_patience: int = 70
    smoothing_window: int = 8
    beta_prior: float = -2.0
    validation_fraction: float = 0.2
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    verbose: bool = True

    def __post_init__(self):
        if isinstance(self.schema, dict):
            self.schema = PanelSchema(**self.schema)
        for name in ("lags", "rolling_windows", "seasonal_periods", "hidden"):
            setattr(self, name, tuple(getattr(self, name)))
        for name in ("own_elasticity_bounds", "cross_elasticity_bounds"):
            setattr(self, name, tuple(getattr(self, name)))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ICDNConfig":
        import yaml

        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ICDNConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
