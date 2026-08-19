"""Public interface of the ICDN library."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ICDNConfig
from .data.dataset import DataLoaderFactory, MultiProductDataset
from .data.features import FeatureBuilder
from .data.panel import PanelBuilder, PanelLayout
from .data.splits import TemporalSplitter
from .model.context import ProductTokenBuilder
from .model.head import IntegrableDemandHead
from .model.icdn import ICDN
from .model.neighbors import ProductMetadata
from .model.splines import SplineBuilder
from .training.checkpoints import load_checkpoint, save_checkpoint
from .training.metrics import (
    collect_targets,
    predict_demand,
    predict_elasticities,
    regression_metrics,
)
from .training.trainer import Trainer, resolve_device, seed_everything

# Highest log-demand that still converts to a finite level.
_MAX_LOG_DEMAND = 700.0


class ICDNModel:
    """Estimates demand and price elasticities from a retail panel.

    The model learns a smooth log-demand surface conditioned on context and
    reads elasticities off its derivatives, so predictions and elasticities
    always come from the same fitted object.

    Example:
        >>> model = ICDNModel(ICDNConfig(n_products=5))
        >>> model.fit(panel)
        >>> elasticities = model.elasticities()

    Args:
        config: pipeline configuration. Defaults reproduce the reference setup.
    """

    def __init__(self, config: ICDNConfig | None = None):
        self.config = config or ICDNConfig()
        self.layout: PanelLayout | None = None
        self.history: dict = {}
        self._model: ICDN | None = None
        self._panel_builder: PanelBuilder | None = None
        self._device = resolve_device(self.config.device)
        self._train_panel: pd.DataFrame | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def products(self) -> list:
        """Modelled products, in the positional order used by every output."""
        self._require_fitted()
        return list(self.layout.products)

    def fit(self, panel: pd.DataFrame) -> "ICDNModel":
        """Fits the model on a long panel of store, product and period rows.

        The panel only needs identifiers, price, units and a promotional flag.
        Lags, seasonality and competitive context are engineered internally.
        """
        cfg = self.config
        features = FeatureBuilder(cfg)
        long_df = features.run(panel)

        self._panel_builder = PanelBuilder(cfg)
        wide = self._panel_builder.fit_transform(
            long_df, features.shared_features, features.product_features
        )
        self.layout = self._panel_builder.layout

        splitter = TemporalSplitter(period_col=cfg.schema.period)
        train_wide, val_wide = splitter.single_split(wide, train_frac=1.0 - cfg.validation_fraction)
        if val_wide.empty:
            raise ValueError(
                "the panel does not have enough periods to build a validation split. "
                "Provide more history or lower validation_fraction."
            )

        seed_everything(cfg.seed)
        self._model = self._build_model(train_wide)
        loaders = self._build_loaders(train_wide, val_wide)

        trainer = Trainer(cfg)
        self.history = trainer.fit(
            self._model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            warmup_train_loader=loaders["warmup_train"],
            warmup_val_loader=loaders["warmup_val"],
            meta=self.product_metadata(),
        )
        self._train_panel = panel
        return self

    # ── Inference ───────────────────────────────────────────────────────────

    def predict(self, panel: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predicts demand for every store, period and product.

        Returns a long frame with the identifier columns of your schema plus
        ``predicted_demand``, ``predicted_log_demand`` and, where available,
        the observed demand.
        """
        wide, loader = self._prepare(panel)
        y_hat = predict_demand(self._model, loader, self._device, self.product_metadata())
        y_true, mask = collect_targets(loader)

        frame = self._melt(
            wide,
            {"predicted_log_demand": y_hat, "log_demand": y_true, "observed": mask},
        )
        frame["predicted_demand"] = self._to_levels(frame["predicted_log_demand"])
        frame["demand"] = self._to_levels(frame["log_demand"]).where(frame["observed"] > 0)
        frame = frame.drop(columns=["observed"])
        return frame

    def elasticities(
        self,
        panel: pd.DataFrame | None = None,
        aggregate: bool = True,
    ) -> pd.DataFrame:
        """Own- and cross-price elasticities.

        Each row reports how the demand of ``product`` responds to the price of
        ``competitor``. Rows where both coincide are own-price elasticities.

        Args:
            panel: data to evaluate. Defaults to the training panel.
            aggregate: when True, summarises each store and product pair with
                its mean, dispersion and a 95% interval across periods. When
                False, returns one row per observation.
        """
        wide, loader = self._prepare(panel)
        E = predict_elasticities(self._model, loader, self._device, self.product_metadata())
        _, mask = collect_targets(loader)

        rows = self._elasticity_rows(wide, E, mask)
        if not aggregate:
            return rows

        schema = self.config.schema
        grouped = rows.groupby([schema.store, "product", "competitor", "kind"], observed=True)
        summary = grouped["elasticity"].agg(
            elasticity="mean",
            std="std",
            ci_low=lambda s: s.quantile(0.025),
            ci_high=lambda s: s.quantile(0.975),
            n_obs="size",
        )
        return summary.reset_index()

    def evaluate(self, panel: pd.DataFrame | None = None) -> dict[str, float]:
        """Masked MAE, RMSE and R2 of log-demand on the given panel."""
        _, loader = self._prepare(panel)
        y_hat = predict_demand(self._model, loader, self._device, self.product_metadata())
        y_true, mask = collect_targets(loader)
        return regression_metrics(y_true, y_hat, mask)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Writes weights, configuration, layout and encoders to a single file."""
        self._require_fitted()
        return save_checkpoint(path, self._model, self.config, self.layout)

    @classmethod
    def load(cls, path: str | Path) -> "ICDNModel":
        """Restores a model saved with :meth:`save`."""
        payload = load_checkpoint(path)

        model = cls(ICDNConfig.from_dict(payload["config"]))
        model.layout = PanelLayout.from_dict(payload["layout"])
        model._panel_builder = PanelBuilder(model.config)
        model._panel_builder.layout = model.layout

        network = model._instantiate(spline_prices=None)
        selector = network.head.neighbor_selector
        if selector is not None and payload.get("frozen_pairs") is not None:
            selector.set_frozen_graph(payload["frozen_pairs"], payload["frozen_edge_bonus"])
        network.load_state_dict(payload["state_dict"])
        model._model = network.to(model._device)
        return model

    # ── Internals ───────────────────────────────────────────────────────────

    def product_metadata(self) -> ProductMetadata:
        """Static product attributes that bias competitor selection."""
        if self.layout is None:
            raise RuntimeError("the panel layout is unknown, call fit() or load() first")
        layout = self.layout

        def as_long(values):
            return None if values is None else torch.tensor(values, dtype=torch.long)

        return ProductMetadata(
            category=as_long(layout.category_codes),
            brand=as_long(layout.brand_codes),
            style=as_long(layout.style_codes),
            size=None if layout.sizes is None else torch.tensor(layout.sizes, dtype=torch.float32),
        )

    def _build_model(self, train_wide: pd.DataFrame) -> ICDN:
        n = self.layout.n_products
        prices = train_wide[[f"log_price_{i}" for i in range(n)]].to_numpy()
        return self._instantiate(spline_prices=prices).to(self._device)

    def _instantiate(self, spline_prices: np.ndarray | None) -> ICDN:
        cfg, layout = self.config, self.layout
        n = layout.n_products

        # When rebuilding from a checkpoint the knots are placeholders: the
        # saved buffers overwrite them on load_state_dict.
        prices = spline_prices if spline_prices is not None else np.zeros((2, n))
        splines = SplineBuilder().build_basis(prices, n_knots=cfg.n_knots)

        tokens = ProductTokenBuilder(
            n=n,
            n_stores=layout.n_stores,
            n_shared_features=len(layout.shared_features),
            n_product_features=len(layout.product_features),
            d_store=cfg.d_store,
            n_brands=layout.n_brands,
            d_brand=cfg.d_brand,
            n_styles=layout.n_styles,
            d_style=cfg.d_style,
        )
        head = IntegrableDemandHead(
            context_dim=tokens.d_token,
            K_splines=cfg.n_knots,
            n=n,
            k_neighbors=cfg.k_neighbors,
            hidden=cfg.hidden,
            act=cfg.activation,
            dropout=cfg.dropout,
            enforce_negative_beta=True,
            use_cross=cfg.use_cross,
            same_category_first=cfg.same_category_first,
        )
        return ICDN(context_builder=tokens, price_splines=splines, head=head, n=n)

    def _build_loaders(self, train_wide: pd.DataFrame, val_wide: pd.DataFrame) -> dict:
        cfg = self.config
        factory = DataLoaderFactory(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=self._device.type == "cuda",
        )
        period = cfg.schema.period

        def dataset(frame: pd.DataFrame) -> MultiProductDataset:
            return MultiProductDataset(frame, self.layout, period_col=period)

        loaders = {
            "train": factory.train(dataset(train_wide)),
            "val": factory.evaluate(dataset(val_wide)),
            "warmup_train": None,
            "warmup_val": None,
        }

        if cfg.warmup_epochs > 0:
            loaders["warmup_train"] = factory.train(dataset(self._smooth(train_wide)))
            loaders["warmup_val"] = factory.evaluate(dataset(self._smooth(val_wide)))

        return loaders

    def _smooth(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Replaces demand by a trailing moving average for the warm-up phase."""
        cfg = self.config
        smoothed = wide.copy()
        columns = [f"log_demand_{i}" for i in range(self.layout.n_products)]
        grouped = smoothed.groupby(cfg.schema.store, observed=True)[columns]
        smoothed[columns] = grouped.transform(
            lambda s: s.rolling(cfg.smoothing_window, min_periods=1).mean()
        )
        return smoothed

    def _prepare(self, panel: pd.DataFrame | None):
        self._require_fitted()
        cfg = self.config
        panel = self._train_panel if panel is None else panel
        if panel is None:
            raise ValueError(
                "no panel supplied and the training panel is unavailable "
                "(it is not stored inside checkpoints). Pass the data explicitly."
            )

        features = FeatureBuilder(cfg)
        wide = self._panel_builder.transform(features.run(panel))
        dataset = MultiProductDataset(wide, self.layout, period_col=cfg.schema.period)
        factory = DataLoaderFactory(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=self._device.type == "cuda",
        )
        return wide, factory.evaluate(dataset)

    def _melt(self, wide: pd.DataFrame, matrices: dict[str, np.ndarray]) -> pd.DataFrame:
        schema = self.config.schema
        products = self.layout.products
        frames = []
        for i, product in enumerate(products):
            block = wide[[schema.store, schema.period]].copy()
            block["product"] = product
            for name, values in matrices.items():
                block[name] = values[:, i]
            frames.append(block)
        return pd.concat(frames, ignore_index=True).sort_values(
            [schema.store, schema.period, "product"], kind="stable"
        ).reset_index(drop=True)

    def _elasticity_rows(
        self,
        wide: pd.DataFrame,
        E: np.ndarray,
        mask: np.ndarray,
    ) -> pd.DataFrame:
        schema = self.config.schema
        products = self.layout.products
        n = len(products)

        selector = self._model.head.neighbor_selector
        if selector is not None and selector.frozen_pairs is not None:
            pairs = selector.frozen_pairs.cpu().numpy()
            cross = list(zip(pairs[0].tolist(), pairs[1].tolist(), strict=True))
        else:
            cross = [(i, j) for i in range(n) for j in range(n) if i != j]

        entries = [(i, i, "own") for i in range(n)] + [(i, j, "cross") for i, j in cross]

        frames = []
        for i, j, kind in entries:
            observed = (mask[:, i] > 0) & (mask[:, j] > 0)
            if not observed.any():
                continue
            block = wide.loc[observed, [schema.store, schema.period]].copy()
            block["product"] = products[i]
            block["competitor"] = products[j]
            block["kind"] = kind
            block["elasticity"] = E[observed, i, j]
            frames.append(block)

        if not frames:
            return pd.DataFrame(
                columns=[schema.store, schema.period, "product", "competitor", "kind", "elasticity"]
            )
        return pd.concat(frames, ignore_index=True)

    def _to_levels(self, log_values: pd.Series) -> pd.Series:
        if self.config.schema.values_are_log:
            return log_values
        # Clipping keeps the column finite and non-negative even when an
        # undertrained model emits extreme values.
        levels = np.expm1(log_values.astype("float64").clip(upper=_MAX_LOG_DEMAND))
        return levels.clip(lower=0.0)

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("the model is not fitted yet, call fit() or load() first")
