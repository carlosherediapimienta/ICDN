"""Selection of the modelled products and reshaping into the wide panel."""

from dataclasses import dataclass

import pandas as pd

from ..config import ICDNConfig
from .encoders import LabelEncoder
from .features import LOG_DEMAND, LOG_PRICE

STORE_INDEX = "store_index"


@dataclass
class PanelLayout:
    """Fixed description of the modelled panel, shared by training and inference.

    Product positions are what tie everything together: position ``i`` always
    refers to ``products[i]``, both in the wide columns and in the elasticity
    matrix.
    """

    products: list
    shared_features: list[str]
    product_features: list[str]
    store_encoder: LabelEncoder
    brand_codes: list[int] | None = None
    style_codes: list[int] | None = None
    category_codes: list[int] | None = None
    sizes: list[float] | None = None
    n_brands: int = 0
    n_styles: int = 0

    @property
    def n_products(self) -> int:
        return len(self.products)

    @property
    def n_stores(self) -> int:
        return self.store_encoder.size

    def to_dict(self) -> dict:
        payload = {
            "products": list(self.products),
            "shared_features": list(self.shared_features),
            "product_features": list(self.product_features),
            "store_encoder": self.store_encoder.to_dict(),
            "brand_codes": self.brand_codes,
            "style_codes": self.style_codes,
            "category_codes": self.category_codes,
            "sizes": self.sizes,
            "n_brands": self.n_brands,
            "n_styles": self.n_styles,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "PanelLayout":
        payload = dict(payload)
        payload["store_encoder"] = LabelEncoder.from_dict(payload["store_encoder"])
        return cls(**payload)


class PanelBuilder:
    """Turns the engineered long panel into the wide matrix the model consumes.

    Fitting selects the products that share the densest store-period overlap,
    drops sparse ones and freezes the layout. Transforming reshapes any panel
    to that layout, so training and inference always align.
    """

    def __init__(self, config: ICDNConfig):
        self.config = config
        self.schema = config.schema
        self.layout: PanelLayout | None = None
        self._price_fallback_mean: pd.Series | None = None

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        shared_features: list[str],
        product_features: list[str],
    ) -> "PanelBuilder":
        products = self._select_products(df)
        filtered = self._filter_sparse(df, products)
        products = [p for p in products if p in set(filtered[self.schema.product])]
        if len(products) < 2:
            raise ValueError(
                "fewer than two products survived the density filter. "
                "Lower min_coverage or provide a denser panel."
            )

        store_encoder = LabelEncoder().fit(filtered[self.schema.store])
        layout = PanelLayout(
            products=products,
            shared_features=list(shared_features),
            product_features=list(product_features),
            store_encoder=store_encoder,
        )
        self._attach_metadata(filtered, layout)
        self.layout = layout
        price_pivot = self._pivot(filtered, LOG_PRICE, products)
        self._price_fallback_mean = price_pivot.mean()
        return self

    def _select_products(self, df: pd.DataFrame) -> list:
        product, store, period = self.schema.product, self.schema.store, self.schema.period
        available = df[product].unique().tolist()

        if self.config.n_products is None or self.config.n_products >= len(available):
            return available

        # Greedily grow the set of products that keeps the largest common
        # footprint of store-period observations.
        coverage = {
            code: set(zip(group[store], group[period], strict=True))
            for code, group in df.groupby(product, observed=True)
        }
        first = max(coverage, key=lambda c: len(coverage[c]))
        selected = [first]
        intersection = coverage[first]

        while len(selected) < self.config.n_products:
            remaining = [c for c in coverage if c not in selected]
            if not remaining:
                break
            best = max(remaining, key=lambda c: len(intersection & coverage[c]))
            selected.append(best)
            intersection &= coverage[best]

        return selected

    def _filter_sparse(self, df: pd.DataFrame, products: list) -> pd.DataFrame:
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        df = df[df[product].isin(products)].copy()

        total_store_periods = df[[store, period]].drop_duplicates().shape[0]
        coverage = (
            df.groupby(product)[[store, period]].apply(lambda g: g.drop_duplicates().shape[0])
            / max(total_store_periods, 1)
        )
        kept = set(coverage[coverage >= self.config.min_coverage].index)
        df = df[df[product].isin(kept)].copy()

        if self.config.min_products is not None:
            counts = df.groupby([store, period])[product].nunique()
            dense = counts[counts >= self.config.min_products].index
            df = (
                df.set_index([store, period])
                .loc[lambda d: d.index.isin(dense)]
                .reset_index()
            )
        return df

    def _attach_metadata(self, df: pd.DataFrame, layout: PanelLayout) -> None:
        """Freezes the static attributes of each product position."""
        product = self.schema.product
        static = df.drop_duplicates(subset=[product]).set_index(product)

        def codes_for(column: str | None, reserve_zero: bool) -> tuple[list[int] | None, int]:
            if column is None or column not in static.columns:
                return None, 0
            encoder = LabelEncoder(reserve_zero=reserve_zero).fit(static[column])
            values = static.loc[layout.products, column]
            return encoder.transform(values, strict=False).tolist(), encoder.size

        layout.brand_codes, layout.n_brands = codes_for(self.schema.brand, reserve_zero=True)
        layout.style_codes, layout.n_styles = codes_for(self.schema.style, reserve_zero=True)
        layout.category_codes, _ = codes_for(self.schema.category, reserve_zero=False)

        if self.schema.size is not None and self.schema.size in static.columns:
            sizes = pd.to_numeric(static.loc[layout.products, self.schema.size], errors="coerce")
            layout.sizes = sizes.fillna(1.0).astype(float).tolist()

    # ── Transform ───────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.layout is None:
            raise RuntimeError("PanelBuilder.transform called before fit")

        layout = self.layout
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        products = layout.products
        n = len(products)

        df = df[df[product].isin(products)].copy()
        if df.empty:
            raise ValueError("none of the modelled products appear in this panel")

        price = self._pivot(df, LOG_PRICE, products)
        # Prices must exist for every position, so gaps are carried forward and
        # backward inside each store before falling back to the column mean.
        price = price.groupby(level=0, group_keys=False).apply(lambda g: g.ffill().bfill())
        fallback = self._price_fallback_mean if self._price_fallback_mean is not None else price.mean()
        price = price.fillna(fallback)
        price.columns = [f"log_price_{i}" for i in range(n)]

        demand = self._pivot(df, LOG_DEMAND, products)
        obs_mask = demand.notna().astype(float)
        obs_mask.columns = [f"obs_mask_{i}" for i in range(n)]
        demand = demand.fillna(0.0)
        demand.columns = [f"log_demand_{i}" for i in range(n)]

        shared = df.groupby([store, period], observed=True)[layout.shared_features].first()

        blocks = [demand, obs_mask, shared]
        for feature in layout.product_features:
            block = self._pivot(df, feature, products).fillna(0.0)
            block.columns = [f"{feature}_{i}" for i in range(n)]
            blocks.append(block)

        wide = price.join(blocks, how="left").reset_index()
        wide[STORE_INDEX] = layout.store_encoder.transform(wide[store])
        return wide.sort_values([store, period]).reset_index(drop=True)

    def fit_transform(
        self,
        df: pd.DataFrame,
        shared_features: list[str],
        product_features: list[str],
    ) -> pd.DataFrame:
        return self.fit(df, shared_features, product_features).transform(df)

    def _pivot(self, df: pd.DataFrame, value: str, products: list) -> pd.DataFrame:
        return (
            df.pivot_table(
                index=[self.schema.store, self.schema.period],
                columns=self.schema.product,
                values=value,
                aggfunc="mean",
                observed=True,
            )
            .reindex(columns=products)
            .sort_index()
        )
