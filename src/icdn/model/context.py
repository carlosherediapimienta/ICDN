"""Context encoding: raw batch features to per-product latent representations."""

import torch
import torch.nn as nn


class ProductTokenBuilder(nn.Module):
    """Builds one context token per product from a batch.

    Ech token concatenates features shared by the whole observation (store
    embedding, seasonality, promotional pressure) and features specific to the
    product (lags, rolling means, competitive context, brand and style
    embeddings):
        token_i = [
            store_emb |
            shared |
            brand_emb_i |
            style_emb_i |
            product_feats_i
        ]
    The number of shared and per-product features is passed explicitly so the
    module stays independent of any particular column list. 

    Recall that the hetereogeniety of the product is captured by:
     Heterogeneity = specific features + specific state + specific spline basis
    """

    def __init__(
        self,
        n: int,
        n_stores: int,
        n_shared_features: int,
        n_product_features: int,
        d_store: int = 16,
        n_brands: int = 1,
        d_brand: int = 8,
        n_styles: int = 1,
        d_style: int = 8,
    ):
        super().__init__()
        self.n = n
        self.n_shared_features = n_shared_features
        self.n_product_features = n_product_features

        self.emb_store = nn.Embedding(n_stores, d_store)
        # Code 0 is reserved for unknown brands and styles.
        self.emb_brand = nn.Embedding(max(n_brands, 1), d_brand, padding_idx=0)
        self.emb_style = nn.Embedding(max(n_styles, 1), d_style, padding_idx=0)

    @property
    def d_token(self) -> int:
        return (
            self.emb_store.embedding_dim
            + self.n_shared_features
            + self.emb_brand.embedding_dim
            + self.emb_style.embedding_dim
            + self.n_product_features
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns the context tokens with shape (B, n, d_token)."""
        B = batch["ids"].shape[0]

        store_emb = self.emb_store(batch["ids"][:, 0])
        shared = torch.cat([store_emb, batch["shared_feats"]], dim=1)
        shared = shared.unsqueeze(1).expand(-1, self.n, -1)

        brand_emb = self.emb_brand(batch["product_cat"][:, :, 0])
        style_emb = self.emb_style(batch["product_cat"][:, :, 1])

        return torch.cat(
            [shared, brand_emb, style_emb, batch["product_feats"]],
            dim=-1
        )


class SharedProductEncoder(nn.Module):
    """MLP applied independently to every product token with shared weights.

    Learns non-linear feature combinations inside a token but does not model
    product-to-product interactions, which are handled downstream by the
    neighbor selector. Smooth activations are used instead of ReLU to keep the
    demand surface differentiable everywhere.
    """

    ACTIVATIONS = {"tanh": nn.Tanh, "softplus": nn.Softplus, "gelu": nn.GELU}

    def __init__(self, d_in: int, hidden=(256, 128, 64), act: str = "tanh", dropout: float = 0.0):
        super().__init__()
        if act not in self.ACTIVATIONS:
            raise ValueError(
                f"activation '{act}' not supported, use one of {list(self.ACTIVATIONS)}"
            )

        layers: list[nn.Module] = []
        prev = d_in
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(self.ACTIVATIONS[act]())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width

        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Maps (B, n, d_token) tokens to (B, n, out_dim) latent vectors."""
        B, n, d_token = tokens.shape
        h = self.net(tokens.reshape(B * n, d_token))
        return h.view(B, n, self.out_dim)
