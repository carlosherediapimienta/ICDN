"""Demand head: latent context to demand parameters to elasticities."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context import SharedProductEncoder
from .demand import DemandCalculator
from .neighbors import ProductMetadata, SparseNeighborSelector


class DemandParameterHead(nn.Module):
    """Predicts the demand-surface parameters from the latent representation.

    Produced parameters:
        b:          (B, n) intercept per product.
        beta:       (B, n) linear own-price coefficient.
        w:          (B, n, K) own-price spline weights.
        beta_cross: (B, P) linear cross-price coefficient per directed pair.
        w_cross:    (B, P, K) cross-price spline weights per directed pair.
        u:          (B, P, K, K) bilinear interaction tensor per directed pair.

    With ``use_cross=False`` the cross tensors are empty and no cross-price
    parameters are allocated.
    """

    def __init__(
        self,
        hidden_dim: int,
        K_splines: int,
        n: int,
        enforce_negative_beta: bool = False,
        use_cross: bool = True,
    ):
        super().__init__()
        self.n = n
        self.K_splines = K_splines
        self.use_cross = use_cross
        self.enforce_negative_beta = enforce_negative_beta

        self.head_b = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)
        self.head_w = nn.Linear(hidden_dim, K_splines)

        if use_cross:
            self.head_beta_cross = nn.Linear(2 * hidden_dim, 1)
            self.head_w_cross = nn.Linear(2 * hidden_dim, K_splines)
            self.head_cross = nn.Linear(2 * hidden_dim, K_splines * K_splines)
            # Every ordered pair i != j, so both directions are kept.
            pairs = (~torch.eye(n, dtype=torch.bool)).nonzero(as_tuple=False).T
        else:
            pairs = torch.empty(2, 0, dtype=torch.long)

        self.register_buffer("_pairs", pairs)

    def run(self, h: torch.Tensor, pairs: torch.Tensor | None = None, linear_warmup: bool = False) -> dict[str, torch.Tensor]:
        B, _, _ = h.shape
        K = self.K_splines
        active_pairs = pairs if pairs is not None else self._pairs

        b = self.head_b(h).squeeze(-1)
        beta_raw = self.head_beta(h).squeeze(-1)
        # -softplus is always negative, encoding the prior that a price rise
        # lowers demand. The splines can still bend the curve locally, so this
        # constrains the baseline slope rather than the elasticity itself.
        beta = -F.softplus(beta_raw) if self.enforce_negative_beta else beta_raw
        w = (
            torch.zeros(B, self.n, K, device=h.device, dtype=h.dtype)
            if linear_warmup
            else self.head_w(h)
        )

        if self.use_cross:
            i_idx, j_idx = active_pairs[0], active_pairs[1]
            n_active = active_pairs.shape[1]
            h_ij = torch.cat([h[:, i_idx, :], h[:, j_idx, :]], dim=-1)

            beta_cross = self.head_beta_cross(h_ij).squeeze(-1)
            if linear_warmup:
                w_cross = torch.zeros(B, n_active, K, device=h.device, dtype=h.dtype)
                u = torch.zeros(B, n_active, K, K, device=h.device, dtype=h.dtype)
            else: 
                w_cross = self.head_w_cross(h_ij)
                u = self.head_cross(h_ij).view(B, n_active, K, K)
        else:
            beta_cross = torch.empty(B, 0, device=h.device, dtype=h.dtype)
            w_cross = torch.empty(B, 0, K, device=h.device, dtype=h.dtype)
            u = torch.empty(B, 0, K, K, device=h.device, dtype=h.dtype)

        return {
            "b": b,
            "beta": beta,
            "beta_cross": beta_cross,
            "w": w,
            "w_cross": w_cross,
            "u": u,
            "pairs": active_pairs,
        }

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class IntegrableDemandHead(nn.Module):
    """Multiproduct demand head derived from an integrable scalar potential.

    Chains the shared encoder, the sparse competitor selection, the parameter
    head and the demand calculator into a single forward pass.
    """

    def __init__(
        self,
        context_dim: int,
        K_splines: int,
        n: int,
        k_neighbors: int = 2,
        hidden=(256, 128, 64),
        act: str = "tanh",
        dropout: float = 0.0,
        enforce_negative_beta: bool = False,
        use_cross: bool = True,
    ):
        super().__init__()
        self.encoder = SharedProductEncoder(context_dim, hidden=hidden, act=act, dropout=dropout)
        hidden_dim = self.encoder.out_dim

        self.param_head = DemandParameterHead(
            hidden_dim=hidden_dim,
            K_splines=K_splines,
            n=n,
            enforce_negative_beta=enforce_negative_beta,
            use_cross=use_cross,
        )
        self.demand_calc = DemandCalculator()
        self.neighbor_selector = (
            SparseNeighborSelector(d_hidden=hidden_dim, k_neighbors=k_neighbors)
            if use_cross
            else None
        )

    def run(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        dBx: torch.Tensor,
        return_E: bool = False,
        meta: ProductMetadata | None = None,
        linear_warmup: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        h = self.encoder(tokens)

        pairs, attn_weights = None, None
        if self.neighbor_selector is not None:
            pairs, attn_weights = self.neighbor_selector.run(h, meta=meta)

        params = self.param_head.run(h, pairs=pairs, linear_warmup=linear_warmup)

        y_hat, eps_hat, E = self.demand_calc.run(
            b=params["b"],
            beta=params["beta"],
            w=params["w"],
            x=x,
            Bx=Bx,
            dBx=dBx,
            beta_cross=params["beta_cross"],
            w_cross=params["w_cross"],
            u=params["u"],
            pairs=params["pairs"],
            attn_weights=attn_weights,
            return_E=return_E,
        )
        
        params["attn_weights"] = attn_weights
        if return_E and E is not None:
            params["E"] = E
        return y_hat, eps_hat, params

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)
