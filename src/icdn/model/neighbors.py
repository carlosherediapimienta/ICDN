"""Sparse selection of the directed competitor graph used for cross-price terms."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProductMetadata:
    """Optional product attributes that bias competitor selection.

    All fields are indexed by product position and every one of them is
    optional. When a field is missing the corresponding bias is simply not
    applied, so the selector falls back to pure attention.

    Attributes:
        category: (n,) long. Products only compete inside the same category.
        brand:    (n,) long. Same-brand pairs receive an additive bonus.
        style:    (n,) long. Same-style pairs receive an additive bonus.
        size:     (n,) float. Similar pack sizes receive an additive bonus.
    """

    category: torch.Tensor | None = None
    brand: torch.Tensor | None = None
    style: torch.Tensor | None = None
    size: torch.Tensor | None = None

    def is_empty(self) -> bool:
        return all(f is None for f in (self.category, self.brand, self.style, self.size))

    def to(self, device: torch.device) -> "ProductMetadata":
        def move(tensor):
            return None if tensor is None else tensor.to(device)

        return ProductMetadata(
            category=move(self.category),
            brand=move(self.brand),
            style=move(self.style),
            size=move(self.size),
        )


class SparseNeighborSelector(nn.Module):
    """Builds a sparse directed competitor graph and weights its edges.

    The selection happens in two stages:

    1. Structural candidates shared by the whole batch. Products are ranked by
       aggregated attention scores plus a metadata bonus, and the top ``k``
       competitors per focal product are kept.
    2. Per-sample soft weights. A softmax over each focal product's candidates
       gives heterogeneous edge strengths per observation without breaking
       vectorization.

    Args:
        d_hidden: dimension of the latent vectors produced by the encoder.
        d_attn: dimension of the query and key projections.
        k_neighbors: competitors kept per focal product.
        init_brand_bonus: initial additive bonus for same-brand pairs.
        init_style_bonus: initial additive bonus for same-style pairs.
        init_size_bonus: initial additive bonus for similar-size pairs.
        size_gamma: decay rate of the size similarity in log space.
        same_category_first: prefer same-category candidates when the category
            is available.
    """

    def __init__(
        self,
        d_hidden: int,
        d_attn: int = 16,
        k_neighbors: int = 2,
        init_brand_bonus: float = 0.30,
        init_style_bonus: float = 0.10,
        init_size_bonus: float = 0.10,
        size_gamma: float = 1.0,
        same_category_first: bool = False,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_hidden, d_attn, bias=False)
        self.k_proj = nn.Linear(d_hidden, d_attn, bias=False)
        self.scale = math.sqrt(d_attn)

        self.k = k_neighbors
        self.gamma = size_gamma
        self.same_category_first = same_category_first

        # Effective bonus is softplus(raw), so it stays positive during training.
        self.brand_bonus_raw = nn.Parameter(torch.tensor(_inv_softplus(init_brand_bonus)))
        self.style_bonus_raw = nn.Parameter(torch.tensor(_inv_softplus(init_style_bonus)))
        self.size_bonus_raw = nn.Parameter(torch.tensor(_inv_softplus(init_size_bonus)))

        # Populated by freeze_graph() and carried inside the checkpoint.
        self.register_buffer("frozen_pairs", None)
        self.register_buffer("frozen_edge_bonus", None)

    def _meta_bonus(
        self,
        meta: ProductMetadata | None,
        n: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (not_self, same_category, bonus), each with shape (n, n)."""
        not_self = ~torch.eye(n, dtype=torch.bool, device=device)
        bonus = torch.zeros(n, n, device=device)

        if meta is None or meta.is_empty():
            # Without a category every product competes with every other one.
            return not_self, torch.ones(n, n, dtype=torch.bool, device=device), bonus

        meta = meta.to(device)

        if meta.category is not None:
            same_cat = meta.category.unsqueeze(0) == meta.category.unsqueeze(1)
        else:
            same_cat = torch.ones(n, n, dtype=torch.bool, device=device)

        if meta.brand is not None:
            same_brand = (meta.brand.unsqueeze(0) == meta.brand.unsqueeze(1)).float()
            bonus = bonus + F.softplus(self.brand_bonus_raw) * same_brand

        if meta.style is not None:
            same_style = (meta.style.unsqueeze(0) == meta.style.unsqueeze(1)).float()
            bonus = bonus + F.softplus(self.style_bonus_raw) * same_style

        if meta.size is not None:
            # Similarity decays with the distance between sizes in log space.
            log_size = torch.log(meta.size.float() + 1e-6)
            log_dist = (log_size.unsqueeze(0) - log_size.unsqueeze(1)).abs()
            bonus = bonus + F.softplus(self.size_bonus_raw) * torch.exp(-self.gamma * log_dist)

        return not_self, same_cat, bonus.masked_fill(~not_self, 0.0)

    @torch.no_grad()
    def accumulate_mean_scores(self, h_iter, meta: ProductMetadata | None = None) -> torch.Tensor:
        """Averages the (n, n) score matrix over every batch yielded by ``h_iter``.

        The caller is responsible for producing the latent vectors in eval mode
        and without gradients. The result feeds ``freeze_graph``.
        """
        device = next(self.parameters()).device

        acc = None
        count = 0
        for h in h_iter:
            h = h.to(device)
            n = h.shape[1]
            not_self, _, bonus = self._meta_bonus(meta, n, device)
            Q, K = self.q_proj(h), self.k_proj(h)
            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
            scores = scores + bonus.unsqueeze(0)
            scores = scores.masked_fill(~not_self.unsqueeze(0), float("-inf"))
            batch_mean = scores.mean(dim=0)
            acc = batch_mean if acc is None else acc + batch_mean
            count += 1

        if acc is None or count == 0:
            raise RuntimeError("accumulate_mean_scores received an empty iterator")
        return acc / count

    def freeze_graph(
        self,
        global_mean_scores: torch.Tensor,
        meta: ProductMetadata | None = None,
    ) -> None:
        """Fixes the competitor graph and precomputes its metadata bonus.

        Once frozen, ``run`` uses a sparse path costing O(B * E * d_attn)
        instead of materializing the dense (B, n, n) score matrix. Call it once
        after training converges and before evaluating or serving.
        """
        device = global_mean_scores.device
        n = global_mean_scores.shape[0]
        not_self, same_cat, bonus = self._meta_bonus(meta, n, device)
        pairs = self._build_pairs(global_mean_scores, not_self, same_cat)
        self.set_frozen_graph(pairs, bonus[pairs[0], pairs[1]])

    def set_frozen_graph(self, pairs: torch.Tensor, edge_bonus: torch.Tensor) -> None:
        """Restores a previously frozen graph, e.g. when loading a checkpoint."""
        self.frozen_pairs = pairs
        self.frozen_edge_bonus = edge_bonus

    def _build_pairs(
        self,
        mean_scores: torch.Tensor,
        not_self: torch.Tensor,
        same_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Selects the top-k directed edges per focal product, shape (2, n * k)."""
        n = mean_scores.shape[0]
        k_eff = min(self.k, n - 1)
        device = mean_scores.device

        if k_eff == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        # A large additive constant guarantees that same-category candidates
        # outrank any fallback. Rows with too few of them fill the remaining
        # slots with the best available candidates, with no branching needed.
        if self.same_category_first:
            priority_bias = (same_cat & not_self).float() * 1e6
        else:
            priority_bias = torch.zeros(n, n, device=device)

        biased = (mean_scores + priority_bias).masked_fill(~not_self, float("-inf"))
        _, top_j = torch.topk(biased, k_eff, dim=1)

        i_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, k_eff).reshape(-1)
        return torch.stack([i_idx, top_j.reshape(-1)], dim=0)

    def run(
        self,
        h: torch.Tensor,
        meta: ProductMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the directed edges (2, E) and their per-sample weights (B, E)."""
        B, n, _ = h.shape
        device = h.device

        if n <= 1:
            return (
                torch.empty(2, 0, dtype=torch.long, device=device),
                torch.empty(B, 0, dtype=h.dtype, device=device),
            )

        Q, K = self.q_proj(h), self.k_proj(h)

        if self.frozen_pairs is not None:
            pairs = self.frozen_pairs
            i_idx, j_idx = pairs[0], pairs[1]
            n_edges = i_idx.numel()
            k_eff = n_edges // n
            edge_logits = (Q[:, i_idx, :] * K[:, j_idx, :]).sum(-1) / self.scale
            edge_logits = edge_logits + self.frozen_edge_bonus.unsqueeze(0)
            edge_weights = F.softmax(edge_logits.view(B, n, k_eff), dim=-1).reshape(B, n_edges)
            return pairs, edge_weights

        not_self, same_cat, bonus = self._meta_bonus(meta, n, device)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        scores = scores + bonus.unsqueeze(0)
        scores = scores.masked_fill(~not_self.unsqueeze(0), float("-inf"))

        pairs = self._build_pairs(scores.mean(dim=0), not_self, same_cat)
        if pairs.numel() == 0:
            return pairs, torch.empty(B, 0, dtype=h.dtype, device=device)

        i_idx, j_idx = pairs[0], pairs[1]
        n_edges = i_idx.numel()
        k_eff = n_edges // n
        edge_logits = scores[:, i_idx, j_idx].view(B, n, k_eff)
        edge_weights = F.softmax(edge_logits, dim=-1).reshape(B, n_edges)
        return pairs, edge_weights

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def _inv_softplus(x: float) -> float:
    return torch.log(torch.expm1(torch.tensor(float(x)))).item()
