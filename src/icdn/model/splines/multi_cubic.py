"""Vectorized cubic spline basis shared by all products."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCubicSplineBasis(nn.Module):
    """Cubic truncated-power spline basis evaluated for n products at once.

    Every product shares the number of knots K but keeps its own knot
    locations and price normalization, so all products are evaluated in a
    single batched operation instead of a per-product loop.

    Basis definition for product i and knot k::

        xs      = (x - shift_i) / scale_i
        u_ik    = relu(xs - knot_ik)
        B_ik    = u_ik ** 3

    Derivatives are corrected by the chain rule so they are expressed with
    respect to the original (un-normalized) log-price.

    Shapes:
        x    : (B, n)     log-price per observation and product
        Bx   : (B, n, K)  basis
        dBx  : (B, n, K)  first derivative, drives the elasticity
        ddBx : (B, n, K)  second derivative, drives the smoothness penalty
    """

    def __init__(
        self,
        knots: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        if knots.ndim != 2:
            raise ValueError(f"knots must have shape (n, K), got {tuple(knots.shape)}")
        n, _ = knots.shape

        shift = shift.reshape(n).float()
        scale = scale.reshape(n).float()
        if torch.any(scale <= 0):
            raise ValueError("scale must be strictly positive for every product")

        # Pre-scale the knots so forward() is a single broadcast operation.
        knots_scaled = (knots.float() - shift[:, None]) / scale[:, None]

        self.register_buffer("knots", knots_scaled)
        self.register_buffer("shift", shift)
        self.register_buffer("scale", scale)

    @property
    def n(self) -> int:
        return int(self.knots.shape[0])

    @property
    def K(self) -> int:
        return int(self.knots.shape[1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (B, n), got {tuple(x.shape)}")
        if x.shape[1] != self.n:
            raise ValueError(f"x has n={x.shape[1]} but the basis was built for n={self.n}")

        xs = (x.float() - self.shift[None, :]) / self.scale[None, :]
        u = F.relu(xs[:, :, None] - self.knots[None, :, :])

        inv_scale = (1.0 / self.scale)[None, :, None]
        Bx = u**3
        dBx = 3.0 * (u**2) * inv_scale
        ddBx = 6.0 * u * (inv_scale**2)
        return Bx, dBx, ddBx
