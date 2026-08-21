"""Integrable Context-Dependent Demand Network."""

import torch
import torch.nn as nn

from .context import ProductTokenBuilder
from .head import IntegrableDemandHead
from .neighbors import ProductMetadata
from .splines import MultiCubicSplineBasis


class ICDN(nn.Module):
    """Integrable Context-Dependent Demand Network.

    Runs the full forward pass in three steps:

    1. ``ProductTokenBuilder`` encodes store, calendar, promotional and
       competitive context into one token per product.
    2. ``MultiCubicSplineBasis`` evaluates the spline basis and its derivatives
       for every log-price at once.
    3. ``IntegrableDemandHead`` maps tokens and splines to log-demand and
       elasticities.

    Elasticities are exact derivatives of a single fitted log-demand surface,
    which is what makes the estimates derivative-coherent.
    """

    def __init__(
        self,
        context_builder: ProductTokenBuilder,
        price_splines: MultiCubicSplineBasis,
        head: IntegrableDemandHead,
        n: int,
    ):
        super().__init__()
        self.context_builder = context_builder
        self.price_splines = price_splines
        self.head = head
        self.n = n

    def run(
        self,
        batch: dict,
        return_parts: bool = False,
        compute_E: bool = False,
        meta: ProductMetadata | None = None,
        linear_warmup: bool = False,
    ):
        tokens = self.context_builder(batch)
        x = batch["prices"]
        Bx, dBx, ddBx = self.price_splines(x)

        y_hat, eps_hat, aux = self.head.run(
            tokens=tokens,
            x=x,
            Bx=Bx,
            dBx=dBx,
            return_E=compute_E,
            meta=meta,
            linear_warmup=linear_warmup,
        )

        if return_parts:
            aux.update({"tokens": tokens, "Bx": Bx, "dBx": dBx, "ddBx": ddBx})
            return y_hat, eps_hat, aux

        return y_hat, eps_hat

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @torch.no_grad()
    def encode(self, batch: dict) -> torch.Tensor:
        """Latent representation of every product, used to freeze the graph."""
        return self.head.encoder(self.context_builder(batch))
