"""Training objective: demand fit plus economic regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvatureCalculator:
    """Second derivative of log-demand with respect to its own log-price."""

    def run(
        self,
        w: torch.Tensor,
        ddBx: torch.Tensor,
        u: torch.Tensor | None,
        Bx: torch.Tensor,
        pairs: torch.Tensor | None,
        attn_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Kept in float32 for numerical stability under mixed precision.
        w, ddBx = w.float(), ddBx.float()
        B, _, _ = w.shape

        kappa = (w * ddBx).sum(-1)

        no_cross = u is None or pairs is None or u.numel() == 0 or pairs.numel() == 0
        if no_cross:
            return kappa

        i_idx, j_idx = pairs[0], pairs[1]
        contrib = torch.einsum("bpk,bpkl,bpl->bp", ddBx[:, i_idx], u.float(), Bx.float()[:, j_idx])
        if attn_weights is not None:
            contrib = contrib * attn_weights.float()
        i_exp = i_idx.unsqueeze(0).expand(B, -1)
        return kappa.scatter_add(1, i_exp, contrib.to(kappa.dtype))


class SmoothnessPenalty:
    """Mean squared curvature, discouraging wiggly demand curves.

    Higher values push the model toward smoother, more monotone curves and
    reduce overfitting in price regions with little data.
    """

    def __init__(self):
        self.curvature_calc = CurvatureCalculator()

    def run(
        self,
        w: torch.Tensor,
        ddBx: torch.Tensor,
        u: torch.Tensor,
        Bx: torch.Tensor,
        pairs: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kappa = self.curvature_calc.run(w, ddBx, u, Bx, pairs, attn_weights)
        return (kappa**2).mean()


class PositivityPenalty:
    """Soft penalty on positive own-price elasticities (Giffen behaviour)."""

    def run(self, eps_hat: torch.Tensor, obs_mask: torch.Tensor | None = None) -> torch.Tensor:
        penalty = F.relu(eps_hat)
        if obs_mask is not None:
            return (penalty * obs_mask).sum() / obs_mask.sum().clamp(min=1.0)
        return penalty.mean()


class ElasticityLoss(nn.Module):
    """Compound objective ``L_fit + lambda_smooth * L_smooth + lambda_elast * L_elast``.

    ``L_fit`` is a Huber loss over observed log-demands only. ``L_smooth``
    penalizes curvature. ``L_elast`` applies asymmetric squared hinges that keep
    own-price elasticities inside ``[l_own, r_own]`` and cross-price ones inside
    ``[l_cross, r_cross]``.
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        lambda_smooth: float = 0.0,
        lambda_elast: float = 0.0,
        l_own: float = -5.0,
        r_own: float = 0.0,
        l_cross: float = -1.0,
        r_cross: float = 1.0,
        rho_own_low: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.fit_loss = nn.HuberLoss(delta=float(huber_delta), reduction=reduction)
        self.smoothness_penalty = SmoothnessPenalty()
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_elast = float(lambda_elast)
        self.l_own, self.r_own = float(l_own), float(r_own)
        self.l_cross, self.r_cross = float(l_cross), float(r_cross)
        self.rho_own_low = float(rho_own_low)

    def run(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        obs_mask: torch.Tensor,
        w: torch.Tensor,
        ddBx: torch.Tensor,
        u: torch.Tensor,
        Bx: torch.Tensor,
        pairs: torch.Tensor,
        E: torch.Tensor | None = None,
        attn_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = obs_mask.bool()
        loss_fit = self.fit_loss(y_hat[mask], y_true[mask]) if mask.any() else y_hat.new_tensor(0.0)

        if self.lambda_smooth > 0.0:
            loss_smooth = self.smoothness_penalty.run(w, ddBx, u, Bx, pairs, attn_weights)
        else:
            loss_smooth = y_hat.new_tensor(0.0)

        diag = torch.arange(E.shape[1], device=E.device) if E is not None else None

        if self.lambda_elast > 0.0 and E is not None:
            _, n, _ = E.shape

            bounds_low = E.new_full((n, n), self.l_cross)
            bounds_high = E.new_full((n, n), self.r_cross)
            rho = E.new_ones(n, n)
            bounds_low[diag, diag] = self.l_own
            bounds_high[diag, diag] = self.r_own
            rho[diag, diag] = self.rho_own_low

            # Entries outside the sparse graph stay at zero in E, so they never
            # contribute to the penalty regardless of the mask.
            m = obs_mask.float()
            M = m.unsqueeze(2) * m.unsqueeze(1)

            upper_viol = F.relu(E - bounds_high.unsqueeze(0)) ** 2
            lower_viol = F.relu(bounds_low.unsqueeze(0) - E) ** 2
            penalty = M * (upper_viol + rho.unsqueeze(0) * lower_viol)
            loss_elast = penalty.sum() / M.sum().clamp(min=1.0)
        else:
            loss_elast = y_hat.new_tensor(0.0)

        loss = loss_fit + self.lambda_smooth * loss_smooth + self.lambda_elast * loss_elast

        eps_hat = E[:, diag, diag].detach() if E is not None else y_hat.new_zeros(y_hat.shape)
        logs = {
            "loss": loss.detach(),
            "loss_fit": loss_fit.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_elast": loss_elast.detach(),
            "eps_mean": eps_hat.mean(),
            "eps_p50": eps_hat.median(),
            "obs_frac": obs_mask.mean().detach(),
        }
        return loss, logs

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)
