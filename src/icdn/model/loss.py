"""Training objective: demand fit plus economic regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Classes

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
        obs_mask: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kappa = self.curvature_calc.run(w, ddBx, u, Bx, pairs, attn_weights)
        return _mean_per_observation(kappa**2, obs_mask)

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
    ):
        super().__init__()
        self.fit_loss = nn.HuberLoss(delta=float(huber_delta), reduction="none")
        self.smoothness_penalty = SmoothnessPenalty()
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_elast = float(lambda_elast)
        self.l_own, self.r_own = float(l_own), float(r_own)
        self.l_cross, self.r_cross = float(l_cross), float(r_cross)

    @staticmethod
    def _sparse_elasticity_penalty(
        own: torch.Tensor, # (B, n)
        cross: torch.Tensor | None, # (B, P), P = n * k
        pairs: torch.Tensor, # (2, P)
        obs_mask: torch.Tensor, # (B, n)
        own_bounds: tuple[float, float],
        cross_bounds: tuple[float, float],
    ) -> torch.Tensor:
        
        mask = obs_mask.to(dtype=own.dtype)
        own_mask = mask
        
        own_lo, own_hi = own_bounds
        own_penalty = (
            F.relu(own - own_hi).square()
            + F.relu(own_lo - own).square()
        )

        if cross is None or pairs is None or pairs.numel() == 0:
            count = own_mask.sum(dim=1)
            total = (own_penalty * own_mask).sum(dim=1)
        else:
            cross_lo, cross_hi = cross_bounds 

            i_idx, j_idx = pairs[0], pairs[1]
            cross_mask = mask[:, i_idx] * mask[:, j_idx]
            
            cross_penalty = (
                F.relu(cross - cross_hi).square()
                + F.relu(cross_lo - cross).square()
            )

            count = own_mask.sum(dim=1) + cross_mask.sum(dim=1)
            total = (
                (own_penalty * own_mask).sum(dim=1)
                + (cross_penalty * cross_mask).sum(dim=1)
            )

        per_observation = total / count.clamp_min(min=1.0)
        valid = count > 0 

        return (
            per_observation[valid].mean()
            if valid.any()
            else own.new_tensor(0.0)
        )

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
        own_elasticity: torch.Tensor | None = None,
        cross_elasticity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        huber = self.fit_loss(y_hat, y_true)
        loss_fit = _mean_per_observation(huber, obs_mask)

        if self.lambda_smooth > 0.0:
            loss_smooth = self.smoothness_penalty.run(
                w, ddBx, u, Bx, pairs, obs_mask, attn_weights
            )
        else:
            loss_smooth = y_hat.new_tensor(0.0)

        diag = torch.arange(E.shape[1], device=E.device) if E is not None else None

        if self.lambda_elast > 0.0:
            if own_elasticity is not None:
                loss_elast = self._sparse_elasticity_penalty(
                    own=own_elasticity, cross=cross_elasticity, pairs=pairs, obs_mask=obs_mask,
                    own_bounds=(self.l_own, self.r_own), cross_bounds=(self.l_cross, self.r_cross)
                )
            elif E is not None:
                _, n, _ = E.shape

                bounds_low = E.new_full((n, n), self.l_cross)
                bounds_high = E.new_full((n, n), self.r_cross)
                bounds_low[diag, diag] = self.l_own
                bounds_high[diag, diag] = self.r_own

                # Normalize over observed diagonal and active edges.
                m = obs_mask.float()
                M = m.unsqueeze(2) * m.unsqueeze(1)
                
                active = torch.eye(n, device=E.device, dtype=E.dtype)
                if pairs is not None and pairs.numel() > 0:
                    active[pairs[0], pairs[1]] = 1.0
                active_mask = M * active.unsqueeze(0)

                upper_viol = F.relu(E - bounds_high.unsqueeze(0)) ** 2
                lower_viol = F.relu(bounds_low.unsqueeze(0) - E) ** 2
                penalty = upper_viol + lower_viol
                loss_elast = _mean_per_observation(penalty, active_mask)
            else:
                loss_elast = y_hat.new_tensor(0.0)
        else:
            loss_elast = y_hat.new_tensor(0.0)

        loss = loss_fit + self.lambda_smooth * loss_smooth + self.lambda_elast * loss_elast

        own_elasticity_detached = own_elasticity.detach() if own_elasticity is not None else(
            E[:, diag, diag].detach() if E is not None else y_hat.new_zeros(y_hat.shape)
        )
        logs = {
            "loss": loss.detach(),
            "loss_fit": loss_fit.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_elast": loss_elast.detach(),
            "eps_mean": own_elasticity_detached.mean(),
            "eps_p50": own_elasticity_detached.median(),
            "obs_frac": obs_mask.mean().detach(),
        }
        return loss, logs

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)



# --- Functions 

def _mean_per_observation(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean within each store-week, then across store-weeks with support.

    ``values`` and ``mask`` share shape ``(B,...)``. Store-weeks whose mask
    sums to zero are dropped so they do not contribute a dummy 0.
    """
    m = mask.float()
    dims = tuple(range(1, values.ndim))
    count = m.sum(dim=dims)
    per_obs = (values * m).sum(dim=dims) / count.clamp(min=1.0)
    valid = count > 0
    if not valid.any():
        return values.new_tensor(0.0)
    return per_obs[valid].mean()

