"""Assembly of log-demand and elasticities from the predicted parameters."""

import torch


class DemandCalculator:
    """Turns demand parameters into predicted log-demand and elasticities.

    Own-price terms come from a linear coefficient plus a spline expansion.
    Cross-price terms are added only for the directed pairs selected by the
    neighbor graph, weighted by the attention of each edge.
    """

    def run(
        self,
        b: torch.Tensor,
        beta: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        dBx: torch.Tensor,
        beta_cross: torch.Tensor,
        w_cross: torch.Tensor,
        u: torch.Tensor,
        pairs: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
        return_E: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Computes demand and elasticities.

        Returns:
            y_hat: (B, n) predicted log-demand.
            eps_hat: (B, n) own-price elasticity, the diagonal of E.
            E: (B, n, n) full elasticity matrix, or None when return_E is False.
        """
        B, n, _ = Bx.shape

        y_hat = b + beta * x + (w * Bx).sum(dim=-1)
        eps_hat = beta + (w * dBx).sum(dim=-1)

        has_cross = (
            u is not None
            and pairs is not None
            and pairs.numel() > 0
            and u.numel() > 0
        )
        if not has_cross:
            E = None
            if return_E:
                E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)
                diag = torch.arange(n, device=Bx.device)
                E[:, diag, diag] = eps_hat
            return y_hat, eps_hat, E

        i_idx, j_idx = pairs[0], pairs[1]
        Bx_i, Bx_j = Bx[:, i_idx, :], Bx[:, j_idx, :]
        dBx_i, dBx_j = dBx[:, i_idx, :], dBx[:, j_idx, :]
        x_j = x[:, j_idx]

        # g_i gains beta_ij * x_j + w_ij' B_j(x_j) + B_i(x_i)' U_ij B_j(x_j).
        contrib_y = (
            beta_cross * x_j
            + (w_cross * Bx_j).sum(dim=-1)
            + torch.einsum("bpk,bpkl,bpl->bp", Bx_i, u, Bx_j)
        ).to(Bx.dtype)

        # Only the bilinear term depends on x_i, so it is the sole cross
        # contribution to the own-price elasticity.
        contrib_eps = torch.einsum("bpk,bpkl,bpl->bp", dBx_i, u, Bx_j).to(Bx.dtype)

        if attn_weights is not None:
            contrib_y = contrib_y * attn_weights.to(Bx.dtype)
            contrib_eps = contrib_eps * attn_weights.to(Bx.dtype)

        # Each pair contributes to its focal product i, so contributions are
        # scattered back from the edge axis onto the product axis.
        i_exp = i_idx.unsqueeze(0).expand(B, -1)
        y_hat = y_hat.scatter_add(1, i_exp, contrib_y)
        eps_hat = eps_hat.scatter_add(1, i_exp, contrib_eps)

        E = None
        if return_E:
            E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)
            diag = torch.arange(n, device=Bx.device)
            E[:, diag, diag] = eps_hat

            E_cross = (
                beta_cross
                + (w_cross * dBx_j).sum(dim=-1)
                + torch.einsum("bpk,bpkl,bpl->bp", Bx_i, u, dBx_j)
            ).to(E.dtype)
            if attn_weights is not None:
                E_cross = E_cross * attn_weights.to(E.dtype)

            # Pairs are directed: E_ij and E_ji are learned independently and
            # no symmetry is imposed.
            E[:, i_idx, j_idx] = E_cross

        return y_hat, eps_hat, E
