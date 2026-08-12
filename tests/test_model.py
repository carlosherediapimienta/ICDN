import torch

from icdn.model import (
    ICDN,
    IntegrableDemandHead,
    ProductMetadata,
    ProductTokenBuilder,
    SplineBuilder,
)


def build_network(n=4, n_knots=3, n_shared=5, n_product=6, n_stores=2):
    torch.manual_seed(0)
    prices = torch.randn(50, n).numpy()
    splines = SplineBuilder().build_basis(prices, n_knots=n_knots)
    tokens = ProductTokenBuilder(
        n=n,
        n_stores=n_stores,
        n_shared_features=n_shared,
        n_product_features=n_product,
        d_store=4,
        n_brands=3,
        d_brand=2,
        n_styles=3,
        d_style=2,
    )
    head = IntegrableDemandHead(
        context_dim=tokens.d_token, K_splines=n_knots, n=n, k_neighbors=2, hidden=(8, 4)
    )
    return ICDN(context_builder=tokens, price_splines=splines, head=head, n=n)


def make_batch(B=7, n=4, n_shared=5, n_product=6, n_stores=2):
    return {
        "ids": torch.stack(
            [torch.randint(0, n_stores, (B,)), torch.arange(B)], dim=1
        ),
        "shared_feats": torch.randn(B, n_shared),
        "prices": torch.randn(B, n),
        "demands": torch.randn(B, n),
        "obs_mask": torch.ones(B, n),
        "product_feats": torch.randn(B, n, n_product),
        "product_cat": torch.randint(0, 3, (B, n, 2)),
    }


def test_forward_returns_demand_and_elasticities_per_product():
    model = build_network()
    y_hat, eps_hat = model(make_batch())

    assert y_hat.shape == (7, 4)
    assert eps_hat.shape == (7, 4)
    assert torch.isfinite(y_hat).all()


def test_elasticity_matrix_diagonal_matches_own_price_elasticity():
    model = build_network()
    _, eps_hat, aux = model(make_batch(), return_parts=True, compute_E=True)

    E = aux["E"]
    assert E.shape == (7, 4, 4)
    torch.testing.assert_close(torch.diagonal(E, dim1=1, dim2=2), eps_hat)


def test_model_runs_without_product_metadata():
    model = build_network()
    y_with, _ = model(make_batch(), meta=None)
    y_without, _ = model(make_batch(), meta=ProductMetadata())

    assert torch.isfinite(y_with).all()
    assert torch.isfinite(y_without).all()


def test_spline_derivatives_match_finite_differences():
    prices = torch.randn(200, 2).numpy()
    basis = SplineBuilder().build_basis(prices, n_knots=4)

    x = torch.tensor([[0.3, -0.2]])
    step = 1e-4
    Bx_plus, _, _ = basis(x + step)
    Bx_minus, _, _ = basis(x - step)
    _, dBx, _ = basis(x)

    torch.testing.assert_close((Bx_plus - Bx_minus) / (2 * step), dBx, atol=1e-3, rtol=1e-3)


def test_frozen_graph_keeps_the_same_edges():
    model = build_network()
    selector = model.head.neighbor_selector
    batch = make_batch()

    latents = [model.encode(batch)]
    scores = selector.accumulate_mean_scores(iter(latents))
    selector.freeze_graph(scores)

    pairs_before = selector.frozen_pairs.clone()
    model(batch)
    assert torch.equal(selector.frozen_pairs, pairs_before)
    assert selector.frozen_pairs.shape[1] == 4 * 2
