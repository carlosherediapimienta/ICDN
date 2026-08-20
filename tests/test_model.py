import torch

from torch.utils.data import DataLoader, Dataset

from icdn import ICDNConfig
from icdn.training import Trainer

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
        d_product=4,
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


def test_identical_covariates_still_yield_distinct_product_tokens():
    builder = ProductTokenBuilder(
        n=2, n_stores=1, n_shared_features=3, n_product_features=4,
        d_store=4, d_product=8, n_brands=1, d_brand=2, n_styles=1, d_style=2,
    )
    B = 1
    batch = {
        "ids": torch.zeros(B, 2, dtype=torch.long),
        "shared_feats": torch.zeros(B, 3),
        "product_feats": torch.zeros(B, 2, 4),
        "product_cat": torch.zeros(B, 2, 2, dtype=torch.long),
    }
    tokens = builder(batch)
    assert not torch.allclose(tokens[0, 0], tokens[0, 1])

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


def test_warmup_is_exactly_log_linear():
    torch.manual_seed(0)
    model = build_network()
    batch = make_batch()
    head = model.head.param_head

    # Reproduce lo que debe hacer el trainer al entrar en fase 0.
    for module in (head.head_w, head.head_w_cross, head.head_cross):
        with torch.no_grad():
            module.weight.zero_()
            module.bias.zero_()
        for p in module.parameters():
            p.requires_grad = False

    y_hat, _, aux = model(batch, return_parts=True, linear_warmup=True)
    b, beta, x = aux["b"], aux["beta"], batch["prices"]
    beta_cross, pairs = aux["beta_cross"], aux["pairs"]
    attn = model.head.neighbor_selector.run(model.head.encoder(model.context_builder(batch)))[1]

    linear = b + beta * x
    i_idx, j_idx = pairs[0], pairs[1]
    contrib = attn * beta_cross * x[:, j_idx]
    linear = linear.scatter_add(1, i_idx.unsqueeze(0).expand(x.size(0), -1), contrib)

    torch.testing.assert_close(y_hat, linear, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(aux["w"]) == 0
    assert torch.count_nonzero(aux["w_cross"]) == 0
    assert torch.count_nonzero(aux["u"]) == 0


class _DictBatchDataset(Dataset):
    """One observation per index; DataLoader stacks to a batch dict."""

    def __init__(self, batch: dict):
        self.batch = batch
        self.n = int(batch["ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {key: value[idx] for key, value in self.batch.items()}


def _loader_from_batch(batch: dict, batch_size: int) -> DataLoader:
    return DataLoader(
        _DictBatchDataset(batch),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def test_evaluate_with_train_graph_restores_online_mode():
    torch.manual_seed(0)
    model = build_network()
    selector = model.head.neighbor_selector
    trainer = Trainer(ICDNConfig(device="cpu", verbose=False, lambda_elast=0.0))

    train_loader = _loader_from_batch(make_batch(B=8), batch_size=4)
    val_loader = _loader_from_batch(make_batch(B=6), batch_size=3)
    loss_fn = trainer._build_loss()

    assert selector.frozen_pairs is None
    trainer._evaluate_with_train_graph(
        model, train_loader, val_loader, loss_fn, meta=None
    )
    assert selector.frozen_pairs is None
    assert selector.frozen_edge_bonus is None


def test_validation_uses_train_frozen_graph_not_val_batch():
    torch.manual_seed(0)
    model = build_network()
    selector = model.head.neighbor_selector
    trainer = Trainer(ICDNConfig(device="cpu", verbose=False, lambda_elast=0.0))

    train_batch = make_batch(B=10)
    val_batch = make_batch(B=6)
    train_loader = _loader_from_batch(train_batch, batch_size=5)
    val_loader = _loader_from_batch(val_batch, batch_size=2)
    loss_fn = trainer._build_loss()

    # Reference graph from train (what validation should use).
    trainer.freeze_graph(model, train_loader, meta=None)
    train_pairs = selector.frozen_pairs.clone()
    selector.frozen_pairs = None
    selector.frozen_edge_bonus = None

    # Online selection on a val batch can differ from the train-frozen graph.
    model.eval()
    with torch.no_grad():
        online_pairs, _ = selector.run(model.encode(val_batch))
    assert not torch.equal(online_pairs, train_pairs)

    seen = {}

    def _capture_run(h, meta=None):
        pairs, weights = original_run(h, meta=meta)
        seen["pairs"] = pairs.detach().clone()
        return pairs, weights

    original_run = selector.run
    selector.run = _capture_run
    try:
        trainer._evaluate_with_train_graph(
            model, train_loader, val_loader, loss_fn, meta=None
        )
    finally:
        selector.run = original_run

    assert "pairs" in seen
    assert torch.equal(seen["pairs"], train_pairs)
    assert selector.frozen_pairs is None  # online mode restored