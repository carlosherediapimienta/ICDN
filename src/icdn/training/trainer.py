"""Two-phase training loop for the demand network."""

import copy
import math
import random 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ICDNConfig
from ..model.loss import ElasticityLoss
from ..model.neighbors import ProductMetadata


class Trainer:
    """Fits an ICDN model in two phases and then freezes its competitor graph.

    The warm-up phase fits smoothed demand with the spline weights frozen and set 
    to zero and the price coefficient pinned to a negative prior, which anchors a
    stable downward-sloping baseline. The main phase releases the splines and fits
    the raw series, letting the model capture non-linear and cross-price effects
    without the early instability that a cold start produces.
    """

    def __init__(self, config: ICDNConfig):
        self.config = config
        self.device = resolve_device(config.device)

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warmup_train_loader: DataLoader | None = None,
        warmup_val_loader: DataLoader | None = None,
        meta: ProductMetadata | None = None,
    ) -> dict:
        cfg = self.config
        seed_everything(cfg.seed)
        model.to(self.device)
        if meta is not None:
            meta = meta.to(self.device)

        history = {}

        if cfg.warmup_epochs > 0 and warmup_train_loader is not None:
            self._zero_and_freeze_splines(model)
            self._init_price_prior(model)
            history["warmup"] = self._run_phase(
                model,
                warmup_train_loader,
                warmup_val_loader or val_loader,
                lr=cfg.warmup_lr,
                n_epochs=cfg.warmup_epochs,
                meta=meta,
                linear_warmup=True,
                label="warmup",
            )

        self._freeze_splines(model, frozen=False)
        history["main"] = self._run_phase(
            model,
            train_loader,
            val_loader,
            lr=cfg.lr,
            n_epochs=cfg.epochs,
            meta=meta,
            linear_warmup=False,
            label="main",
        )

        self.freeze_graph(model, train_loader, meta)
        return history

    # ── Phases ──────────────────────────────────────────────────────────────

    def _run_phase(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        n_epochs: int,
        meta: ProductMetadata | None,
        linear_warmup: bool,
        label: str,
    ) -> dict:
        cfg = self.config
        loss_fn = self._build_loss()
        optimizer = self._build_optimizer(model, lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=cfg.lr_patience, min_lr=1e-5
        )
        scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        best_loss = math.inf
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(n_epochs):
            train_loss = self._train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scaler,
                meta,
                linear_warmup,
            )
            val_loss = self._evaluate_epoch(model, val_loader, loss_fn, meta, linear_warmup)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            if optimizer.param_groups[0]["lr"] < previous_lr:
                # A learning-rate drop is a fresh start, so patience resets.
                epochs_without_improvement = 0

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if cfg.verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(
                    f"[{label}] epoch {epoch + 1}/{n_epochs} "
                    f"train={train_loss:.4f} val={val_loss:.4f}"
                )

            if epochs_without_improvement >= cfg.early_stopping_patience:
                if cfg.verbose:
                    print(f"[{label}] early stop at epoch {epoch + 1}")
                break

        model.load_state_dict(best_state)
        history["best_val_loss"] = best_loss
        return history

    def _train_epoch(self, model, loader, loss_fn, optimizer, scaler, meta, linear_warmup: bool = False) -> float:
        cfg = self.config
        model.train()
        total, denom = 0.0, 0.0

        for batch in loader:
            batch = self._to_device(batch)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, logs = self._compute_loss(model, batch, loss_fn, meta, linear_warmup)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                for name, param in model.named_parameters():
                    self._require_finite(param, name)
            else:
                loss, logs = self._compute_loss(model, batch, loss_fn, meta, linear_warmup)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm: {float(grad_norm)}")
                optimizer.step()
                for name, param in model.named_parameters():
                    self._require_finite(param, name)

            weight = batch["obs_mask"].sum().item()
            total += logs["loss"].item() * weight
            denom += weight

        return total / max(denom, 1.0)

    @torch.no_grad()
    def _evaluate_epoch(self, model, loader, loss_fn, meta, linear_warmup: bool = False) -> float:
        model.eval()
        total, denom = 0.0, 0.0

        for batch in loader:
            batch = self._to_device(batch)
            _, logs = self._compute_loss(model, batch, loss_fn, meta, linear_warmup)
            weight = batch["obs_mask"].sum().item()
            total += logs["loss"].item() * weight
            denom += weight

        return total / max(denom, 1.0)

    def _compute_loss(self, model, batch, loss_fn, meta, linear_warmup: bool = False):
        needs_E = self.config.lambda_elast > 0.0
        y_hat, _, aux = model(
            batch,
            return_parts=True,
            compute_E=needs_E,
            meta=meta,
            linear_warmup=linear_warmup,
        )

        # Check for non-finite values in the loss components.
        self._require_finite(y_hat, "y_hat")
        self._require_finite(aux["beta"], "beta")
        self._require_finite(aux["w"], "w")
        self._require_finite(aux["w_cross"], "w_cross")
        self._require_finite(aux["u"], "u")
        self._require_finite(aux["Bx"], "Bx")
        self._require_finite(aux["dBx"], "dBx")
        self._require_finite(aux["ddBx"], "ddBx")

        loss, logs = loss_fn(
            y_hat,
            batch["demands"],
            batch["obs_mask"],
            aux["w"],
            aux["ddBx"],
            aux["u"],
            aux["Bx"],
            aux["pairs"],
            E=aux.get("E"),
            attn_weights=aux.get("attn_weights"),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss}")

        return loss, logs

    # ── Model preparation ───────────────────────────────────────────────────

    def _build_loss(self) -> ElasticityLoss:
        cfg = self.config
        return ElasticityLoss(
            huber_delta=cfg.huber_delta,
            lambda_smooth=cfg.lambda_smooth,
            lambda_elast=cfg.lambda_elast,
            l_own=cfg.own_elasticity_bounds[0],
            r_own=cfg.own_elasticity_bounds[1],
            l_cross=cfg.cross_elasticity_bounds[0],
            r_cross=cfg.cross_elasticity_bounds[1],
        ).to(self.device)

    def _build_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        # Spline, cross-price and bias parameters are left undecayed: shrinking
        # them toward zero would flatten the very curvature we want to learn.
        decayed, undecayed = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "head_w" in name or "cross" in name or "spline" in name:
                undecayed.append(param)
            else:
                decayed.append(param)

        return torch.optim.AdamW(
            [
                {"params": decayed, "weight_decay": self.config.weight_decay},
                {"params": undecayed, "weight_decay": 0.0},
            ],
            lr=lr,
        )

    def _zero_and_freeze(self, module: nn.Linear) -> None:
        with torch.no_grad():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()
        for parameter in module.parameters():
            parameter.requires_grad = False
            
    def _zero_and_freeze_splines(self, model: nn.Module) -> None:
        head = model.head.param_head
        self._zero_and_freeze(head.head_w)
        if head.use_cross:
            self._zero_and_freeze(head.head_w_cross)
            self._zero_and_freeze(head.head_cross)

    def _freeze_splines(self, model: nn.Module, frozen: bool) -> None:
        head = model.head.param_head
        modules = [head.head_w]
        if head.use_cross:
            modules += [head.head_w_cross, head.head_cross]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = not frozen

    def _init_price_prior(self, model: nn.Module) -> None:
        """Starts the linear price coefficient at the configured prior."""
        head = model.head.param_head
        with torch.no_grad():
            head.head_beta.weight.zero_()
            if head.enforce_negative_beta:
                target = max(-self.config.beta_prior, 1e-3)
                head.head_beta.bias.fill_(_inv_softplus(target))
            else:
                head.head_beta.bias.fill_(self.config.beta_prior)
            if head.use_cross:
                head.head_beta_cross.weight.zero_()
                head.head_beta_cross.bias.zero_()

    @torch.no_grad()
    def freeze_graph(
        self,
        model: nn.Module,
        loader: DataLoader,
        meta: ProductMetadata | None = None,
    ) -> None:
        """Fixes the competitor graph using the average scores over training data."""
        selector = model.head.neighbor_selector
        if selector is None:
            return

        model.eval()
        latents = (model.encode(self._to_device(batch)) for batch in loader)
        mean_scores = selector.accumulate_mean_scores(latents, meta=meta)
        selector.freeze_graph(mean_scores, meta=meta)

    def _require_finite(self, tensor: torch.Tensor | None, name: str) -> None:
        if tensor is None or tensor.numel() == 0:
            return
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite values in {name}")

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def seed_everything(seed: int) -> None: 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inv_softplus(x: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(float(x)))))
