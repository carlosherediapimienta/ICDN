"""Persistence of trained models.

A checkpoint carries the weights together with everything needed to rebuild
the exact same pipeline: the configuration, the panel layout, the identifier
encoders and the frozen competitor graph.
"""

from pathlib import Path

import torch

FORMAT_VERSION = 1


def save_checkpoint(path: str | Path, model: torch.nn.Module, config, layout) -> Path:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selector = getattr(model.head, "neighbor_selector", None)
    payload = {
        "format_version": FORMAT_VERSION,
        "config": config.to_dict(),
        "layout": layout.to_dict(),
        "state_dict": model.state_dict(),
        "frozen_pairs": None if selector is None else selector.frozen_pairs,
        "frozen_edge_bonus": None if selector is None else selector.frozen_edge_bonus,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"no ICDN checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {version} is not supported by this version of icdn "
            f"(expected {FORMAT_VERSION})"
        )
    return payload


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path / "model.icdn" if path.is_dir() or path.suffix == "" else path
