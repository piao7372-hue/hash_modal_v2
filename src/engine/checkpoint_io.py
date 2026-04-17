from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    if device != "cuda:0":
        raise ValueError("Formal Stage 4 checkpoint loading requires device cuda:0.")
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=torch.device(device))
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint payload is invalid: {path}")
    if checkpoint.get("device") != device:
        raise RuntimeError(
            f"Checkpoint device mismatch: saved={checkpoint.get('device')}, expected={device}."
        )
    if checkpoint.get("dtype") != "float32":
        raise RuntimeError(
            f"Checkpoint dtype mismatch: saved={checkpoint.get('dtype')}, expected=float32."
        )
    return checkpoint


__all__ = ["load_checkpoint", "save_checkpoint"]
