"""Классификация изображения по 1000 классам ImageNet. PyTorch, ResNet-18."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

MODEL_NAME = "resnet18"
WEIGHTS_NAME = "IMAGENET1K_V1"

_STATE: dict[str, Any] = {}


def _load():
    if "model" not in _STATE:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        started = time.perf_counter()
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.eval()
        _STATE["model"] = model
        _STATE["weights"] = weights
        _STATE["torch"] = torch
        _STATE["load_ms"] = (time.perf_counter() - started) * 1000
        _STATE["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return _STATE


def classify_image(path: str | Path, top_k: int = 5) -> dict[str, Any]:
    """Топ-K классов ImageNet. Предобработка берётся из карточки весов torchvision."""
    if top_k < 1:
        raise ValueError("top_k должен быть не меньше 1.")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {source}")

    from PIL import Image, ImageOps

    state = _load()
    torch = state["torch"]
    weights = state["weights"]
    model = state["model"]
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")

    started = time.perf_counter()
    batch = weights.transforms()(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    scores, indexes = torch.topk(probabilities, k=min(top_k, probabilities.shape[0]))
    inference_ms = (time.perf_counter() - started) * 1000

    categories = weights.meta["categories"]
    top = [
        {"label": categories[int(index)], "score": round(float(score), 4)}
        for score, index in zip(scores.tolist(), indexes.tolist())
    ]
    return {
        "task": "image_classification",
        "framework": "PyTorch (torchvision)",
        "model": MODEL_NAME,
        "weights": WEIGHTS_NAME,
        "parameters": state["parameters"],
        "input": str(source),
        "load_ms": round(state["load_ms"], 1),
        "inference_ms": round(inference_ms, 1),
        "top": top,
    }
