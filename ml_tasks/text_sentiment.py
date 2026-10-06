"""Тональность короткого русского текста. Hugging Face, модель ruBERT-tiny."""

from __future__ import annotations

import time
from typing import Any

MODEL_ID = "cointegrated/rubert-tiny-sentiment-balanced"

LABELS_RU = {
    "negative": "негативная",
    "neutral": "нейтральная",
    "positive": "позитивная",
}

_STATE: dict[str, Any] = {}


def label_ru(label: str) -> str:
    return LABELS_RU.get(label.lower(), label)


def _load():
    if "model" not in _STATE:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.eval()
        _STATE["tokenizer"] = tokenizer
        _STATE["model"] = model
        _STATE["torch"] = torch
        _STATE["load_ms"] = (time.perf_counter() - started) * 1000
        _STATE["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return _STATE


def analyze_sentiment(texts: list[str]) -> dict[str, Any]:
    """Возвращает распределение по классам negative / neutral / positive для каждого текста.

    Словарь пригоден для JSON-ответа будущего HTTP API: на входе строки, на выходе метки и вероятности.
    """
    cleaned = [text.strip() for text in texts]
    if not cleaned or any(not text for text in cleaned):
        raise ValueError("Нужен хотя бы один непустой текст.")

    state = _load()
    torch = state["torch"]
    tokenizer = state["tokenizer"]
    model = state["model"]

    started = time.perf_counter()
    encoded = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1).tolist()
    inference_ms = (time.perf_counter() - started) * 1000

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    items = []
    for text, scores in zip(cleaned, probabilities):
        distribution = [
            {
                "label": id2label[index],
                "label_ru": label_ru(id2label[index]),
                "score": round(float(score), 4),
            }
            for index, score in enumerate(scores)
        ]
        best = max(distribution, key=lambda item: item["score"])
        items.append(
            {
                "text": text,
                "label": best["label"],
                "label_ru": best["label_ru"],
                "score": best["score"],
                "distribution": distribution,
            }
        )

    return {
        "task": "text_sentiment",
        "framework": "Hugging Face Transformers",
        "model": MODEL_ID,
        "parameters": state["parameters"],
        "load_ms": round(state["load_ms"], 1),
        "inference_ms": round(inference_ms, 1),
        "results": items,
    }
