"""Классификация звуковых событий. TensorFlow Hub, модель YAMNet."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_RATE = 16000
MIN_SAMPLES = TARGET_RATE

_STATE: dict[str, Any] = {}


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_RATE) -> np.ndarray:
    """Линейная передискретизация. Для уже 16 кГц возвращает исходный массив."""
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    if source_rate <= 0 or len(audio) == 0:
        raise ValueError("Пустой сигнал или некорректная частота дискретизации.")
    target_length = int(round(len(audio) * target_rate / source_rate))
    source_axis = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_axis = np.linspace(0.0, 1.0, num=max(target_length, 1), endpoint=False)
    return np.interp(target_axis, source_axis, audio).astype(np.float32)


def ensure_min_length(audio: np.ndarray, min_samples: int = MIN_SAMPLES) -> np.ndarray:
    """YAMNet смотрит окна по 0,96 с. Короткий сигнал дополняется тишиной до одной секунды."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= min_samples:
        return audio
    return np.pad(audio, (0, min_samples - len(audio)))


def _load_waveform(path: Path):
    import tensorflow as tf

    contents = tf.io.read_file(str(path))
    try:
        audio, sample_rate = tf.audio.decode_wav(contents, desired_channels=1)
    except tf.errors.InvalidArgumentError as error:
        raise ValueError(f"Ожидался PCM WAV, файл не разобран: {path}") from error
    waveform = tf.squeeze(audio, axis=-1).numpy()
    waveform = resample_linear(waveform, int(sample_rate.numpy()), TARGET_RATE)
    waveform = np.clip(waveform, -1.0, 1.0)
    return ensure_min_length(waveform)


def _class_names(model) -> list[str]:
    raw_path = model.class_map_path().numpy()
    path = raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path)
    names: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names.append(row["display_name"])
    if len(names) != 521:
        raise RuntimeError(f"Ожидалось 521 класс YAMNet, получено {len(names)}.")
    return names


def _load():
    if "model" not in _STATE:
        import tensorflow as tf
        import tensorflow_hub as hub

        started = time.perf_counter()
        model = hub.load(MODEL_URL)
        names = _class_names(model)
        _STATE["model"] = model
        _STATE["class_names"] = names
        _STATE["tf"] = tf
        _STATE["load_ms"] = (time.perf_counter() - started) * 1000
        # у объекта с Hub нет .variables, число параметров здесь не считается
    return _STATE


def classify_audio(path: str | Path, top_k: int = 5) -> dict[str, Any]:
    """Средние по кадрам вероятности 521 класса AudioSet и топ-K событий."""
    if top_k < 1:
        raise ValueError("top_k должен быть не меньше 1.")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {source}")

    state = _load()
    tensorflow = state["tf"]
    waveform = _load_waveform(source)
    started = time.perf_counter()
    scores, _embeddings, _spectrogram = state["model"](tensorflow.constant(waveform, dtype=tensorflow.float32))
    mean_scores = scores.numpy().mean(axis=0)
    inference_ms = (time.perf_counter() - started) * 1000

    order = np.argsort(mean_scores)[::-1][:top_k]
    names = state["class_names"]
    top = [
        {"label": names[int(index)], "score": round(float(mean_scores[index]), 4)}
        for index in order
    ]
    return {
        "task": "audio_events",
        "framework": "TensorFlow Hub",
        "model": MODEL_URL,
        "input": str(source),
        "sample_rate": TARGET_RATE,
        "duration_s": round(len(waveform) / TARGET_RATE, 3),
        "frames": int(scores.shape[0]),
        "load_ms": round(state["load_ms"], 1),
        "inference_ms": round(inference_ms, 1),
        "top": top,
    }
