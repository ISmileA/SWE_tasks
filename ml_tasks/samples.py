"""Демонстрационные файлы: фото из туториала PyTorch и запись мяуканья из туториала YAMNet."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from ml_tasks.paths import SAMPLES

DOG_IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
MIAOW_URL = "https://storage.googleapis.com/audioset/miaow_16k.wav"

DOG_IMAGE = SAMPLES / "dog.jpg"
MIAOW_WAV = SAMPLES / "miaow_16k.wav"
DOG_CLIP = SAMPLES / "dog_clip.mp4"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ml-tasks-course/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            handle.write(response.read())
        temporary.replace(dest)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_clip(image_path: Path, dest: Path, seconds: float = 2.5, fps: int = 12) -> Path:
    """Короткий ролик с медленным наездом. Так видеодетектор получает настоящую последовательность кадров."""
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")
    height, width = frame.shape[:2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    output = dest
    if not writer.isOpened():
        output = dest.with_suffix(".avi")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV не смог открыть VideoWriter ни для mp4, ни для avi.")

    count = max(int(seconds * fps), 2)
    try:
        for index in range(count):
            progress = index / (count - 1)
            scale = 1.0 + 0.15 * progress
            crop_width = max(int(width / scale), 1)
            crop_height = max(int(height / scale), 1)
            x0 = (width - crop_width) // 2
            y0 = int((height - crop_height) * progress * 0.35)
            crop = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
            writer.write(cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR))
    finally:
        writer.release()
    return output


def ensure_samples() -> dict[str, Path]:
    """Скачивает фото и wav, если их ещё нет, и собирает из фото короткий ролик."""
    _download(DOG_IMAGE_URL, DOG_IMAGE)
    audio = MIAOW_WAV
    try:
        _download(MIAOW_URL, MIAOW_WAV)
    except Exception as error:
        audio = synthesize_tone(SAMPLES / "tone_440.wav")
        print(
            f"Не удалось скачать miaow_16k.wav ({error}). Вместо него записан тон 440 Гц: {audio}",
            file=sys.stderr,
        )
    clip = DOG_CLIP
    if not clip.exists() or clip.stat().st_size == 0:
        clip = _write_clip(DOG_IMAGE, DOG_CLIP)
    return {"image": DOG_IMAGE, "audio": audio, "video": clip}


def synthesize_tone(path: Path, frequency: float = 440.0, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """Запасной wav, если внешний файл скачать нельзя. Чистый тон — отдельный класс AudioSet."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    timeline = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * frequency * timeline) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path
