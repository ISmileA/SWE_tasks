"""Сквозной прогон четырёх задач на демонстрационных данных."""

from __future__ import annotations

import json
import sys

from ml_tasks.audio_events import classify_audio
from ml_tasks.image_classification import classify_image
from ml_tasks.paths import OUTPUTS
from ml_tasks.samples import ensure_samples
from ml_tasks.text_sentiment import analyze_sentiment
from ml_tasks.video_detection import detect_video

DEMO_TEXTS = [
    "Прекрасный фильм, рекомендую всем друзьям.",
    "Ужасное качество, больше никогда не куплю.",
    "Сегодня вторник, занятие начинается в десять.",
]


def run_demo(frame_stride: int = 5) -> dict:
    print("Готовлю примеры: фото, wav и короткий ролик...", file=sys.stderr)
    samples = ensure_samples()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("1/4 Тональность текста (Hugging Face)...", file=sys.stderr)
    text = analyze_sentiment(DEMO_TEXTS)
    print("2/4 Звуковые события (TensorFlow Hub)...", file=sys.stderr)
    audio = classify_audio(samples["audio"])
    print("3/4 Классификация изображения (PyTorch)...", file=sys.stderr)
    image = classify_image(samples["image"])
    print("4/4 Детекция на видео (Hugging Face DETR)...", file=sys.stderr)
    video = detect_video(samples["video"], OUTPUTS / "dog_detected.mp4", frame_stride=frame_stride)

    payload = {"text": text, "audio": audio, "image": image, "video": video}
    destination = OUTPUTS / "demo_results.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сводка записана в {destination}", file=sys.stderr)
    return payload
