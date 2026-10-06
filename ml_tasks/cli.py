"""Командная строка четырёх задач."""

from __future__ import annotations

import argparse
import json
import sys

from ml_tasks.audio_events import classify_audio
from ml_tasks.demo import run_demo
from ml_tasks.image_classification import classify_image
from ml_tasks.paths import OUTPUTS
from ml_tasks.text_sentiment import analyze_sentiment
from ml_tasks.video_detection import detect_video


def _print(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml_tasks",
        description="Четыре прикладные задачи на готовых моделях: текст, аудио, изображение, видео.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text = subparsers.add_parser("text", help="Тональность одного или нескольких русских текстов")
    text.add_argument("texts", nargs="+", help="Фразы на русском языке")

    audio = subparsers.add_parser("audio", help="События в WAV  (речь, животные, бытовые звуки)")
    audio.add_argument("path", help="Путь к PCM WAV")
    audio.add_argument("--top-k", type=int, default=5)

    image = subparsers.add_parser("image", help="Классификация изображения, ImageNet")
    image.add_argument("path", help="Путь к изображению")
    image.add_argument("--top-k", type=int, default=5)

    video = subparsers.add_parser("video", help="Рамки объектов на видео")
    video.add_argument("path", help="Путь к видео")
    video.add_argument("--output", default=str(OUTPUTS / "detected.mp4"))
    video.add_argument("--threshold", type=float, default=0.7)
    video.add_argument("--frame-stride", type=int, default=4)

    demo = subparsers.add_parser("demo", help="Прогон всех четырёх задач на встроенных примерах")
    demo.add_argument("--frame-stride", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "text":
            _print(analyze_sentiment(args.texts))
        elif args.command == "audio":
            _print(classify_audio(args.path, top_k=args.top_k))
        elif args.command == "image":
            _print(classify_image(args.path, top_k=args.top_k))
        elif args.command == "video":
            _print(
                detect_video(
                    args.path,
                    args.output,
                    threshold=args.threshold,
                    frame_stride=args.frame_stride,
                )
            )
        elif args.command == "demo":
            _print(run_demo(frame_stride=args.frame_stride))
        else:
            raise AssertionError(args.command)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0
