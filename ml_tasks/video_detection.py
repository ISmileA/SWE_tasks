"""Детекция объектов на кадрах видео. Hugging Face, DETR с основой ResNet-50."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/detr-resnet-50"

_STATE: dict[str, Any] = {}


def _label_name(id2label: dict, index: int) -> str:
    if index in id2label:
        return str(id2label[index])
    return str(id2label.get(str(index), index))


def _load():
    if "model" not in _STATE:
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor

        started = time.perf_counter()
        processor = DetrImageProcessor.from_pretrained(MODEL_ID)
        model = DetrForObjectDetection.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _STATE["processor"] = processor
        _STATE["model"] = model
        _STATE["device"] = device
        _STATE["torch"] = torch
        _STATE["load_ms"] = (time.perf_counter() - started) * 1000
        _STATE["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return _STATE


def _detect_frame(frame_bgr, state: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    import cv2

    torch = state["torch"]
    processor = state["processor"]
    model = state["model"]
    device = state["device"]
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[height, width]], device=device)
    result = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    id2label = model.config.id2label
    detections = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        detections.append(
            {
                "label": _label_name(id2label, int(label)),
                "score": round(float(score), 4),
                "box": [round(float(value), 1) for value in box.tolist()],
            }
        )
    return detections


def _draw(frame_bgr, detections: list[dict[str, Any]]):
    import cv2

    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["box"])
        caption = f"{detection['label']} {detection['score']:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (36, 170, 60), 2)
        cv2.putText(
            frame_bgr,
            caption,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (36, 170, 60),
            2,
            cv2.LINE_AA,
        )
    return frame_bgr


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, path
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать выходное видео (mp4v и MJPG).")
    return writer, fallback


def detect_video(
    path: str | Path,
    output: str | Path,
    threshold: float = 0.7,
    frame_stride: int = 4,
) -> dict[str, Any]:
    """Прогоняет DETR по ключевым кадрам и переносит рамки на промежуточные кадры.

    Модель обучена на отдельных фотографиях COCO, поэтому временной модели нет:
    шаг по кадрам снижает стоимость почти линейно и уменьшает дрожание рамок.
    """
    import cv2

    if frame_stride < 1:
        raise ValueError("frame_stride должен быть не меньше 1.")
    if not 0 < threshold < 1:
        raise ValueError("Порог должен быть между 0 и 1.")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Видео не найдено: {source}")

    state = _load()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV не открыл видео: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, output_path = _open_writer(Path(output), fps, (width, height))

    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "max_score": 0.0})
    frames_read = 0
    frames_inferred = 0
    last_detections: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frames_read % frame_stride == 0:
                last_detections = _detect_frame(frame, state, threshold)
                frames_inferred += 1
                for detection in last_detections:
                    bucket = totals[detection["label"]]
                    bucket["count"] += 1
                    bucket["max_score"] = max(bucket["max_score"], detection["score"])
            writer.write(_draw(frame, last_detections))
            frames_read += 1
    finally:
        capture.release()
        writer.release()
    inference_ms = (time.perf_counter() - started) * 1000

    summary = [
        {"label": label, "keyframes": int(stats["count"]), "max_score": round(stats["max_score"], 4)}
        for label, stats in sorted(totals.items(), key=lambda item: item[1]["max_score"], reverse=True)
    ]
    return {
        "task": "video_detection",
        "framework": "Hugging Face Transformers",
        "model": MODEL_ID,
        "parameters": state["parameters"],
        "device": str(state["device"]),
        "input": str(source),
        "output": str(output_path),
        "threshold": threshold,
        "frame_stride": frame_stride,
        "frames_read": frames_read,
        "frames_inferred": frames_inferred,
        "load_ms": round(state["load_ms"], 1),
        "inference_ms": round(inference_ms, 1),
        "detections": summary,
    }
