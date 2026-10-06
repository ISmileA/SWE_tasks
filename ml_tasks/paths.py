"""Пути проекта. Каталог репозитория — родитель пакета ml_tasks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
