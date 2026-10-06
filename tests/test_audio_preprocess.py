import unittest

import numpy as np

from ml_tasks.audio_events import ensure_min_length, resample_linear
from ml_tasks.text_sentiment import label_ru


class AudioPreprocessTest(unittest.TestCase):
    def test_resample_preserves_duration(self) -> None:
        audio = np.linspace(-1, 1, 8000, dtype=np.float32)
        converted = resample_linear(audio, 8000, 16000)
        self.assertEqual(len(converted), 16000)
        self.assertEqual(converted.dtype, np.float32)

    def test_same_rate_returns_float32(self) -> None:
        audio = np.zeros(1600, dtype=np.float64)
        converted = resample_linear(audio, 16000, 16000)
        self.assertEqual(len(converted), 1600)
        self.assertEqual(converted.dtype, np.float32)

    def test_short_signal_is_padded(self) -> None:
        padded = ensure_min_length(np.ones(100, dtype=np.float32), 16000)
        self.assertEqual(len(padded), 16000)
        self.assertEqual(float(padded[0]), 1.0)
        self.assertEqual(float(padded[-1]), 0.0)


class SentimentLabelsTest(unittest.TestCase):
    def test_russian_names(self) -> None:
        self.assertEqual(label_ru("positive"), "позитивная")
        self.assertEqual(label_ru("NEGATIVE"), "негативная")
        self.assertEqual(label_ru("neutral"), "нейтральная")


if __name__ == "__main__":
    unittest.main()
