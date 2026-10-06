# Серия заданий 1 (с1сз1)

Группа 11-411. Игнатьев Арсений, Николаева Полина.

Четыре задачи на уже обученных моделях и API для тональности текста. Своего обучения нет.

Кто что делал: Игнатьев — модели первой части и сам сервис (`main.py`). Николаева — тесты API и настройка GitHub Actions. Проверяли запуск вместе.

Подробности, метрики и что получилось на примерах — в [REPORT.md](REPORT.md).

## Задачи

| Задача | Фреймворк | Модель |
| --- | --- | --- |
| Тональность русского текста | Hugging Face | `cointegrated/rubert-tiny-sentiment-balanced` |
| События в wav | TensorFlow Hub | YAMNet |
| Класс на фото | PyTorch | ResNet-18 |
| Рамки на видео | Hugging Face | DETR, `facebook/detr-resnet-50` |

Тональность потом удобно обернуть в API: на вход текст, на выход JSON. Остальные команды принимают файлы.

Для DETR нужен пакет `timm`, без него модель не создаётся. Он есть в `requirements.txt`.

## Установка

Нужен Python 3.13. На 3.14 TensorFlow под Windows не ставится, это всплыло при первой попытке.

Видеокарта не нужна. PyTorch лучше ставить CPU-версией, иначе скачивается тяжёлый CUDA-пакет.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

То же самое собирает `scripts/setup.ps1`.

Первый запуск качает веса, это несколько сотен мегабайт. В репозиторий они не входят.

## Запуск

Из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m ml_tasks demo
```

Команда забирает примеры в `samples/` (фото, wav и короткий ролик из фото) и прогоняет все четыре модели. JSON пишется в `outputs/demo_results.json`, ролик с рамками — в `outputs/dog_detected.mp4`.

По отдельности:

```powershell
.\.venv\Scripts\python.exe -m ml_tasks text "Прекрасный фильм, рекомендую всем друзьям."
.\.venv\Scripts\python.exe -m ml_tasks audio samples\miaow_16k.wav
.\.venv\Scripts\python.exe -m ml_tasks image samples\dog.jpg
.\.venv\Scripts\python.exe -m ml_tasks video samples\dog_clip.mp4 --frame-stride 5
```

Для звука нужен обычный wav. Видео — любой файл, который открывает OpenCV.

## Файлы

```
ml_tasks/text_sentiment.py
ml_tasks/audio_events.py
ml_tasks/image_classification.py
ml_tasks/video_detection.py
ml_tasks/demo.py
samples/          примеры
outputs/          результаты, в git не коммитится
```

Модель в каждом файле загружается один раз за запуск. На видео детектор вызывается не каждый кадр, а с шагом `--frame-stride`, иначе на CPU ролик считается слишком долго.

## API тональности

Сервис в `main.py`. Нужны пакеты из `requirements.txt`, в том числе fastapi, uvicorn, httpx и pytest.

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Дальше `POST http://127.0.0.1:8000/predict/` с телом `{"text": "Прекрасный фильм, рекомендую всем друзьям."}`. Ответ: `label`, `label_ru`, `score`. Пустая строка даёт 422.

Тесты:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

При push в GitHub тот же pytest запускает workflow `.github/workflows/tests.yml`. Первые тесты на `/predict/` скачивают веса ruBERT, если их ещё нет в кэше.
