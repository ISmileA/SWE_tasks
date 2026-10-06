from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API тональности текста"}


def test_predict_positive():
    response = client.post(
        "/predict/",
        json={"text": "Прекрасный фильм, рекомендую всем друзьям."},
    )
    data = response.json()
    assert response.status_code == 200
    assert data["label"] == "positive"
    assert data["label_ru"] == "позитивная"
    assert data["score"] > 0.5


def test_predict_negative():
    response = client.post(
        "/predict/",
        json={"text": "Ужасное качество, больше никогда не куплю."},
    )
    data = response.json()
    assert response.status_code == 200
    assert data["label"] == "negative"
    assert data["label_ru"] == "негативная"
    assert data["score"] > 0.5


def test_predict_neutral():
    response = client.post(
        "/predict/",
        json={"text": "Сегодня вторник, занятие начинается в десять."},
    )
    data = response.json()
    assert response.status_code == 200
    assert data["label"] == "neutral"


def test_empty_text():
    response = client.post("/predict/", json={"text": ""})
    assert response.status_code == 422
