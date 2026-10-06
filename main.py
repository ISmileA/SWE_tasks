from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_tasks.text_sentiment import analyze_sentiment

app = FastAPI(title="Тональность текста")


class Item(BaseModel):
    text: str = Field(min_length=1)


class Prediction(BaseModel):
    label: str
    label_ru: str
    score: float


@app.get("/")
def root():
    return {"message": "API тональности текста"}


@app.post("/predict/", response_model=Prediction)
def predict(item: Item):
    try:
        result = analyze_sentiment([item.text])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    row = result["results"][0]
    return Prediction(label=row["label"], label_ru=row["label_ru"], score=row["score"])
