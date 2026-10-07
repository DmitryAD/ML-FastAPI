"""
Сервис предсказания оттока клиентов.

Запуск: uvicorn main:app --reload
"""

from fastapi import FastAPI

from schemas import FeatureVectorChurn

app = FastAPI(title="ML Churn Service")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


# TODO: заглушка для проверки схемы, заменить на вызов модели
@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features
