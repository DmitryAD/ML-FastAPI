"""
Сервис предсказания оттока клиентов.

Запуск: uvicorn main:app --reload
"""

from typing import Annotated

from fastapi import FastAPI, Query

from dataset import TARGET, class_distribution, load_dataset, to_rows
from preprocessing import prepare_data
from schemas import (
    DatasetInfo,
    DatasetRowChurn,
    FeatureVectorChurn,
    SplitInfo,
    SplitPart,
)

app = FastAPI(title="ML Churn Service")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@app.get("/dataset/preview")
def dataset_preview(
    n: Annotated[int, Query(ge=1, le=100)] = 5,
) -> list[DatasetRowChurn]:
    return to_rows(load_dataset().head(n))


@app.get("/dataset/info")
def dataset_info() -> DatasetInfo:
    df = load_dataset()
    return DatasetInfo(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        columns=df.columns.tolist(),
        churn_distribution=class_distribution(df[TARGET]),
    )


@app.get("/dataset/split-info")
def dataset_split_info() -> SplitInfo:
    data = prepare_data(load_dataset())
    return SplitInfo(
        train=SplitPart(
            n_rows=len(data.X_train),
            churn_distribution=class_distribution(data.y_train),
        ),
        test=SplitPart(
            n_rows=len(data.X_test),
            churn_distribution=class_distribution(data.y_test),
        ),
    )


# TODO: заглушка для проверки схемы, заменить на вызов модели
@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features