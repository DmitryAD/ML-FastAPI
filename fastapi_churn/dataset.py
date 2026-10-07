"""Загрузка обучающего датасета."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from schemas import ClassDistribution, DatasetRowChurn

DATA_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"
TARGET = "churn"


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Читает CSV один раз, дальше отдаёт DataFrame из памяти."""
    return pd.read_csv(DATA_PATH)


def to_rows(df: pd.DataFrame) -> list[DatasetRowChurn]:
    return [
        DatasetRowChurn.model_validate(record)
        for record in df.to_dict(orient="records")
    ]


def class_distribution(y: pd.Series) -> ClassDistribution:
    counts = y.value_counts().sort_index()
    shares = counts / counts.sum()
    return ClassDistribution(
        counts={int(label): int(count) for label, count in counts.items()},
        shares={int(label): round(float(share), 4) for label, share in shares.items()},
    )