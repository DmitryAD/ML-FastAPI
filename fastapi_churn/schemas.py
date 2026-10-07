"""Схемы данных для задачи churn."""

from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
    """Признаки клиента, вход для предсказания."""

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    """Строка обучающего датасета: признаки и целевая переменная."""

    churn: int  # 1 - клиент ушёл, 0 - остался


class ClassDistribution(BaseModel):
    """Распределение churn по классам: количество строк и доля."""

    counts: dict[int, int]
    shares: dict[int, float]


class DatasetInfo(BaseModel):
    n_rows: int
    n_columns: int
    columns: list[str]
    churn_distribution: ClassDistribution


class SplitPart(BaseModel):
    n_rows: int
    churn_distribution: ClassDistribution


class SplitInfo(BaseModel):
    train: SplitPart
    test: SplitPart