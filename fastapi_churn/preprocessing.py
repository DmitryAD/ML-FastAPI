"""Подготовка данных к обучению: X/y, пропуски, разбиение на train и test."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import TARGET

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]
# autopay_enabled - флаг 0/1, по смыслу это категория, а не величина
CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # строку без целевой переменной для обучения использовать нельзя
    df = df.dropna(subset=[TARGET])
    return df[FEATURES], df[TARGET].astype(int)


def fill_missing(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в X: медиана для числовых, мода для категориальных.

    Статистики считаются по reference, а не по самому X.
    """
    fill_values = {
        **reference[NUMERIC_FEATURES].median().to_dict(),
        **reference[CATEGORICAL_FEATURES].mode().iloc[0].to_dict(),
    }
    return X.fillna(fill_values)


def prepare_data(df: pd.DataFrame) -> PreparedData:
    X, y = split_features_target(df)

    # stratify сохраняет долю ушедших клиентов одинаковой в train и test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # пропуски заполняю после разбиения и только по train, иначе будет утечка из test
    return PreparedData(
        X_train=fill_missing(X_train, reference=X_train),
        X_test=fill_missing(X_test, reference=X_train),
        y_train=y_train,
        y_test=y_test,
    )