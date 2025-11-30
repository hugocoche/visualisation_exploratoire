import pandas as pd
import numpy as np


def variation_coeff(series: pd.Series) -> float:
    return round(series.std() / series.mean(), 4)


def yule_coeff(series: pd.Series) -> float:
    Q3 = series.quantile(0.75)
    Q2 = series.quantile(0.5)
    Q1 = series.quantile(0.25)
    return round(((Q3 - Q2) - (Q2 - Q1)) / ((Q3 - Q2) + (Q2 - Q1)), 4)


def tukey_outlier(series: pd.Series, threshold: float = 1.5) -> pd.DataFrame:
    Q3 = series.quantile(0.75)
    Q1 = series.quantile(0.25)
    iqr = Q3 - Q1
    lower_bound = Q1 - threshold * iqr
    upper_bound = Q3 + threshold * iqr
    temp_data = pd.DataFrame(series)
    temp_data["bool_col"] = series.apply(
        lambda row: True if lower_bound <= row <= upper_bound else False
    )
    return temp_data


def nice_range(series: pd.Series, precision: int) -> tuple[float, float]:
    """Retourne min et max arrondis à la précision souhaitée"""
    min_val = np.floor(series.min() * 10**precision) / 10**precision
    max_val = np.ceil(series.max() * 10**precision) / 10**precision
    return min_val, max_val
