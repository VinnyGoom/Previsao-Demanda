"""Leakage-aware features and direct multi-step training rows."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "product_id",
    "horizon",
    "day_of_week",
    "month",
    "annual_sin",
    "annual_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]


def make_direct_supervised_data(
    panel: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Build one row per product/target date for a direct forecast horizon.

    All sales-derived features end at the forecast origin (target date minus
    ``horizon`` days). Calendar features describe the target date and are known
    in advance. This is suitable for an expanding-window backtest.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    required = {"date", "product_id", "sales", "stockout"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    ordered = panel.sort_values(["product_id", "date"]).copy()
    group = ordered.groupby("product_id", sort=False)["sales"]

    ordered["horizon"] = horizon
    ordered["day_of_week"] = ordered["date"].dt.dayofweek
    ordered["month"] = ordered["date"].dt.month
    day_of_year = ordered["date"].dt.dayofyear
    ordered["annual_sin"] = (2 * 3.141592653589793 * day_of_year / 365.25).map(
        __import__("math").sin
    )
    ordered["annual_cos"] = (2 * 3.141592653589793 * day_of_year / 365.25).map(
        __import__("math").cos
    )

    for lag in (1, 7, 14, 28):
        ordered[f"lag_{lag}"] = group.shift(horizon + lag - 1)

    sales_at_origin = group.shift(horizon)
    ordered["rolling_mean_7"] = sales_at_origin.groupby(
        ordered["product_id"], sort=False
    ).transform(lambda values: values.rolling(7, min_periods=7).mean())
    ordered["rolling_mean_28"] = sales_at_origin.groupby(
        ordered["product_id"], sort=False
    ).transform(lambda values: values.rolling(28, min_periods=28).mean())

    return ordered.dropna(subset=FEATURE_COLUMNS[1:] + ["sales"]).reset_index(
        drop=True
    )
