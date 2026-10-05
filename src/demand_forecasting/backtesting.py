"""Expanding-window, leakage-aware evaluation for direct forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURE_COLUMNS, make_direct_supervised_data


def expanding_window_backtest(
    panel: pd.DataFrame,
    horizon: int = 7,
    min_train_days: int = 180,
    step_days: int = 28,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare persistence baselines, Ridge and Random Forest by rolling origin.

    For each origin, each direct horizon model is trained only on labeled
    targets dated on or before that origin. Stockout targets are excluded from
    fitting and scoring because sales are censored. For synthetic experiments,
    ``latent_demand`` may be analyzed separately as simulation ground truth,
    but it is never used as a model feature or target here.
    """
    if horizon < 1 or min_train_days < 28 or step_days < 1:
        raise ValueError("horizon, min_train_days and step_days must be positive")
    required = {"date", "product_id", "sales", "stockout"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    dated = panel.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    supervised = {
        h: make_direct_supervised_data(dated, h) for h in range(1, horizon + 1)
    }
    all_dates = pd.DatetimeIndex(sorted(dated["date"].unique()))
    if len(all_dates) <= min_train_days:
        raise ValueError("not enough dates for the requested training window")

    origin_positions = range(min_train_days - 1, len(all_dates) - horizon, step_days)
    records: list[dict[str, object]] = []

    for origin_position in origin_positions:
        origin = all_dates[origin_position]
        for h in range(1, horizon + 1):
            target_date = all_dates[origin_position + h]
            frame = supervised[h]
            train = frame.loc[
                (frame["date"] <= origin) & (~frame["stockout"])
            ].copy()
            test = frame.loc[frame["date"] == target_date].copy()
            test = test.loc[~test["stockout"]]
            if train.empty or test.empty:
                continue

            for baseline_name, feature in (
                ("naive_last_day", "lag_1"),
                ("seasonal_naive_7d", "lag_7"),
            ):
                for row in test.itertuples(index=False):
                    records.append(
                        {
                            "origin_date": origin,
                            "date": target_date,
                            "product_id": row.product_id,
                            "horizon": h,
                            "model": baseline_name,
                            "y_true": float(row.sales),
                            "y_pred": max(0.0, float(getattr(row, feature))),
                        }
                    )

            estimators = {
                "ridge": Ridge(alpha=10.0),
                "random_forest": RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=3,
                    max_features=0.9,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            }
            for model_name, estimator in estimators.items():
                numeric_transformer = (
                    StandardScaler() if model_name == "ridge" else "passthrough"
                )
                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "product",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["product_id"],
                        ),
                        ("numeric", numeric_transformer, FEATURE_COLUMNS[1:]),
                    ],
                    remainder="drop",
                )
                model = Pipeline(
                    steps=[("preprocess", preprocessor), ("regressor", estimator)]
                )
                model.fit(train[FEATURE_COLUMNS], train["sales"])
                predictions = np.maximum(0.0, model.predict(test[FEATURE_COLUMNS]))
                for row, prediction in zip(test.itertuples(index=False), predictions):
                    records.append(
                        {
                            "origin_date": origin,
                            "date": target_date,
                            "product_id": row.product_id,
                            "horizon": h,
                            "model": model_name,
                            "y_true": float(row.sales),
                            "y_pred": float(prediction),
                        }
                    )

    return pd.DataFrame.from_records(records)
