"""Simple append-only forecast/actual comparison for notebook monitoring."""

from __future__ import annotations

import pandas as pd

from .metrics import summarize_metrics


def evaluate_forecast_log(
    forecast_log: pd.DataFrame,
    groupby: tuple[str, ...] = ("model",),
) -> pd.DataFrame:
    """Evaluate forecasts once their actual sales have been observed.

    Expected columns include ``date``, ``product_id``, ``model``, ``y_pred``
    and ``y_true``. Pending forecasts may use null ``y_true`` and are skipped.
    """
    required = {"date", "product_id", "model", "y_pred", "y_true"}
    missing = required.difference(forecast_log.columns)
    if missing:
        raise ValueError(f"forecast_log is missing columns: {sorted(missing)}")
    observed = forecast_log.dropna(subset=["y_true"])
    if observed.empty:
        return pd.DataFrame(columns=[*groupby, "MAE", "WAPE", "bias"])
    return summarize_metrics(observed, groupby=groupby)
