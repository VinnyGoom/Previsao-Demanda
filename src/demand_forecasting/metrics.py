"""Forecast metrics with safe handling for zero-demand groups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return MAE, WAPE and signed bias (positive means overforecast).

    WAPE and normalized bias are undefined when total actual demand is zero and
    are returned as NaN; MAE remains defined.
    """
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual_values.size == 0:
        raise ValueError("actual and predicted cannot be empty")

    absolute_error = np.abs(predicted_values - actual_values)
    denominator = np.abs(actual_values).sum()
    wape = float(absolute_error.sum() / denominator) if denominator else float("nan")
    bias = (
        float((predicted_values - actual_values).sum() / denominator)
        if denominator
        else float("nan")
    )
    return {"MAE": float(absolute_error.mean()), "WAPE": wape, "bias": bias}


def summarize_metrics(
    predictions: pd.DataFrame,
    groupby: tuple[str, ...] = ("model",),
) -> pd.DataFrame:
    """Calculate metrics by the requested prediction columns."""
    required = {"y_true", "y_pred", *groupby}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")

    rows = []
    for keys, frame in predictions.groupby(list(groupby), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groupby, keys))
        row.update(forecast_metrics(frame["y_true"], frame["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)
