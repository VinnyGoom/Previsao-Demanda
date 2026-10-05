import numpy as np
import pandas as pd

from demand_forecasting.features import FEATURE_COLUMNS, make_direct_supervised_data
from demand_forecasting.metrics import forecast_metrics
from demand_forecasting.monitoring import evaluate_forecast_log
from demand_forecasting.synthetic import generate_synthetic_demand


def test_synthetic_generation_is_reproducible_and_has_stockout_signal():
    first = generate_synthetic_demand(periods=90, n_products=2, random_state=7)
    second = generate_synthetic_demand(periods=90, n_products=2, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    assert {"latent_demand", "sales", "stockout"}.issubset(first.columns)
    assert (first["sales"] <= first["latent_demand"]).all()
    assert first["stockout"].any()


def test_direct_features_are_anchored_before_target_date():
    panel = generate_synthetic_demand(periods=100, n_products=1, random_state=11)
    horizon = 3
    features = make_direct_supervised_data(panel, horizon)
    row = features.iloc[0]
    history = panel.loc[panel["product_id"] == row["product_id"]].reset_index(drop=True)
    target_position = history.index[history["date"] == row["date"]][0]

    assert row["lag_1"] == history.loc[target_position - horizon, "sales"]
    assert row["lag_7"] == history.loc[target_position - horizon - 6, "sales"]
    assert not any("latent" in name for name in FEATURE_COLUMNS)


def test_metrics_handle_zero_total_demand():
    result = forecast_metrics(pd.Series([0, 0]), pd.Series([2, 1]))

    assert result["MAE"] == 1.5
    assert np.isnan(result["WAPE"])
    assert np.isnan(result["bias"])


def test_monitoring_ignores_pending_forecasts():
    log = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "product_id": ["P001", "P001"],
            "model": ["baseline", "baseline"],
            "y_pred": [10.0, 20.0],
            "y_true": [12.0, np.nan],
        }
    )

    metrics = evaluate_forecast_log(log)

    assert metrics.loc[0, "MAE"] == 2.0
