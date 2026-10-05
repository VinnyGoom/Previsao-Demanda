"""Deterministic synthetic sales data for notebook experiments.

The generated ``latent_demand`` column is simulation-only ground truth. A real
sales system would not know this value for days when stock was unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_demand(
    start_date: str = "2023-01-01",
    periods: int = 730,
    n_products: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate daily demand and partially censored sales for multiple products.

    Demand combines product-specific volume, trend, weekday effects and annual
    seasonality. Some days have limited inventory, so observed sales can be
    lower than latent demand. The fixed seed makes experiments reproducible.
    """
    if periods < 60:
        raise ValueError("periods must be at least 60 to support lag features")
    if n_products < 1:
        raise ValueError("n_products must be at least 1")

    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    weekday_effects = np.array([0.92, 0.94, 0.98, 1.00, 1.12, 1.25, 1.18])
    rows: list[pd.DataFrame] = []

    for product_number in range(n_products):
        product_id = f"P{product_number + 1:03d}"
        base_demand = float(rng.uniform(18, 85))
        daily_trend = float(rng.uniform(-0.0002, 0.0005))
        phase = float(rng.uniform(0, 2 * np.pi))
        annual_amplitude = float(rng.uniform(0.05, 0.22))
        day_index = np.arange(periods)
        seasonal_year = 1 + annual_amplitude * np.sin(
            2 * np.pi * day_index / 365.25 + phase
        )
        trend = np.maximum(0.5, 1 + daily_trend * day_index)
        expected_demand = (
            base_demand
            * weekday_effects[dates.dayofweek]
            * seasonal_year
            * trend
        )
        latent_demand = rng.poisson(np.maximum(expected_demand, 0.1))

        # A small number of days are unavailable; otherwise stock is finite.
        unavailable = rng.random(periods) < 0.025
        inventory_capacity = np.maximum(
            0, rng.poisson(expected_demand * rng.uniform(0.75, 1.55))
        )
        inventory_capacity[unavailable] = 0
        sales = np.minimum(latent_demand, inventory_capacity)
        stockout = sales < latent_demand

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product_id": product_id,
                    "latent_demand": latent_demand.astype(int),
                    "sales": sales.astype(int),
                    "inventory_available": inventory_capacity > 0,
                    "stockout": stockout,
                }
            )
        )

    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["product_id", "date"]).reset_index(drop=True)
