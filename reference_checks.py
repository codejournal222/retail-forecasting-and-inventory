import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from forecasting import GRIDS, hw


def reference_hw(history, horizon, alpha, beta, gamma):
    # Forecast the equivalent ETS(A,A,A), without optimizing any parameters.

    # Project recurrence: b_new = beta*(level_new-level_old)+(1-beta)*b_old.
    # Thus ETS trend innovation coefficient = alpha*beta.
    # Project season update uses y-level_new, hence ETS gamma=(1-alpha)*gamma.
    # Start both recurrences after the same first-week initialization.
    y = np.asarray(history, float)
    level = y[:7].mean()
    model = ETSModel(
        y[7:],
        error="add",
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="known",
        initial_level=level,
        initial_trend=(y[7:14].mean() - level) / 7,
        initial_seasonal=y[:7] - level,
    )
    # A perfectly fitted synthetic series has zero residual variance. Its
    # likelihood is degenerate, but the deterministic forecast remains valid
    with np.errstate(divide="ignore", invalid="ignore"):
        fitted = model.smooth([alpha, alpha * beta, (1 - alpha) * gamma])
    return np.maximum(0, np.asarray(fitted.forecast(horizon)))


def check_reference(series_map, origins, horizon, delay, output):
    """Check every HW grid candidate at each real-data validation origin."""
    rows = []
    for code, series in series_map.items():
        for origin in origins:
            history = series.iloc[: origin - delay].to_numpy()
            for params in GRIDS["Holt-Winters"]:
                actual = hw(history, horizon + delay, **params)
                reference = reference_hw(history, horizon + delay, **params)
                np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-8)
                rows.append(
                    {
                        "StockCode": code,
                        "Origin": str(series.index[origin].date()),
                        **params,
                        "MaxAbsoluteDifference": float(
                            np.max(np.abs(actual - reference))
                        ),
                    }
                )
    out = Path(output)
    pd.DataFrame(rows).to_csv(out / "holt_winters_reference.csv", index=False)
    result = {
        "status": "PASS",
        "comparisons_run": len(rows),
        "max_absolute_difference": max(x["MaxAbsoluteDifference"] for x in rows),
        "reference": "statsmodels ETS(A,A,A), known matching initialization, no optimization",
        "statsmodels_version": statsmodels.__version__,
        "parameter_mapping": "ETS alpha=alpha; ETS beta=alpha*beta; ETS gamma=(1-alpha)*gamma",
        "interpretation": "Numerical recurrence validation; not evidence of predictive superiority.",
    }
    (out / "holt_winters_reference.json").write_text(json.dumps(result, indent=2))
    return result
