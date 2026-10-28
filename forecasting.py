
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Model names, feature order and validation grids
MODELS = ["Moving average", "Seasonal naive", "Holt-Winters", "Random Forest"]
FEATURES = [
    "lag1",
    "lag7",
    "lag28",
    "mean7",
    "mean28",
    "std28",
    "weekday",
    "month",
    "day_index",
]
GRIDS = {
    "Moving average": [{"window": 7}, {"window": 28}],
    "Seasonal naive": [{}],
    "Holt-Winters": [
        {"alpha": a, "beta": b, "gamma": g}
        for a in (0.2, 0.5)
        for b in (0.0, 0.05)
        for g in (0.1, 0.3)
    ],
    "Random Forest": [
        {"depth": 6, "leaf": 3},
        {"depth": 10, "leaf": 2},
        {"depth": None, "leaf": 5},
    ],
}



# Evaluate forecast errors, including zero-demand days
def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if (
        a.shape != p.shape
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Metrics require matching finite nonempty arrays.")

    # Positive metric bias means overprediction
    e = p - a
    nz = a != 0
    den = np.abs(a).sum()
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "WAPE_pct": float(100 * np.abs(e).sum() / den) if den else None,
        "MAPE_nonzero_pct": (
            float(100 * np.mean(np.abs(e[nz] / a[nz]))) if nz.any() else None
        ),
        "MAPE_coverage_pct": float(100 * nz.mean()),
        "Bias": float(e.mean()),
        "N": len(a),
    }



# Additive Holt-Winters with weekly seasonality
def hw(train, horizon, alpha=0.2, beta=0.05, gamma=0.1):
    # Additive weekly smoother, cross-checked against statsmodels ETS(A,A,A).

    # This uses the updated level in the seasonal recurrence. Equivalent ETS
    # innovation coefficients are alpha, alpha*beta and (1-alpha)*gamma.
    # The first week initialises seasonality; updates start at observation 7.
    
    y = np.asarray(train, float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError(
            "Holt-Winters needs a finite, nonempty one-dimensional history."
        )
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    if any(not np.isfinite(x) or not 0 <= x <= 1 for x in (alpha, beta, gamma)):
        raise ValueError("Smoothing coefficients must lie between zero and one.")
    m = 7
    if len(y) < 14:
        return np.repeat(y.mean(), horizon)

    # Initialise level, trend and the seven seasonal offsets
    level = y[:m].mean()
    trend = (y[m : 2 * m].mean() - level) / m
    seasonal = y[:m] - level

    # Update the state one observed calendar day at a time
    for t in range(m, len(y)):
        old = level
        j = t % m
        level = alpha * (y[t] - seasonal[j]) + (1 - alpha) * (level + trend)
        trend = beta * (level - old) + (1 - beta) * trend
        seasonal[j] = gamma * (y[t] - level) + (1 - gamma) * seasonal[j]
    return np.maximum(
        0,
        [
            level + h * trend + seasonal[(len(y) + h - 1) % m]
            for h in range(1, horizon + 1)
        ],
    )



# Construct features from past observations only
def feature_row(history, date):
    v = np.asarray(history, float)
    return [
        v[-1],
        v[-7],
        v[-28],
        v[-7:].mean(),
        v[-28:].mean(),
        v[-28:].std(),
        date.dayofweek,
        date.month,
        len(v),
    ]



# Fit a model and produce a future daily forecast
def predict(train, model, horizon, params=None, closed_weekdays=(5,)):
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")
    if len(train) < 35:
        raise ValueError("At least 35 calendar days are required.")
    if not train.index.equals(
        pd.date_range(train.index[0], periods=len(train), freq="D")
    ):
        raise ValueError("Forecast input must have a regular daily calendar.")
    if not np.isfinite(train).all() or (train < 0).any():
        raise ValueError("Demand must be finite and nonnegative.")
    params = params or {}
    hist = list(train.to_numpy(float))
    dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon)

    # Each model receives the same historical calendar and horizon
    if model == "Moving average":
        recent = train.iloc[-params.get("window", 28) :]
        open_days = recent[~recent.index.dayofweek.isin(closed_weekdays)]
        out = np.repeat(float(open_days.mean()) if len(open_days) else 0.0, horizon)
    elif model == "Seasonal naive":
        out = np.tile(hist[-7:], int(np.ceil(horizon / 7)))[:horizon]
    elif model == "Holt-Winters":
        out = hw(hist, horizon, **params)
    elif model == "Random Forest":
        X = np.asarray(
            [feature_row(hist[:i], train.index[i]) for i in range(28, len(hist))]
        )
        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=params.get("depth", 6),
            min_samples_leaf=params.get("leaf", 3),
            random_state=42,
            n_jobs=1,
        )

        # Align each past-only feature row with its observed target
        rf.fit(X, np.asarray(hist[28:]))
        out = []

        # Recursively append predictions because future actuals are unknown
        for date in dates:
            p = max(0.0, float(rf.predict(np.asarray([feature_row(hist, date)]))[0]))
            if date.dayofweek in closed_weekdays:
                p = 0.0
            out.append(p)
            hist.append(p)  # recursive predictions, never future actuals
    else:
        raise ValueError(f"Unknown model {model}")

    # Apply the same closure rule and return dated predictions
    out = np.asarray(out, float)
    out[np.isin(dates.dayofweek, closed_weekdays)] = 0
    return pd.Series(out, index=dates)



# Allow recent cancellation labels to mature
def available_predict(
    history, model, horizon, params=None, closed_weekdays=(5,), reporting_delay=1
):
    """Leave time for cancellation labels to mature before using history."""
    mature = history.iloc[:-reporting_delay] if reporting_delay else history
    return predict(
        mature, model, horizon + reporting_delay, params, closed_weekdays
    ).iloc[reporting_delay:]



# Select parameters and estimate safety stock on validation data
def tune(series, origins, horizon, closed_weekdays=(5,), reporting_delay=1):
    chosen = {}
    rows = []
    calibration = []
    for model in MODELS:
        best = None
        for params in GRIDS[model]:
            preds = []
            actuals = []
            errors = []
            for origin in origins:
                p = available_predict(
                    series.iloc[:origin],
                    model,
                    horizon,
                    params,
                    closed_weekdays,
                    reporting_delay,
                ).to_numpy()
                a = series.iloc[origin : origin + horizon].to_numpy()
                preds.extend(p)
                actuals.extend(a)
                errors.append(float(a.sum() - p.sum()))

            # Compare candidate settings using pooled validation MAE
            score = metrics(actuals, preds)
            rows.append({"Model": model, "Parameters": str(params), **score})
            if best is None or score["MAE"] < best[0]:
                best = (score["MAE"], params, errors)
        chosen[model] = {
            "params": best[1],
            "validation_mae": best[0],
            "safety_stock": float(max(0, np.quantile(best[2], 0.9))),
            "calibration_n": len(best[2]),
        }
        calibration.extend(
            {
                "Model": model,
                "Origin": series.index[o],
                "Horizon": horizon,
                "CumulativeError": e,
            }
            for o, e in zip(origins, best[2])
        )
    return chosen, rows, calibration



# Select product strata using the initial training period
def select_products(demand, cutoff, n_each=3):
    """Choose strata using the initial training segment only, never the test data."""
    d = demand[demand.Date < cutoff]
    start = demand.Date.min()
    days = (cutoff - start).days
    g = d.groupby("StockCode").agg(
        Units=("Quantity", "sum"),
        ActiveDays=("Date", "nunique"),
        FirstDate=("Date", "min"),
    )
    g = g[(g.FirstDate <= start + pd.Timedelta(days=28)) & (g.ActiveDays >= 10)].copy()
    g["ZeroShare"] = 1 - g.ActiveDays / days
    g = g.sort_values(["Units", "StockCode"], ascending=[False, True])
    chosen = []
    for code in g.head(n_each).index:
        chosen.append((code, "high volume"))
    remaining = g.drop(index=[x[0] for x in chosen])
    mid = max(0, len(remaining) // 2 - n_each // 2)
    for code in remaining.iloc[mid : mid + n_each].index:
        chosen.append((code, "middle volume"))
    remaining = g.drop(index=[x[0] for x in chosen]).sort_values(
        ["ZeroShare", "Units"], ascending=[False, False]
    )
    for code in remaining.head(n_each).index:
        chosen.append((code, "intermittent"))
    return pd.DataFrame(
        [
            {
                "StockCode": c,
                "Stratum": s,
                "SelectionUnits": g.loc[c, "Units"],
                "SelectionZeroShare": g.loc[c, "ZeroShare"],
            }
            for c, s in chosen
        ]
    )
