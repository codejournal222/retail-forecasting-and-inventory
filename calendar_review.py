import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import MODELS, available_predict, metrics, tune


def past_weekday_fill(series, unknown_dates):
    # Replace flagged dates with mean of up to four earlier observed same weekdays.
    # This is a sensitivity scenario, not recovered demand. Imputed values are
    # excluded from the donor pool; neither interpolation nor future donors are used.
    
    result = series.copy()
    unknown = series.index.isin(unknown_dates)
    for position in np.flatnonzero(unknown):
        prior = series.iloc[:position]
        eligible = (~unknown[:position]) & (
            prior.index.dayofweek == series.index[position].dayofweek
        )
        donors = prior[eligible].tail(4)
        result.iloc[position] = float(donors.mean()) if len(donors) else 0.0
    return result


def review_calendar(calendar, series_map, origins, starts, config, delay, output):
    out = Path(output)
    unknown = pd.DatetimeIndex(calendar.loc[calendar.ZeroAssumptionNeedsReview, "Date"])
    review = calendar.loc[calendar.ZeroAssumptionNeedsReview, ["Date"]].copy()
    review["Weekday"] = review.Date.dt.day_name()
    review["Evidence"] = (
        "No rows in supplied source; closure versus missing extraction unconfirmed"
    )
    review["BaselineTreatment"] = "Zero recorded sales; not proven zero demand"
    review["RequiredEvidence"] = (
        "Business trading calendar or original extraction/fulfilment records"
    )
    review.to_csv(out / "calendar_review.csv", index=False)
    horizon = config["lead_days"] + config["review_days"]
    dates = next(iter(series_map.values())).index
    # Do not tune or score using a fabricated validation target.
    valid_origins = [
        o for o in origins if not dates[o : o + horizon].isin(unknown).any()
    ]
    if not valid_origins:
        result = {
            "status": "NOT_EVALUATED",
            "reason": "No validation horizon has fully observed source dates.",
        }
        (out / "calendar_sensitivity.json").write_text(json.dumps(result, indent=2))
        pd.DataFrame(columns=["Scenario", "Model", "MAE", "N"]).to_csv(
            out / "calendar_sensitivity.csv", index=False
        )
        return result
    scored = []
    scenarios = ("Zero recorded sales", "Past weekday mean (sensitivity only)")
    for code, series in series_map.items():
        for scenario in scenarios:
            training = (
                series
                if scenario == scenarios[0]
                else past_weekday_fill(series, unknown)
            )
            fitted, _, _ = tune(
                training, valid_origins, horizon, config["closed_weekdays"], delay
            )
            for fold, start in enumerate(starts, 1):
                for offset in range(0, 28, config["review_days"]):
                    origin = start + offset
                    for model in MODELS:
                        prediction = available_predict(
                            training.iloc[:origin],
                            model,
                            config["review_days"],
                            fitted[model]["params"],
                            config["closed_weekdays"],
                            delay,
                        )
                        for step in range(min(config["review_days"], 28 - offset)):
                            date = dates[origin + step]
                            if date in unknown:
                                continue
                            scored.append(
                                {
                                    "Scenario": scenario,
                                    "StockCode": code,
                                    "Fold": fold,
                                    "Model": model,
                                    "Date": date,
                                    "Actual": float(series.iloc[origin + step]),
                                    "Predicted": float(prediction.iloc[step]),
                                }
                            )
    detail = pd.DataFrame(scored)
    summary = pd.DataFrame(
        [
            {
                "Scenario": scenario,
                "Model": model,
                **metrics(group.Actual, group.Predicted),
            }
            for (scenario, model), group in detail.groupby(
                ["Scenario", "Model"], sort=False
            )
        ]
    )
    detail.to_csv(out / "calendar_sensitivity_detail.csv", index=False)
    summary.to_csv(out / "calendar_sensitivity.csv", index=False)
    result = {
        "status": "COMPLETED",
        "unknown_dates": len(unknown),
        "test_unknown_dates": [
            str(date.date())
            for start in starts
            for date in dates[start : start + 28]
            if date in unknown
        ],
        "validation_origins_used": [str(dates[o].date()) for o in valid_origins],
        "comparison": "Same products/origins; both scenarios scored only on dates with source coverage or configured closure.",
        "limitation": "Alternative is a sensitivity assumption, not recovered demand. No inventory outcomes are claimed for imputed demand.",
    }
    (out / "calendar_sensitivity.json").write_text(json.dumps(result, indent=2))
    return result
