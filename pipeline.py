import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy

from cleaning import load_source, clean_transactions, match_cancellations
from forecasting import MODELS, metrics, tune, select_products, available_predict
from inventory import simulate


# Default experimental and inventory assumptions
DEFAULTS = {
    "lead_days": 7,
    "review_days": 7,
    "future_days": 14,
    "n_each_stratum": 3,
    "cancellation_hours": 24,
    "exclude_last_day": True,
    "closed_weekdays": [5],
    "holding_rate": 0.2,
    "shortage_fraction": 0.5,
    "order_cost": 5.0,
    "initial_stock_days": 14,
    "stock_overrides": {},
    "products": [],
    "unknown_day_policy": "zero",
}



# Shared export and configuration helpers
def save_json(path, obj):
    def convert(x):
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            return str(x)[:10]
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    Path(path).write_text(
        json.dumps(obj, indent=2, default=convert, allow_nan=False), encoding="utf-8"
    )


def validate_config(c):
    for k in ["lead_days", "review_days", "future_days", "n_each_stratum"]:
        if type(c[k]) is not int or not 1 <= c[k] <= 60:
            raise ValueError(f"{k} must be an integer between 1 and 60.")
    if c["lead_days"] + c["review_days"] > 28:
        raise ValueError("This experiment supports lead + review up to 28 days.")
    for k in [
        "holding_rate",
        "shortage_fraction",
        "order_cost",
        "initial_stock_days",
        "cancellation_hours",
    ]:
        if not np.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f"{k} must be finite and nonnegative.")
    if c["cancellation_hours"] > 48:
        raise ValueError(
            "Cancellation windows above 48 hours require a different availability design."
        )
    if any(x not in range(7) for x in c["closed_weekdays"]):
        raise ValueError("Closed weekdays must be 0 (Monday) through 6 (Sunday).")
    for code, stock in c["stock_overrides"].items():
        if not np.isfinite(stock) or stock < 0:
            raise ValueError(f"Invalid current stock for {code}")
    if c["unknown_day_policy"] not in {"zero", "error"}:
        raise ValueError(
            "unknown_day_policy must be 'zero' (explicit assumption) or 'error' (strict)."
        )



# Objectives 1-8: Run the experiment
def run(input_path, output="output", config=None, source_override=None, progress=print):

    # Set up this run and validate its assumptions
    c = {**DEFAULTS, **(config or {})}
    validate_config(c)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # Objective 1: Import the supplied records
    progress("1/9 Import and validation")
    if source_override is None:
        source, provenance = load_source(input_path)
    else:
        source = source_override.copy()
        p = Path(input_path)
        provenance = {
            "input_name": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "reconstructed_from_cleaned_workbook": True,
            "provenance_limit": "Previously discarded rows cannot be recovered.",
        }

    # Objective 2: Clean, match and retain the row-level audit
    progress("2/9 Cleaning and cancellation matching")
    audit_rows, demand, pairs, audit = clean_transactions(
        source, c["cancellation_hours"]
    )
    source.to_csv(out / "source_records.csv.gz", index=False, compression="gzip")
    audit_rows.to_csv(out / "transaction_audit.csv.gz", index=False, compression="gzip")
    demand.to_csv(out / "demand_cleaned.csv.gz", index=False, compression="gzip")
    pairs.to_csv(out / "cancellation_pairs.csv", index=False)
    save_json(out / "cleaning_audit.json", audit)

    # Check how the cancellation-window assumption changes retained demand
    sens = []
    base = audit_rows.copy()
    base.loc[base.Category == "Inferred_Cancelled", "Category"] = "Demand"
    for hours in sorted(set([0, 1, 24, c["cancellation_hours"]])):
        pp = match_cancellations(base, hours)
        sens.append(
            {
                "WindowHours": hours,
                "MatchedOrders": len(pp),
                "UnitsRemoved": float(base.loc[pp.OrderID, "Quantity"].sum()),
            }
        )
    pd.DataFrame(sens).to_csv(out / "cleaning_sensitivity.csv", index=False)

    # Build a complete calendar and select products before validation
    progress("3/9 Daily calendar and product selection")
    start = demand.Date.min()
    last = demand.Date.max()
    end = last - pd.Timedelta(days=1) if c["exclude_last_day"] else last
    dates = pd.date_range(start, end)
    if len(dates) < 280:
        raise ValueError(
            "At least 280 calendar days are needed for the three-window experiment."
        )

    # Define chronological validation and held-out test windows
    starts = [len(dates) - 126, len(dates) - 70, len(dates) - 28]
    H = c["lead_days"] + c["review_days"]
    delay = math.ceil(c["cancellation_hours"] / 24)
    # Four non-overlapping validation horizons, all ending before first test
    val_origins = [starts[0] - 4 * H + i * H for i in range(4)]
    if val_origins[0] - delay < 60:
        raise ValueError(
            "Not enough pre-validation history for this lead/review choice."
        )
    selection_cutoff = dates[val_origins[0] - delay]
    if c["products"]:
        codes = [str(x).strip().upper() for x in c["products"]]
        absent = set(codes) - set(demand.StockCode)
        if absent:
            raise ValueError(f"Unknown products {sorted(absent)}")
        selected = pd.DataFrame(
            {"StockCode": list(dict.fromkeys(codes)), "Stratum": "user selected"}
        )
    else:
        selected = select_products(demand, selection_cutoff, c["n_each_stratum"])
    if selected.empty:
        raise ValueError("No products have sufficient history.")
    descriptions = demand.groupby("StockCode").Description.agg(
        lambda x: x.dropna().iloc[0] if x.notna().any() else ""
    )
    selected["Description"] = selected.StockCode.map(descriptions)
    selected.to_csv(out / "products.csv", index=False)

    # Flag unknown zero-sales dates and check the closure assumption
    calendar = pd.DataFrame({"Date": dates})
    observed = set(
        source.InvoiceDate.pipe(pd.to_datetime, errors="coerce").dt.normalize().dropna()
    )
    calendar["AnySourceRecords"] = calendar.Date.isin(observed)
    calendar["AssumedClosed"] = calendar.Date.dt.dayofweek.isin(c["closed_weekdays"])
    calendar["ZeroAssumptionNeedsReview"] = (
        ~calendar.AnySourceRecords & ~calendar.AssumedClosed
    )
    calendar.to_csv(out / "calendar_audit.csv", index=False)
    unknown_count = int(calendar.ZeroAssumptionNeedsReview.sum())
    if unknown_count:
        progress(
            f"Calendar review: {unknown_count} non-closed dates have no source records; closure is unconfirmed."
        )
        if c["unknown_day_policy"] == "error":
            raise ValueError(
                "Unresolved calendar gaps. Review calendar_audit.csv or explicitly select unknown_day_policy='zero'."
            )
    series_map = {
        code: demand[demand.StockCode == code]
        .groupby("Date")
        .Quantity.sum()
        .reindex(dates, fill_value=0)
        .astype(float)
        for code in selected.StockCode
    }
    if any(
        s[s.index.dayofweek.isin(c["closed_weekdays"])].sum() > 0
        for s in series_map.values()
    ):
        raise ValueError(
            "Observed sales conflict with configured closed weekdays. Correct the calendar assumption."
        )
    daily = pd.DataFrame(series_map)
    daily.index.name = "Date"
    daily.to_csv(out / "daily_product_sales.csv")
    sales = daily.reset_index().melt(
        id_vars="Date", var_name="StockCode", value_name="Units"
    )
    revenue = (
        demand.assign(RevenueGBP=demand.Quantity * demand.UnitPrice)
        .groupby(["Date", "StockCode"])
        .RevenueGBP.sum()
    )
    sales = sales.merge(revenue.reset_index(), on=["Date", "StockCode"], how="left")
    sales["RevenueGBP"] = sales.RevenueGBP.fillna(0)
    sales = sales.merge(calendar, on="Date", how="left")
    progress("Independent Holt-Winters reference checks")
    from reference_checks import check_reference

    hw_check = check_reference(series_map, val_origins, H, delay, out)
    progress("Calendar sensitivity: compare zero filling with a past-only alternative")
    from calendar_review import review_calendar

    calendar_check = review_calendar(
        calendar, series_map, val_origins, starts, c, delay, out
    )

    # Prepare output records for the forecasting and inventory experiments
    tuning = []
    calibration = []
    test_rows = []
    protection_rows = []
    inv_rows = []
    inv_stats = []
    future = []
    recommend = []
    params_all = {}
    cost_sens = []
    folds = []
    for k, o in enumerate(starts):
        folds.append({"Fold": k + 1, "Start": dates[o], "End": dates[o + 27]})

    # Fit, validate and evaluate each selected product independently
    for j, (code, s) in enumerate(series_map.items()):
        progress(
            f"4–6/9 Product {j+1}/{len(series_map)}: {code} — validation, forecasting, inventory"
        )
        fitted, tr, cr = tune(s, val_origins, H, c["closed_weekdays"], delay)
        params_all[code] = fitted
        tuning.extend({"StockCode": code, **x} for x in tr)
        calibration.extend({"StockCode": code, **x} for x in cr)
        winner = min(MODELS, key=lambda m: fitted[m]["validation_mae"])
        # Prices are fixed using only initial training observations, never future prices
        price_rows = demand[
            (demand.StockCode == code)
            & (demand.Date < selection_cutoff)
            & (demand.UnitPrice > 0)
        ]
        price = float(price_rows.UnitPrice.median()) if len(price_rows) else 1.0

        # Compare every model under equal starting stock in each test window
        for fold, o in enumerate(starts, 1):
            initial = math.ceil(
                s.iloc[max(0, o - 28 - delay) : o - delay].mean()
                * c["initial_stock_days"]
            )
            for model in MODELS:
                forecasts = {}
                for t in range(0, 28, c["review_days"]):
                    origin = o + t
                    pred = available_predict(
                        s.iloc[:origin],
                        model,
                        max(H, c["review_days"]),
                        fitted[model]["params"],
                        c["closed_weekdays"],
                        delay,
                    )
                    forecasts[t] = pred.to_numpy()
                    if t + H <= 28:
                        protection_rows.append(
                            {
                                "StockCode": code,
                                "Fold": fold,
                                "Model": model,
                                "Origin": dates[origin],
                                "HorizonDays": H,
                                "Actual": float(s.iloc[origin : origin + H].sum()),
                                "Predicted": float(pred.iloc[:H].sum()),
                            }
                        )
                    # Each actual day occurs once per fold/model; horizons 1..review.
                    for h in range(min(c["review_days"], 28 - t)):
                        test_rows.append(
                            {
                                "StockCode": code,
                                "Fold": fold,
                                "Model": model,
                                "Origin": dates[origin],
                                "Date": dates[origin + h],
                                "Horizon": h + 1,
                                "Actual": float(s.iloc[origin + h]),
                                "Predicted": float(pred.iloc[h]),
                            }
                        )
                rows, summary = simulate(
                    s.iloc[o : o + 28],
                    forecasts,
                    initial,
                    fitted[model]["safety_stock"],
                    c["lead_days"],
                    c["review_days"],
                    price,
                    c["holding_rate"],
                    c["shortage_fraction"],
                    c["order_cost"],
                )
                inv_rows.extend(
                    {"StockCode": code, "Fold": fold, "Model": model, **x} for x in rows
                )
                inv_stats.append(
                    {
                        "StockCode": code,
                        "Fold": fold,
                        "Model": model,
                        "InitialStock": initial,
                        "PriceProxy": price,
                        **summary,
                    }
                )
                for label, holding, shortage in [
                    ("low shortage", c["holding_rate"], 0.25),
                    ("high shortage", c["holding_rate"], 1.0),
                    ("high holding", 0.4, c["shortage_fraction"]),
                ]:
                    _, ss = simulate(
                        s.iloc[o : o + 28],
                        forecasts,
                        initial,
                        fitted[model]["safety_stock"],
                        c["lead_days"],
                        c["review_days"],
                        price,
                        holding,
                        shortage,
                        c["order_cost"],
                    )
                    cost_sens.append(
                        {
                            "Scenario": label,
                            "StockCode": code,
                            "Fold": fold,
                            "Model": model,
                            "TotalCost": ss["TotalCost"],
                            "LostUnits": ss["LostUnits"],
                        }
                    )

        # Generate the forward forecast using the validation-selected model
        p = available_predict(
            s,
            winner,
            max(c["future_days"], H),
            fitted[winner]["params"],
            c["closed_weekdays"],
            delay,
        )
        for date, value in p.iloc[: c["future_days"]].items():
            future.append(
                {
                    "StockCode": code,
                    "Model": winner,
                    "Date": date,
                    "Forecast": float(value),
                }
            )
        stock = float(
            c["stock_overrides"].get(
                code,
                math.ceil(
                    s.iloc[-28 - delay : -delay if delay else None].mean()
                    * c["initial_stock_days"]
                ),
            )
        )
        safety = fitted[winner]["safety_stock"]
        lead_demand = float(p.iloc[: c["lead_days"]].sum())
        protection = float(p.iloc[:H].sum())
        recommend.append(
            {
                "StockCode": code,
                "Description": descriptions.get(code, ""),
                "SelectedModel": winner,
                "LeadDays": c["lead_days"],
                "ReviewDays": c["review_days"],
                "ForecastLeadUnits": lead_demand,
                "ForecastProtectionUnits": protection,
                "SafetyStock": safety,
                "CurrentStock": stock,
                "OnOrder": 0.0,
                "StockSource": (
                    "user supplied"
                    if code in c["stock_overrides"]
                    else "illustrative assumption"
                ),
                "ReorderPoint": lead_demand + safety,
                "OrderUpTo": protection + safety,
                "SuggestedOrder": max(0, math.ceil(protection + safety - stock)),
                "CalibrationN": fitted[winner]["calibration_n"],
            }
        )

    # Combine detailed predictions and inventory outcomes into summaries
    predictions = pd.DataFrame(test_rows)
    inventory = pd.DataFrame(inv_stats)
    comparison = pd.DataFrame(
        [
            {"Model": m, **metrics(g.Actual, g.Predicted)}
            for m, g in predictions.groupby("Model", sort=False)
        ]
    )
    detail = pd.DataFrame(
        [
            {
                "StockCode": code,
                "Fold": fold,
                "Model": m,
                **metrics(g.Actual, g.Predicted),
            }
            for (code, fold, m), g in predictions.groupby(
                ["StockCode", "Fold", "Model"]
            )
        ]
    )
    inv_summary = (
        inventory.groupby("Model", sort=False)
        .agg(
            Demand=("Demand", "sum"),
            LostUnits=("LostUnits", "sum"),
            StockoutDays=("StockoutDays", "sum"),
            Orders=("Orders", "sum"),
            AverageStock=("AverageStock", "mean"),
            FinalStock=("FinalStock", "sum"),
            FinalOnOrder=("FinalOnOrder", "sum"),
            HoldingCost=("HoldingCost", "sum"),
            ShortageCost=("ShortageCost", "sum"),
            OrderingCost=("OrderingCost", "sum"),
            TotalCost=("TotalCost", "sum"),
        )
        .reset_index()
    )
    inv_summary["FillRate"] = np.where(
        inv_summary.Demand > 0,
        1 - inv_summary.LostUnits / inv_summary.Demand.replace(0, np.nan),
        1.0,
    )
    protection = pd.DataFrame(protection_rows)
    protection_comparison = pd.DataFrame(
        [
            {"Model": m, **metrics(g.Actual, g.Predicted)}
            for m, g in protection.groupby("Model", sort=False)
        ]
    )

    # Collect named tables for CSV, workbook and report exports
    tables = {
        "Sales": sales,
        "Calendar_Review": pd.read_csv(out / "calendar_review.csv"),
        "Calendar_Sensitivity": pd.read_csv(out / "calendar_sensitivity.csv"),
        "HW_Reference": pd.read_csv(
            out / "holt_winters_reference.csv", dtype={"StockCode": str}
        ),
        "Model_Comparison": comparison,
        "Forecast_Detail": predictions,
        "Fold_Metrics": detail,
        "Protection_Comparison": protection_comparison,
        "Protection_Forecasts": protection,
        "Inventory_Comparison": inv_summary,
        "Inventory_Detail": pd.DataFrame(inv_rows),
        "Inventory_Folds": inventory,
        "Future_Forecast": pd.DataFrame(future),
        "Replenishment": pd.DataFrame(recommend),
        "Validation": pd.DataFrame(tuning),
        "Calibration": pd.DataFrame(calibration),
        "Products": selected,
        "Cleaning": pd.DataFrame(audit["category_counts"]),
        "Cleaning_Sensitivity": pd.DataFrame(sens),
        "Cost_Sensitivity": pd.DataFrame(cost_sens),
        "Test_Windows": pd.DataFrame(folds),
    }
    progress("7–8/9 Reports and export data")
    for name, table in tables.items():
        table.to_csv(out / (name.lower() + ".csv"), index=False)

    # Record inputs, settings, source hashes and environment versions
    manifest = {
        "version": "2.1.0",
        "holt_winters_reference": hw_check,
        "calendar_sensitivity": calendar_check,
        "workbook_schema_version": 2,
        "provenance": provenance,
        "config": c,
        "reporting_delay_days": delay,
        "analysis_start": start,
        "analysis_end": end,
        "excluded_terminal_date": last if c["exclude_last_day"] else None,
        "selection_cutoff": selection_cutoff,
        "validation_origins": [dates[i] for i in val_origins],
        "test_windows": folds,
        "parameters": params_all,
        "calendar_unknown_zero_days": int(calendar.ZeroAssumptionNeedsReview.sum()),
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    save_json(out / "manifest.json", manifest)
    from reporting import make_report

    make_report(demand, tables, audit, manifest, out)
    # Numeric/date data for the bundled workbook renderer. CSV audit files retain full detail
    save_json(
        out / "workbook_data.json",
        {
            k: json.loads(v.to_json(orient="records", date_format="iso"))
            for k, v in tables.items()
            if k not in ["Inventory_Detail", "Forecast_Detail"]
        },
    )
    progress(f'Completed. Open {out / "report.html"}; launch app.py for objective 9.')
    return tables, manifest



# Command-line entry point
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", default="output")
    parser.add_argument("--config")
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Export the complete reproducible workbook, including Summary and Sales.",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    run(args.input, args.output, config)
    if args.excel:
        from local_excel import export_excel

        export_excel(args.output)


if __name__ == "__main__":
    main()
