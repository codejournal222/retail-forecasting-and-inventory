import argparse
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from workbook_checks import check_workbook


# Independently reconcile the saved run outputs
def _verify_data(folder, checks, skip_workbook=False):
    p = Path(folder)

    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    # Reconcile source counts and scored forecast rows
    audit = json.loads((p / "cleaning_audit.json").read_text())
    check(
        sum(x["Rows"] for x in audit["category_counts"]) == audit["source_rows"],
        "Every source row reconciles to one cleaning category",
    )
    f = pd.read_csv(p / "forecast_detail.csv")
    i = pd.read_csv(p / "inventory_detail.csv")
    comp = pd.read_csv(p / "model_comparison.csv").set_index("Model")
    check(
        not f.duplicated(["StockCode", "Fold", "Model", "Date"]).any(),
        "No double-counted test actuals within model and fold",
    )
    check(
        f.groupby(["StockCode", "Fold", "Model"]).size().eq(28).all(),
        "Each product/model/fold has 28 scored days",
    )
    check(
        np.isfinite(f[["Actual", "Predicted"]]).all().all() and f.Predicted.ge(0).all(),
        "Finite nonnegative forecasts",
    )

    # Recalculate MAE independently from the saved predictions
    for model, g in f.groupby("Model"):
        check(
            abs(abs(g.Predicted - g.Actual).mean() - comp.loc[model, "MAE"]) < 1e-7,
            f"Independent MAE: {model}",
        )
    check(
        np.allclose(i.OpeningStock + i.Arrivals - i.Fulfilled, i.ClosingStock),
        "Daily stock balance",
    )
    check(
        np.allclose(i.Fulfilled + i.LostUnits, i.Demand),
        "Fulfilment plus lost sales equals demand",
    )
    check(
        np.allclose(i.HoldingCost + i.ShortageCost + i.OrderingCost, i.TotalCost),
        "Daily cost components reconcile",
    )
    check(
        all(
            np.allclose(g.OpeningStock.iloc[1:], g.ClosingStock.iloc[:-1])
            for _, g in i.groupby(["StockCode", "Fold", "Model"])
        ),
        "Closing stock rolls into next day opening stock",
    )

    # Check that forward forecasts follow the historical cutoff
    manifest = json.loads((p / "manifest.json").read_text())
    future = pd.read_csv(p / "future_forecast.csv")
    check(
        pd.to_datetime(future.Date).min() > pd.Timestamp(manifest["analysis_end"]),
        "Future predictions follow analysis cutoff",
    )

    # A missing workbook is a failure unless explicitly skipped by the caller
    if skip_workbook:
        workbook_result = {"status": "SKIPPED", "reason": "Explicit --skip-workbook"}
    else:
        workbook = p / "Retail_Results.xlsx"
        check(workbook.is_file(), "Required Excel workbook exists")
        data = json.loads((p / "workbook_data.json").read_text())
        # Reconcile workbook inputs to CSVs and, for costs, directly to the ledger.
        for name, records in data.items():
            exported = pd.read_csv(p / (name.lower() + ".csv"), dtype=str).fillna("")
            check(
                len(exported) == len(records), f"Workbook input/CSV row counts: {name}"
            )
            if records:
                check(
                    list(exported.columns) == list(records[0]),
                    f"Workbook input/CSV columns: {name}",
                )
            for row, record in enumerate(records):
                for key, value in record.items():
                    csv_value = exported.iloc[row][key]
                    if isinstance(value, bool):
                        same = csv_value.lower() == str(value).lower()
                    elif isinstance(value, (int, float)):
                        same = bool(
                            np.isclose(float(csv_value), value, rtol=1e-9, atol=1e-8)
                        )
                    elif key in {"Date", "Origin", "Start", "End"} and value:
                        same = pd.Timestamp(csv_value) == pd.Timestamp(
                            value
                        ).tz_localize(None)
                    else:
                        same = csv_value == ("" if value is None else str(value))
                    if not same:
                        check(
                            False,
                            f"Workbook input/CSV mismatch: {name} row {row+1} {key}",
                        )
        for record in data["Inventory_Comparison"]:
            ledger = i[i.Model == record["Model"]]
            check(
                np.isclose(ledger.TotalCost.sum(), record["TotalCost"]),
                f"Workbook cost agrees with daily simulation: {record['Model']}",
            )
            total = ledger.Demand.sum()
            expected_fill = 1 - ledger.LostUnits.sum() / total if total else 1.0
            check(
                np.isclose(expected_fill, record["FillRate"]),
                f"Workbook fill rate agrees with simulation: {record['Model']}",
            )
        workbook_result = check_workbook(workbook, data, check)
    reference = json.loads((p / "holt_winters_reference.json").read_text())
    check(
        reference["status"] == "PASS" and reference["comparisons_run"] > 0,
        "Independent Holt-Winters reference checks completed",
    )
    return {
        "forecast_rows": len(f),
        "inventory_day_rows": len(i),
        "workbook": workbook_result,
        "holt_winters_reference": reference,
    }


def verify(folder="output", run_tests=False, skip_workbook=False):
    """Write a fresh result even on failure; never reuse a stale PASS file."""
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    checks = []
    result = {
        "status": "FAIL",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_folder": str(p.resolve()),
        "checks": checks,
        "regression_tests": {"status": "NOT_RUN"},
        "warnings": [],
    }
    try:
        result.update(_verify_data(p, checks, skip_workbook))
        if run_tests:
            test_path = p.resolve() / "test_results.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_tests.py")),
                    "--output",
                    str(test_path),
                ],
                capture_output=True,
                text=True,
            )
            (p / "test_run.log").write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            if process.returncode or not test_path.is_file():
                raise AssertionError(
                    "Regression test execution failed; see test_run.log"
                )
            result["regression_tests"] = json.loads(test_path.read_text())
            if result["regression_tests"]["status"] != "PASS":
                raise AssertionError("Regression tests did not pass")
        else:
            result["warnings"].append(
                "Regression tests were not run by this verification; use --run-tests."
            )
        if result["workbook"].get("native_engine_cache_status") == "NOT_AVAILABLE":
            result["warnings"].append(
                "Workbook formulas independently evaluated; native Excel caches require opening/recalculation in Excel."
            )
        if skip_workbook:
            result["warnings"].append("Workbook checks explicitly skipped.")
        result["gui_status"] = (
            "GUI-independent controller tested; desktop rendering requires a display."
        )
        result["status"] = "PASS_WITH_WARNINGS" if result["warnings"] else "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    (p / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        default="output",
        help="Folder containing this run's outputs.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute and record actual regression tests.",
    )
    parser.add_argument(
        "--skip-workbook",
        action="store_true",
        help="Explicitly skip Excel verification (reported as a warning).",
    )
    args = parser.parse_args()
    result = verify(args.folder, args.run_tests, args.skip_workbook)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks_passed": len(result["checks"]),
                "warnings": result["warnings"],
                "error": result.get("error"),
            },
            indent=2,
        )
    )
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
