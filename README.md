# Retail demand forecasting and inventory simulation

A Python study of whether machine-learning demand forecasts lead to
better inventory decisions than simple statistical baselines. It uses the public
[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) data
(a UK online gift retailer, 1 Dec 2010 to 9 Dec 2011).

**Research question:** do moving-average, seasonal-naive, Holt-Winters and
Random Forest forecasts differ in forecast accuracy and in simulated inventory
cost and service level under a periodic-review order-up-to policy?

## What it does

| Stage | Description |
|---|---|
| 1. Import | Reads the raw UCI workbook/CSV (or a previously cleaned, multi-sheet workbook) with row-level provenance and a SHA-256 hash |
| 2. Clean | Classifies every row into one exclusive category (demand, returns, adjustments, non-product charges, duplicates, invalid, review) and infers fully cancelled orders by optimal one-to-one matching |
| 3. Calendar | Builds a full daily calendar, flags dates with no source records, and tests how sensitive results are to that |
| 4. Forecast | Tunes four models on chronological validation windows; evaluates on three later 28-day test windows |
| 5. Evaluate | MAE, RMSE, WAPE, zero-safe MAPE and bias, plus accuracy over the lead-time + review protection period |
| 6. Inventory | Simulates lost-sales replenishment with identical starting stock and costs for every model |
| 7. Report | Self-contained `report.html` with eight charts and the audit trail |
| 8. Excel | `Retail_Results.xlsx` with Summary, Sales, Products, Forecasts, Inventory and Model_Comparison sheets; stock cells are editable and order quantities recalculate |
| 9. Interface | Optional Tkinter desktop app (`app.py`) |

Nine products are analysed by default (three high-volume, three mid-volume, three
intermittent), chosen from training data only.

## Quick start

Requires Python 3.12

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt   # exact tested versions
# or: python -m pip install -r requirements.txt    # looser version ranges

python run_tests.py                                # 28 tests
python pipeline.py "Online Retail.xlsx" --output output --config config.json --excel
python verify_outputs.py output --run-tests        # independent checks on the results
```

**Data is not included.** Download `Online Retail.xlsx` from the
[UCI repository](https://archive.ics.uci.edu/dataset/352/online+retail) and pass
its path to `pipeline.py`. A full run takes about 3.5 minutes on one CPU core.

Results are written to `output/`; open `output/report.html` first.

### Desktop interface

```bash
python app.py
```

Needs Python with Tk support (included with standard Windows and macOS installers;
on Debian/Ubuntu install `python3-tk`). The analysis code and tests do not need
Tk.

## Configuration

Defaults live in `config.json` and can be overridden by the app form.

| Setting | Default | Meaning |
|---|---|---|
| `lead_days` / `review_days` | 7 / 7 | Supplier lead time and review interval (their sum must be 28 or less) |
| `future_days` | 14 | Days forecast beyond the data |
| `n_each_stratum` | 3 | Products per volume group |
| `products` | `[]` | Specific stock codes; empty means automatic selection |
| `cancellation_hours` | 24 | Window for matching a cancellation to its order |
| `closed_weekdays` | `[5]` | Weekdays assumed closed (5 = Saturday) |
| `exclude_last_day` | `true` | Drop the partial final day (9 Dec 2011) |
| `holding_rate` / `shortage_fraction` / `order_cost` | 0.2 / 0.5 / 5.0 | Illustrative annual holding rate, lost-sale penalty as a share of price, and fixed cost per order (GBP) |
| `stock_overrides` | `{}` | Current stock by product code |
| `unknown_day_policy` | `"zero"` | `"zero"` treats undocumented dates as zero sales and warns; `"error"` stops the run |

## Results on the default run

Three held-out 28-day windows, nine products (lower is better, except fill rate).

| Model | Daily MAE (units) | Protection-period MAE | Simulated cost (GBP) | Fill rate |
|---|---|---|---|---|
| Moving average | **31.3** | 226.7 | 1,740 | 78.7% |
| Seasonal naive | 37.8 | **221.1** | 1,577 | 80.1% |
| Holt-Winters | 37.0 | 415.8 | 2,350 | 76.3% |
| Random Forest | 41.6 | 384.3 | **1,575** | **81.7%** |

The simple moving average is the most accurate on daily error. Random Forest has
the lowest simulated cost and highest fill rate, but only just ahead of seasonal
naive (about 0.15% lower cost), so no clear machine-learning advantage appears.
This is a small, exploratory sample, not a general claim about ML versus
traditional methods. See the limitations below.

## Repository layout

| File | Purpose |
|---|---|
| `pipeline.py` | Runs the whole experiment from the command line |
| `cleaning.py` | Import, row classification, cancellation matching and audit |
| `forecasting.py` | The four models, metrics, tuning and product selection |
| `inventory.py` | Periodic-review lost-sales simulation |
| `calendar_review.py` | Flags undocumented dates and runs the zero-fill sensitivity test |
| `reference_checks.py` | Checks the hand-written Holt-Winters against statsmodels ETS(A,A,A) |
| `reporting.py` | Charts and the HTML report |
| `local_excel.py` | Excel exporter |
| `workbook_checks.py` | Validates the workbook's contents and formulas |
| `verify_outputs.py` | Re-checks a finished output folder and records the result |
| `app.py` | Optional desktop interface |
| `test_pipeline.py`, `test_quality.py` | Regression tests (found automatically by `run_tests.py`) |
| `run_tests.py` | Runs all tests and writes `test_results.json` |
| `config.json` | Default settings |

## Method notes

- **Cancellations:** a full reversal is inferred only when customer, product,
  quantity and price all match and the cancellation falls within the window.
  Matching maximises the number of pairs, then minimises total delay. Partial and
  anonymous returns are not netted.
- **No leakage:** every model sees only past data. A reporting delay holds back
  the most recent day so that cancellation labels can mature. Random Forest
  forecasts recursively and never sees future actuals.
- **Holt-Winters:** additive, weekly seasonality, implemented by hand and
  numerically checked against statsmodels. This validates the code, not its
  forecasting skill.
- **Safety stock:** the 90th percentile of cumulative validation under-forecasts.
  With four calibration origins this is exploratory, not a service guarantee.
- **Reproducibility:** fixed random seed, single-threaded training, and a
  `manifest.json` recording the input hash, configuration, package versions and
  source-file hashes.

## Limitations

- Sales stand in for demand. Stockouts and lost demand are not observed.
- About one year of data and nine products; no seasonality across years.
- Sixteen non-closed dates have no records (Christmas break and several UK bank
  holidays). They are treated as zero sales unless you change the policy. The
  sensitivity analysis suggests the ranking does not change.
- Costs and starting stock are illustrative, not from a real business.
- No real savings or deployment results are claimed.

## Testing

`python run_tests.py` runs 28 tests covering cleaning, leakage, inventory
accounting, Holt-Winters agreement with statsmodels, Excel formulas and the
verifier. `verify_outputs.py` reconciles a finished run (row counts, daily stock
balances, costs, MAE recomputed from saved predictions and every workbook table).
Excel formulas were checked with an independent evaluator and in LibreOffice; the
workbook has not been tested in Microsoft Excel.

## References

- Hyndman and Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html) (rolling-origin evaluation)
- [statsmodels ETSModel](https://www.statsmodels.org/stable/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html)
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
