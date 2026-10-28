import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Historical analysis, comparison charts and offline report
def make_report(demand, tables, audit, manifest, out):
    out = Path(out)
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "figure.dpi": 130,
        }
    )
    files = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(charts / name)
        plt.close(fig)
        files.append(name)

    # 1. Daily retained units and their rolling average
    daily = (
        demand.groupby("Date")
        .Quantity.sum()
        .reindex(pd.date_range(demand.Date.min(), demand.Date.max()), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(daily.index, daily.values, lw=0.9, color="#344b75")
    ax.plot(
        daily.rolling(28).mean(), lw=2, label="28-calendar-day mean", color="#df8541"
    )
    ax.set(title="Corrected aggregate sales", ylabel="Units")
    ax.legend()
    finish(fig, "01_daily.png")

    # 2. Monthly retained sales value
    rev = (
        demand.assign(Revenue=demand.Quantity * demand.UnitPrice)
        .groupby("Date")
        .Revenue.sum()
    )
    monthly = rev.resample("MS").sum()
    labels = [x.strftime("%b %Y") for x in monthly.index]
    if demand.Date.max() != demand.Date.max() + pd.offsets.MonthEnd(0):
        labels[-1] += "*"
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(labels, monthly, color="#344b75")
    ax.tick_params(axis="x", rotation=40)
    ax.set(
        title="Retained sales revenue · final month is partial (*)",
        ylabel="GBP, excludes returns",
    )
    finish(fig, "02_monthly.png")

    # 3. Average sales by calendar weekday
    dow = daily.groupby(daily.index.dayofweek).mean().reindex(range(7), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow, color="#344b75")
    ax.set(title="Average units per calendar weekday", ylabel="Units per day")
    finish(fig, "03_weekday.png")

    # 4. Largest products by retained units
    top = demand.groupby("StockCode").Quantity.sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index, top.values, color="#344b75")
    ax.set(title="Top products after cleaning", xlabel="Units")
    finish(fig, "04_products.png")

    # 5. Non-UK country totals
    countries = (
        demand[demand.Country != "United Kingdom"]
        .assign(Revenue=lambda x: x.Quantity * x.UnitPrice)
        .groupby("Country")
        .Revenue.sum()
        .nlargest(10)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(countries.index.tolist(), countries.to_numpy(), color="#344b75")
    ax.set(title="Top non-UK markets · retained sales revenue", xlabel="GBP")
    finish(fig, "05_countries.png")

    # 6. Actual and forecast units for one illustrative product/window
    preds = tables["Forecast_Detail"]
    code = tables["Products"].StockCode.iloc[0]
    p = preds[(preds.StockCode == code) & (preds.Fold == 3)]
    fig, ax = plt.subplots(figsize=(10, 4))
    actual = p.drop_duplicates("Date").sort_values("Date")
    ax.plot(actual.Date, actual.Actual, color="black", label="Actual", lw=2)
    for m, g in p.groupby("Model"):
        ax.plot(g.Date, g.Predicted, label=m, lw=1.1)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set(
        title=f"Product {code} · final test window · forecasts renewed at each review",
        ylabel="Units",
    )
    ax.legend(fontsize=8, ncol=3)
    finish(fig, "06_forecasts.png")

    # 7. Pooled held-out forecast errors
    comp = tables["Model_Comparison"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(comp.Model, comp.MAE, color="#344b75")
    ax.set(
        title=f"Held-out forecast error · {len(tables['Products'])} products",
        ylabel="MAE (units/product/day)",
    )
    ax.tick_params(axis="x", rotation=12)
    finish(fig, "07_accuracy.png")

    # 8. Simulated inventory cost and service outcomes
    inv = tables["Inventory_Comparison"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(inv.Model, inv.TotalCost, color="#344b75")
    axs[0].set(title="Simulated relevant cost", ylabel="GBP")
    axs[1].bar(inv.Model, inv.FillRate * 100, color="#5b8d80")
    axs[1].set(title="Simulated unit fill rate", ylabel="%")
    for ax in axs:
        ax.tick_params(axis="x", rotation=30)
    finish(fig, "08_inventory.png")

    # HTML helpers: escape table content and embed chart bytes
    def table(df):
        return df.to_html(
            index=False, border=0, float_format=lambda v: f"{v:,.2f}", escape=True
        )

    def picture(name):
        return (
            '<img alt="'
            + html.escape(name)
            + '" src="data:image/png;base64,'
            + base64.b64encode((charts / name).read_bytes()).decode()
            + '">'
        )

    best = comp.loc[comp.MAE.idxmin(), "Model"]
    costbest = inv.loc[inv.TotalCost.idxmin(), "Model"]
    config = manifest["config"]
    period = f"{str(manifest['analysis_start'])[:10]} to {str(manifest['analysis_end'])[:10]}"

    # Build the offline report; adjacent string fragments concatenate exactly
    text = (
        "<!doctype html>"
        '<html lang="en">'
        '<meta charset="utf-8">'
        "<title>Retail forecasting study</title>\n"
        "<style>body{font:16px/1.65 system-ui,sans-serif;"
        "color:#233044;"
        "background:#edf0f4;"
        "margin:0}main{max-width:1080px;"
        "margin:30px auto;"
        "background:white;"
        "padding:45px;"
        "border-radius:12px}h1{font-size:36px;"
        "line-height:1.15}h2{margin-top:42px;"
        "color:#344b75}.eyebrow{color:#667a91;"
        "letter-spacing:2px;"
        "font-size:12px}.note{background:#fff2d9;"
        "padding:18px;"
        "border-left:4px solid #d29c41}table{border-collapse:collapse;"
        "width:100%;"
        "font-size:13px}th,td{text-align:right;"
        "padding:9px;"
        "border-bottom:1px solid "
        "#dde3eb}th:first-child,td:first-child{text-align:left}th{background:#e9eef5}img{max-width:100%;"
        "height:auto}.scroll{overflow-x:auto}code{background:#edf0f4;"
        "padding:2px 5px}a{color:#305c99}@media "
        "print{body{background:white}main{margin:0;"
        "padding:0}h2{break-after:avoid}img,table{break-inside:avoid}}</style>"
        "\n"
        "<main>"
        "<h1>Forecast accuracy and inventory outcomes</h1>\n"
        "<p>"
        f"{period}"
        " · "
        f"{len(tables['Products'])}"
        " selected products · three held-out windows · "
        f"{config['lead_days']}"
        "-day lead time / "
        f"{config['review_days']}"
        "-day review</p>\n"
        "<p>"
        "<strong>"
        f"{html.escape(best)}"
        "</strong> has the lowest pooled test MAE. <strong>"
        f"{html.escape(costbest)}"
        "</strong> has the lowest simulated relevant cost under the stated "
        "assumptions. These are descriptive results for this sample, not "
        "proof of general superiority.</p>\n"
        "<h2>1. Cleaning and validation</h2>"
        "<p>"
        f"{audit['source_rows']:,}"
        " source rows reconcile to mutually exclusive categories. "
        f"{audit['matched_pairs']:,}"
        " inferred cancelled orders ("
        f"{audit['matched_units']:,.0f}"
        " units) are removed from demand. Matching maximises valid pair "
        "count, then minimises delay, requiring identical customer, product, "
        "quantity and price.</p>"
        f"{table(tables['Cleaning'])}"
        "\n"
        "<p>Unmatched and partial returns stay in the audit. Zero-price "
        "positives without descriptions are quarantined. Duplicate exclusions"
        " and all original source values are retained in compressed CSVs. "
        "Changing the cancellation window changes the definition of the "
        "target, not merely model performance.</p>"
        f"{table(tables['Cleaning_Sensitivity'])}"
        "\n"
        "<h2>2. Calendar and historical analysis</h2>"
        "<p>Full calendar-day series include product zero-sales dates. A day "
        "without source records is assumed zero recorded sales, not proven "
        "zero demand. "
        f"{manifest['calendar_unknown_zero_days']}"
        " non-closed dates require calendar review. Saturdays are an explicit"
        " default closure assumption;"
        " no future holiday calendar has been invented. The terminal date is "
        "excluded by default as potentially incomplete. Revenue means "
        "retained sales value, not accounting net revenue.</p>\n"
        f"{''.join((picture(f) for f in files[:5]))}"
        "\n"
        "<h2>3. Experimental design</h2>"
        "<p>Products are selected using only data before "
        f"{str(manifest['selection_cutoff'])[:10]}"
        ". High-volume, middle-volume and intermittent products provide "
        "different forecasting challenges. User-selected products override "
        "this sampling rule.</p>"
        '<div class="scroll">'
        f"{table(tables['Products'])}"
        "</div>\n"
        "<p>Four non-overlapping validation horizons choose parameters by "
        "MAE;"
        " model selection also uses validation only. The 90th percentile of "
        "signed cumulative validation errors (actual minus predicted), "
        "floored at zero, provides a safety buffer. With only four "
        "calibration origins this is an illustrative buffer, not a "
        "demonstrated 90% service guarantee. The same validation sample "
        "chooses parameters and estimates the buffer, so calibration is "
        "exploratory rather than independent.</p>\n"
        "<p>Test windows are separate from validation. Forecasts are "
        "generated at each review using only prior mature history, then "
        "recursively predict multiple days without access to intervening "
        "actuals. A "
        f"{manifest['reporting_delay_days']}"
        "-day reporting delay allows cancellation labels to mature. Earlier "
        "test observations may enter later training only once available. "
        "Product models use calendar lags 1, 7 and 28 and shifted rolling "
        "features;"
        " Random Forest month/weekday variables are known at forecast "
        "time.</p>"
        f"{table(tables['Test_Windows'])}"
        "\n"
        "<h2>4. Forecast results</h2>"
        f"{table(comp)}"
        "<p>Daily metrics cover horizons one through the review interval, "
        "with each test actual counted once per model. WAPE aggregates "
        "absolute errors against observed units. MAPE excludes zero actuals "
        "and reports coverage;"
        " it must not be interpreted as accuracy on every day. Pooled metrics"
        " weight larger-volume products more heavily. Product/fold results "
        "are exported separately.</p>"
        "<h3>Lead time plus review: cumulative forecast accuracy</h3>"
        f"{table(tables['Protection_Comparison'])}"
        "<p>These metrics compare cumulative units over the full protection "
        "period. Only forecasts whose entire horizon fits inside the relevant"
        " test window are scored. Adjacent origins overlap, so errors are not"
        " independent;"
        " no significance claim is made.</p>"
        f"{picture('06_forecasts.png')}"
        f"{picture('07_accuracy.png')}"
        "\n"
        "<h2>5. Inventory experiment</h2>"
        "<p>At the start of each day, receive scheduled stock, review "
        "inventory when due, place an order, then fulfil demand. An order "
        "arrives after "
        f"{config['lead_days']}"
        " calendar days. The order-up-to target covers lead time plus review "
        "interval, with a model-specific validation buffer. Inventory "
        "position includes on-order stock. All models start each test window "
        "with the same stock based on prior sales;"
        " there is no initial pipeline. Unfulfilled demand is lost, not "
        "backordered. Historical demand is assumed observable after each day,"
        " even when the simulated policy loses sales;"
        " real lost demand may be unobserved.</p>\n"
        "<p>Illustrative annual holding rate "
        f"{config['holding_rate']:.0%}"
        ";"
        " shortage penalty "
        f"{config['shortage_fraction']:.0%}"
        " of a training-only selling-price proxy per lost unit;"
        " fixed order cost £"
        f"{config['order_cost']:.2f}"
        ". Purchase cost, revenue/profit, terminal liquidation and actual "
        "supplier constraints are excluded. Final stock and outstanding "
        "orders are reported so the finite-window boundary remains visible. "
        "The experiment is a policy comparison, not a complete profit "
        "calculation.</p>"
        '<div class="scroll">'
        f"{table(inv)}"
        "</div>"
        f"{picture('08_inventory.png')}"
        "\n"
        "<p>Cost sensitivity scenarios are exported. They revalue the same "
        "fixed policies;"
        " they do not claim each policy has been re-optimised for the new "
        "cost assumptions.</p>\n"
        "<h2>6. Future forecasts and replenishment demonstration</h2>"
        "<p>Future means after this historical sample's analysis cutoff, not "
        "today's date. The selected model for each product comes from "
        "validation, is refitted using available history, and predicts future"
        " calendar dates. Current stock is illustrative unless explicitly "
        "supplied;"
        " on-order stock defaults to zero in the example workbook and can be "
        "edited there. Suggested order = max(0, ceil(protection-period "
        "forecast + buffer − stock − on-order)). The lead-time reorder point "
        "is also shown for reference;"
        " the simulator uses periodic order-up-to control.</p>"
        '<div class="scroll">'
        f"{table(tables['Replenishment'][['StockCode', 'SelectedModel', 'ForecastProtectionUnits', 'SafetyStock', 'CurrentStock', 'SuggestedOrder', 'StockSource']])}"
        "</div>\n"
        "<h2>7. What can and cannot be concluded</h2>"
        "<p>The findings are based on a limited sample of products "
        "and approximately one year of data. Recorded sales may understate demand if products were out of stock, "
        "but historical stock availability is unknown. Cancellation matching relies on assumptions about related transactions, "
        "and safety-stock estimates use only a few validation periods. Simpler forecasting methods performed well in this "
        "comparison, highlighting the importance of testing machine-learning models against basic alternatives. "
        "Assessing whether these findings apply more broadly would require additional products, several years of data, "
        "costs that reflect actual business operations, and a verified trading calendar.</p>\n"
        "</main>"
        "</html>"
    )
    reference = manifest["holt_winters_reference"]
    quality = f"""
    <h2>8. Independent model validation and unresolved source dates</h2>
    <p>The historical charts summarize
    all retained products; model and inventory comparisons concern only the
    selected product sample.</p>
    <p>Holt–Winters was compared against statsmodels ETS(A,A,A), with matching
    initial states and translated smoothing coefficients, without optimization.
    All {reference['comparisons_run']} real-data validation/grid comparisons passed.
    Maximum absolute forecast difference: {reference['max_absolute_difference']:.3g}
    units. This validates the numerical recurrence, not forecast superiority.</p>
    <p>The following dates have no source records and are not configured closures.
    Their cause is unconfirmed. They include dates outside the Christmas period.
    The baseline explicitly assumes zero recorded sales. Set unknown_day_policy
    to error to stop a run with unresolved gaps.</p>
    <div class="scroll">{table(tables['Calendar_Review'])}</div>
    <h3>Calendar sensitivity on common observed test dates</h3>
    <p>The alternative replaces flagged historical values with the mean of up to
    four earlier observed dates of the same weekday. Future donors and previously
    imputed donors are excluded. Products, origins and scoring dates are held
    fixed; parameters are selected again on eligible validation horizons.
    Neither scenario is scored against invented actuals on missing-source dates.
    These metrics therefore have a different denominator from the main table.
    No inventory benefits are asserted using imputed demand.</p>
    <div class="scroll">{table(tables['Calendar_Sensitivity'])}</div>
    <p>Reference: <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html">statsmodels ETSModel</a>.</p>
    """
    text = text.replace("VERSION 2", "VERSION 2.1").replace(
        "</main>", quality + "</main>"
    )
    (out / "report.html").write_text(text, encoding="utf-8")
