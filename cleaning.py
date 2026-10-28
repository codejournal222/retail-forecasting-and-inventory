
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Required fields and explicit cleaning rules
COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
NON_PRODUCT = {
    "POST",
    "DOT",
    "M",
    "D",
    "S",
    "AMAZONFEE",
    "CRUK",
    "PADS",
    "B",
    "DCGSSGIRL",
    "DCGSSBOY",
}



# Objective 1: Import records and preserve their provenance
def load_source(path):
    path = Path(path)
    if path.name.lower().endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path, dtype={"InvoiceNo": "string", "StockCode": "string"})
        df["SourceSheet"], df["SourceRow"] = "CSV", np.arange(2, len(df) + 2)
        recovered = False
    else:
        book = pd.ExcelFile(path)
        recovered = "Demand_Cleaned" in book.sheet_names
        names = (
            [
                n
                for n in [
                    "Demand_Cleaned",
                    "Returns",
                    "Adjustments",
                    "Non_Product_Charges",
                    "Cancelled_Orders",
                ]
                if n in book.sheet_names
            ]
            if recovered
            else [book.sheet_names[0]]
        )
        frames = []
        for name in names:
            d = pd.read_excel(
                book,
                sheet_name=name,
                dtype={"InvoiceNo": "string", "StockCode": "string"},
            )
            missing = set(COLUMNS) - set(d.columns)
            if missing:
                raise ValueError(f"{name}: missing required columns {sorted(missing)}")
            d = d[COLUMNS].copy()
            d["SourceSheet"], d["SourceRow"] = name, np.arange(2, len(d) + 2)
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)

    # Validate the combined input before returning it
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The input contains no transaction rows.")
    return df, {
        "input_name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reconstructed_from_cleaned_workbook": recovered,
        "provenance_limit": (
            "Previously discarded rows cannot be recovered."
            if recovered
            else "Raw input supplied."
        ),
    }



# Objective 2: Normalise identifiers
def canonical_id(s):
    return s.astype("string").str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)



# And match full cancellations one-to-one
def match_cancellations(df, hours=24):
    """Maximum-cardinality, then minimum-total-delay matching within exact key groups.

    Inferred withdrawals only: equal customer, product, quantity AND price.
    Partial or anonymous cancellations remain unmatched, never silently netted.
    Integer row IDs are independent of caller index labels.
    """
    if hours <= 0:
        return pd.DataFrame(columns=["OrderID", "CancelID", "GapHours"])

    # Only identified customers can support a full reversal match
    orders = df[(df.Category == "Demand") & df.CustomerID.notna()]
    cancels = df[
        (df.Category == "Returns")
        & df.CustomerID.notna()
        & (df.Quantity < 0)
        & (df.UnitPrice >= 0)
    ].copy()
    cancels["Quantity"] = -cancels.Quantity
    keys = ["CustomerID", "StockCode", "Quantity", "UnitPrice"]
    og = orders.groupby(keys, sort=False)
    order_keys = set(og.indices)
    matches = []
    for key, c in cancels.groupby(keys, sort=False):
        if key not in order_keys:
            continue
        o = og.get_group(key)
        # Split by temporal connected components to bound dense assignment memory
        events = pd.concat([o.assign(_kind="o"), c.assign(_kind="c")]).sort_values(
            ["InvoiceDate", "RowID"]
        )
        cluster = (events.InvoiceDate.diff() > pd.Timedelta(hours=hours)).cumsum()
        for _, part in events.groupby(cluster):
            oo = part[part._kind == "o"]
            cc = part[part._kind == "c"]
            if oo.empty or cc.empty:
                continue
            if len(oo) * (len(cc) + len(oo)) > 10_000_000:
                raise ValueError(
                    "Cancellation group too large for exact matching; narrow window or review this group."
                )
            gap = (
                cc.InvoiceDate.to_numpy()[None, :] - oo.InvoiceDate.to_numpy()[:, None]
            ) / np.timedelta64(1, "h")
            valid = (gap >= 0) & (gap <= hours)
            penalty = min(len(oo), len(cc)) + 1
            costs = np.full((len(oo), len(cc) + len(oo)), float(penalty))
            costs[:, : len(cc)] = np.where(valid, gap / hours, penalty * 1000)
            rr, cols = linear_sum_assignment(costs)
            for i, j in zip(rr, cols):
                if j < len(cc) and valid[i, j]:
                    matches.append(
                        (int(oo.iloc[i].RowID), int(cc.iloc[j].RowID), float(gap[i, j]))
                    )
    return pd.DataFrame(matches, columns=["OrderID", "CancelID", "GapHours"])



# And classify transactions and build the cleaning audit
def clean_transactions(source, cancellation_hours=24):
    if source.empty:
        raise ValueError("The input contains no transaction rows.")
    if set(COLUMNS) - set(source.columns):
        raise ValueError("Required transaction columns are missing.")

    # Keep stable internal row IDs, independent of the source index
    d = source.copy().reset_index(drop=True)
    d["RowID"] = np.arange(len(d))
    raw_missing = d[COLUMNS].isna().sum().to_dict()
    # Preserve original values separately; the audit includes stable source coordinates
    duplicates = d.duplicated(COLUMNS, keep="first")
    for col in ["InvoiceNo", "StockCode", "CustomerID"]:
        d[col] = canonical_id(d[col])
        d[col] = d[col].replace("", pd.NA)
    d["InvoiceNo"] = d.InvoiceNo.str.upper()
    d["StockCode"] = d.StockCode.str.upper()
    for col in ["Quantity", "UnitPrice"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["InvoiceDate"] = pd.to_datetime(d.InvoiceDate, errors="coerce")

    # Identify invalid values before applying the category precedence
    bad = (
        d.InvoiceDate.isna()
        | d.InvoiceNo.isna()
        | d.StockCode.isna()
        | ~np.isfinite(d.Quantity)
        | ~np.isfinite(d.UnitPrice)
        | (d.Quantity == 0)
        | ((d.Quantity.fillna(0) % 1) != 0)
    )
    cancellation = d.InvoiceNo.str.startswith("C", na=False)

    # Assign mutually exclusive categories; later rules take precedence
    d["Category"] = "Demand"
    d.loc[d.StockCode.isin(NON_PRODUCT), "Category"] = "Non_Product"
    d.loc[(d.Quantity < 0) & ~cancellation, "Category"] = "Adjustments"
    d.loc[d.UnitPrice < 0, "Category"] = "Adjustments"
    d.loc[cancellation & (d.UnitPrice >= 0), "Category"] = "Returns"
    bad |= cancellation & (d.Quantity >= 0)
    # Missing descriptions on zero-price positives are not evidence of fulfilled sales
    d.loc[
        (d.Quantity > 0)
        & (d.UnitPrice == 0)
        & d.Description.isna()
        & (d.Category == "Demand"),
        "Category",
    ] = "Review"
    d.loc[bad, "Category"] = "Invalid"
    d.loc[duplicates, "Category"] = "Duplicate"
    d["HasCustomerID"] = d.CustomerID.notna()
    d["ZeroPrice"] = d.UnitPrice.eq(0)
    d["MissingDescription"] = d.Description.isna()
    d["Date"] = d.InvoiceDate.dt.normalize()

    # Link both sides of each inferred cancellation for auditability
    pairs = match_cancellations(d, cancellation_hours)
    d["MatchedRowID"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    if not pairs.empty:
        for col, other in [("OrderID", "CancelID"), ("CancelID", "OrderID")]:
            d.loc[pairs[col], "MatchedRowID"] = pairs[other].to_numpy()
        d.loc[pairs.OrderID, "Category"] = "Inferred_Cancelled"

    # Extract the modelling target and reconcile every source row
    demand = d[d.Category == "Demand"].copy()
    if demand.empty:
        raise ValueError("No valid demand remains after cleaning.")
    assert demand.Quantity.gt(0).all() and demand.UnitPrice.ge(0).all()
    counts = d.Category.value_counts().rename_axis("Category").reset_index(name="Rows")
    assert counts.Rows.sum() == len(source)
    audit = {
        "source_rows": len(source),
        "raw_missing": raw_missing,
        "category_counts": counts.to_dict("records"),
        "matched_pairs": len(pairs),
        "matched_units": float(
            d.loc[d.Category == "Inferred_Cancelled", "Quantity"].sum()
        ),
        "remaining_unmatched_returns": int(
            ((d.Category == "Returns") & d.MatchedRowID.isna()).sum()
        ),
        "clean_demand_rows": len(demand),
        "demand_units": float(demand.Quantity.sum()),
        "cancellation_window_hours": cancellation_hours,
        "assumptions": [
            "Exact duplicates excluded but preserved; may contain genuine repeated lines.",
            "Same-price full reversals within the time window are inferred withdrawals, not proven non-fulfilment.",
            "Partial returns and unmatched returns retained separately. Sales are a proxy for demand; stockouts are unobserved.",
            "Zero-price positive rows with missing description quarantined for review.",
        ],
    }
    return d, demand, pairs, audit
