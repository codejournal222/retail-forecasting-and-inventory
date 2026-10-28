import math

import numpy as np


# Simulate periodic-review replenishment and lost sales
def simulate(
    actual,
    forecasts,
    initial_stock,
    safety,
    lead,
    review,
    unit_price,
    holding_rate=0.2,
    shortage_fraction=0.5,
    order_cost=5.0,
):
    # At start of day: receive, review/order, then satisfy demand. L-day lead time

    # Protection period is L+R calendar days. Inventory position includes pipeline
    # No backorders, no returns restocking, no minimum order, infinite supplier capacity
    # Same starting stock and costs must be supplied to every competing model.
    if lead < 1 or review < 1:
        raise ValueError("Lead time and review interval must be positive.")
    if (
        min(
            initial_stock,
            safety,
            unit_price,
            holding_rate,
            shortage_fraction,
            order_cost,
        )
        < 0
    ):
        raise ValueError("Inventory inputs must be nonnegative.")

    # Initialise on-hand stock, outstanding orders and the daily ledger
    stock = float(initial_stock)
    pending = []
    rows = []
    for t, (date, demand) in enumerate(actual.items()):

        # 1. Receive deliveries due at the start of this day
        arrivals = sum(q for due, q in pending if due == t)
        pending = [(due, q) for due, q in pending if due > t]
        opening = stock
        stock += arrivals

        # 2. Review inventory position, including stock already in transit
        position = stock + sum(q for _, q in pending)
        ordered = 0
        target = 0.0
        if t % review == 0:
            if t not in forecasts:
                raise ValueError(f"Missing forecast at review day {t}.")
            target = max(0, float(np.sum(forecasts[t][: lead + review])) + safety)
            ordered = max(0, math.ceil(target - position))
            if ordered:
                pending.append((t + lead, ordered))

        # 3. Fulfil today's demand; unmet units are lost rather than backordered
        fulfilled = min(stock, float(demand))
        lost = float(demand) - fulfilled
        stock -= fulfilled

        # 4. Charge the assumed holding, shortage and ordering costs
        hc = stock * unit_price * holding_rate / 365
        sc = lost * unit_price * shortage_fraction
        oc = order_cost if ordered else 0.0
        rows.append(
            {
                "Date": date,
                "OpeningStock": opening,
                "Arrivals": arrivals,
                "InventoryPosition": position,
                "OrderUpTo": target,
                "Ordered": ordered,
                "Demand": float(demand),
                "Fulfilled": fulfilled,
                "LostUnits": lost,
                "ClosingStock": stock,
                "OnOrder": sum(q for _, q in pending),
                "HoldingCost": hc,
                "ShortageCost": sc,
                "OrderingCost": oc,
                "TotalCost": hc + sc + oc,
            }
        )
        assert abs(opening + arrivals - fulfilled - stock) < 1e-7
        assert abs(fulfilled + lost - demand) < 1e-7

    # Summarise the full simulation window
    total = sum(x["Demand"] for x in rows)
    lost = sum(x["LostUnits"] for x in rows)
    summary = {
        "Demand": total,
        "LostUnits": lost,
        "FillRate": 1 - lost / total if total else 1.0,
        "StockoutDays": sum(x["LostUnits"] > 0 for x in rows),
        "Orders": sum(x["Ordered"] > 0 for x in rows),
        "AverageStock": float(np.mean([x["ClosingStock"] for x in rows])),
        "FinalStock": stock,
        "FinalOnOrder": sum(q for _, q in pending),
        "HoldingCost": sum(x["HoldingCost"] for x in rows),
        "ShortageCost": sum(x["ShortageCost"] for x in rows),
        "OrderingCost": sum(x["OrderingCost"] for x in rows),
        "TotalCost": sum(x["TotalCost"] for x in rows),
    }
    return rows, summary
