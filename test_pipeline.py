import unittest

import numpy as np
import pandas as pd

from cleaning import clean_transactions, match_cancellations
from forecasting import metrics, predict, available_predict, hw, feature_row
from inventory import simulate
from app import options


# Synthetic transaction helper
def row(inv, time, q=10, price=2, cust=1):
    return dict(
        InvoiceNo=inv,
        StockCode="A",
        Description="Item",
        Quantity=q,
        InvoiceDate=pd.Timestamp("2011-01-10 " + time),
        UnitPrice=price,
        CustomerID=cust,
        Country="UK",
    )



# Cleaning and cancellation regression tests
class CleaningTests(unittest.TestCase):
    def test_two_orders_two_cancellations(self):
        d = pd.DataFrame(
            [
                row("1", "09:00"),
                row("2", "10:00"),
                row("C1", "10:05", -10),
                row("C2", "10:10", -10),
                row("3", "15:00", 5),
            ]
        )
        a, demand, p, report = clean_transactions(d)
        self.assertEqual(len(p), 2)
        self.assertEqual(demand.Quantity.sum(), 5)
        self.assertEqual(p.OrderID.nunique(), 2)
        self.assertEqual(p.CancelID.nunique(), 2)

    def test_max_cardinality_beats_nearest_greedy(self):
        # A can match X only; B can match either; nearest B-X must not block A-X, B-Y
        rows = [
            row("A", "00:00"),
            row("B", "09:00"),
            row("CX", "09:01", -10),
            row("CY", "15:00", -10),
            row("extra", "16:00", 5),
        ]
        a, d, p, _ = clean_transactions(pd.DataFrame(rows), 10)
        self.assertEqual(len(p), 2)
        self.assertEqual(
            set(map(tuple, p[["OrderID", "CancelID"]].to_numpy())), {(0, 2), (1, 3)}
        )

    def test_price_partial_anonymous_not_matched(self):
        rows = [
            row("A", "09:00"),
            row("C1", "09:01", -10, 3),
            row("C2", "09:02", -5),
            row("C3", "09:03", -10, cust=None),
        ]
        a, d, p, _ = clean_transactions(pd.DataFrame(rows))
        self.assertTrue(p.empty)
        self.assertEqual(len(d), 1)

    def test_invalid_duplicate_partition(self):
        rows = [
            row("A", "09:00"),
            row("A", "09:00"),
            row("B", "09:01", 0),
            row("C", "09:02", np.nan),
            row("D", "09:03", price=np.nan),
            row("CE", "09:04", -1, -2),
        ]
        a, d, p, r = clean_transactions(pd.DataFrame(rows))
        self.assertEqual(len(d), 1)
        self.assertEqual(sum(x["Rows"] for x in r["category_counts"]), len(rows))
        self.assertEqual((a.Category == "Duplicate").sum(), 1)
        self.assertEqual((a.Category == "Invalid").sum(), 3)
        self.assertEqual((a.Category == "Adjustments").sum(), 1)

    def test_empty(self):
        with self.assertRaises(ValueError):
            clean_transactions(pd.DataFrame())

    def test_custom_index_and_zero_price_review(self):
        d = pd.DataFrame(
            [
                row("A", "09:00"),
                row("C1", "10:00", -10),
                row("B", "11:00", 5),
                row("D", "12:00", 3, 0),
            ]
        )
        d.index = ["x", "y", "z", "z"]
        d.loc[d.InvoiceNo == "D", "Description"] = None
        a, demand, p, _ = clean_transactions(d)
        self.assertEqual(len(p), 1)
        self.assertEqual(len(demand), 1)
        self.assertEqual((a.Category == "Review").sum(), 1)



# Forecasting, calendar and availability regression tests
class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.s = pd.Series(
            np.tile([10.0, 20, 30, 40, 50, 0, 60], 30),
            index=pd.date_range("2010-01-04", periods=210),
        )

    def test_zero_safe_metrics(self):
        m = metrics([0, 10], [2, 9])
        self.assertEqual(m["MAPE_nonzero_pct"], 10)
        self.assertEqual(m["MAPE_coverage_pct"], 50)
        self.assertIsNone(metrics([0, 0], [1, 2])["WAPE_pct"])
        self.assertIsNone(metrics([0, 0], [1, 2])["MAPE_nonzero_pct"])

    def test_seasonal_prediction_and_hw(self):
        p = predict(self.s, "Seasonal naive", 14, closed_weekdays=())
        np.testing.assert_allclose(p, np.tile(self.s.values[-7:], 2))
        np.testing.assert_allclose(hw(np.ones(70) * 10, 14), 10)
        np.testing.assert_allclose(hw(self.s.values, 7), self.s.values[-7:], atol=1e-7)

    def test_reporting_delay(self):
        a = available_predict(
            self.s,
            "Moving average",
            7,
            {"window": 7},
            closed_weekdays=(),
            reporting_delay=1,
        )
        altered = self.s.copy()
        altered.iloc[-1] = 1e9
        b = available_predict(
            altered,
            "Moving average",
            7,
            {"window": 7},
            closed_weekdays=(),
            reporting_delay=1,
        )
        pd.testing.assert_series_equal(a, b)
        self.assertEqual(a.index[0], self.s.index[-1] + pd.Timedelta(days=1))

    def test_rf_recursive_future_independence(self):
        p = available_predict(self.s, "Random Forest", 14, reporting_delay=1)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p >= 0).all())
        self.assertTrue((p[p.index.dayofweek == 5] == 0).all())
        # Exactly the same past must produce the same forecast regardless of external future values
        bigger = pd.concat([self.s, pd.Series([1e8] * 14, index=p.index)])
        q = available_predict(
            bigger.iloc[: len(self.s)], "Random Forest", 14, reporting_delay=1
        )
        pd.testing.assert_series_equal(p, q)

    def test_irregular_calendar_rejected(self):
        with self.assertRaises(ValueError):
            predict(self.s.iloc[:-2].drop(self.s.index[3]), "Moving average", 7)

    def test_moving_average_does_not_count_closure_twice(self):
        s = self.s.copy()
        s[:] = 10.0
        s[s.index.dayofweek == 5] = 0.0
        p = predict(s, "Moving average", 7, {"window": 7})
        self.assertAlmostEqual(p.sum(), 60.0)

    def test_features_no_current_target(self):
        f = feature_row(np.arange(1.0, 31.0), pd.Timestamp("2011-02-01"))
        self.assertEqual(f[:3], [30, 24, 3])
        self.assertEqual(f[3], 27)



# Inventory accounting and interface input tests
class InventoryTests(unittest.TestCase):
    def test_arrival_timing_and_accounting(self):
        actual = pd.Series([5.0] * 5, index=pd.date_range("2011-01-01", periods=5))
        forecasts = {i: np.ones(3) * 5 for i in range(5)}
        rows, s = simulate(actual, forecasts, 0, 0, 2, 1, 2.0)
        self.assertEqual(rows[0]["Ordered"], 15)
        self.assertEqual(rows[0]["Arrivals"], 0)
        self.assertEqual(rows[1]["Arrivals"], 0)
        self.assertEqual(rows[2]["Arrivals"], 15)
        self.assertEqual(
            rows[1]["Ordered"], 0
        )  # on-order stock prevents duplicate replenishment
        for r in rows:
            self.assertAlmostEqual(
                r["OpeningStock"] + r["Arrivals"] - r["Fulfilled"], r["ClosingStock"]
            )
        self.assertAlmostEqual(
            s["TotalCost"], s["HoldingCost"] + s["ShortageCost"] + s["OrderingCost"]
        )

    def test_zero_demand(self):
        rows, s = simulate(
            pd.Series([0.0] * 7, index=pd.date_range("2011-01-01", periods=7)),
            {0: np.zeros(14)},
            0,
            0,
            7,
            7,
            1,
        )
        self.assertEqual(s["FillRate"], 1)
        self.assertEqual(s["TotalCost"], 0)

    def test_ui_validation(self):
        c = options("7", "7", "14", "85123a, 22197", '{"85123A": 10}')
        self.assertEqual(c["products"], ["85123A", "22197"])
        with self.assertRaises(ValueError):
            options("0", "7", "14", "", "{}")
        with self.assertRaises(ValueError):
            options("7", "7", "14", "", '{"A": -1}')


if __name__ == "__main__":
    unittest.main(verbosity=2)
