import unittest

import pandas as pd

from src.data_generator import generate_rule_based
from src.data_validator import validate_transactions
from src.loyalty_rules import apply_points_balance, points_for_purchase, recalculate_balances


class LoyaltyRulesTests(unittest.TestCase):
    def test_segment_earning_ratios(self):
        self.assertEqual(points_for_purchase(100, "Standard", 1.0), 100)
        self.assertEqual(points_for_purchase(100, "Premium", 1.0), 150)
        self.assertEqual(points_for_purchase(100, "VIP", 1.0), 200)
        self.assertEqual(points_for_purchase(100, "VIP", 1.5), 300)

    def test_redemption_cannot_exceed_available_balance(self):
        self.assertEqual(apply_points_balance(400, earned=0, redeemed=300), 100)
        with self.assertRaises(ValueError):
            apply_points_balance(100, redeemed=101)

    def test_balance_recalculation_sorts_by_date_and_caps_redemptions(self):
        transactions = pd.DataFrame([
            {"transaction_id": "LS-2", "customer_id": "C1", "transaction_date": "2025-01-03", "transaction_type": "redeem", "purchase_amount": 4, "points_earned": 10, "points_redeemed": 300, "points_balance": 0},
            {"transaction_id": "LS-1", "customer_id": "C1", "transaction_date": "2025-01-01", "transaction_type": "purchase", "purchase_amount": 100, "points_earned": 100, "points_redeemed": 0, "points_balance": 100},
        ])
        actual = recalculate_balances(transactions)
        self.assertEqual(actual.transaction_id.tolist(), ["LS-1", "LS-2"])
        self.assertEqual(actual.points_balance.tolist(), [100, 0])
        self.assertEqual(actual.loc[1, "points_redeemed"], 100)
        self.assertEqual(actual.loc[1, "purchase_amount"], 0)

    def test_generated_points_and_end_date_are_inclusive(self):
        data = generate_rule_based(120, 12, "2025-04-17", "2025-04-17", 1.25, seed=11)
        self.assertEqual(len(data), 120)
        self.assertTrue((pd.to_datetime(data.transaction_date).dt.date == pd.Timestamp("2025-04-17").date()).all())
        self.assertTrue(all(check["status"] == "Pass" for check in validate_transactions(data)))
        purchases = data[data.transaction_type == "purchase"]
        expected = [points_for_purchase(r.purchase_amount, r.customer_segment, r.earning_ratio) for r in purchases.itertuples()]
        self.assertEqual(purchases.points_earned.tolist(), expected)

    def test_customer_preferences_and_frequency_vary_consistently(self):
        data = generate_rule_based(240, 12, seed=4)
        customer_counts = data.groupby("customer_id").size()
        self.assertGreater(customer_counts.nunique(), 1)
        preferred_share = data.groupby("customer_id").product_category.apply(lambda values: values.value_counts(normalize=True).iloc[0])
        self.assertGreater(float(preferred_share.median()), 0.5)
        self.assertEqual(data.groupby("customer_id").customer_segment.nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
