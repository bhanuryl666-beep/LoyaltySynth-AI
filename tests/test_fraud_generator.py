import unittest

import pandas as pd

from src.data_generator import generate_rule_based
from src.data_validator import validate_transactions
from src.fraud_generator import inject_fraud


class FraudSimulationTests(unittest.TestCase):
    def test_labels_match_real_injected_behaviors(self):
        clean = generate_rule_based(360, 12, "2025-01-01", "2025-12-31", seed=5)
        suspicious = inject_fraud(clean, 0.10, seed=12)
        flagged = suspicious[suspicious.fraud_flag == 1]
        self.assertEqual(len(flagged), round(len(clean) * 0.10))
        self.assertIn("Abnormal accumulation", set(flagged.fraud_type))
        self.assertIn("Large redemption", set(flagged.fraud_type))
        self.assertIn("Rapid redemption", set(flagged.fraud_type))
        self.assertIn("Repeated suspicious activity", set(flagged.fraud_type))
        for customer, events in flagged.groupby("customer_id"):
            if (events.fraud_type == "Rapid redemption").any():
                redemptions = suspicious[(suspicious.customer_id == customer) & (suspicious.fraud_flag == 1) & (suspicious.transaction_type == "redeem")]
                times = pd.to_datetime(redemptions.transaction_date).sort_values()
                self.assertTrue((times.diff().dropna() <= pd.Timedelta(5, unit="min")).any())
        self.assertTrue(all(check["status"] == "Pass" for check in validate_transactions(suspicious)))

    def test_zero_rate_does_not_create_labels(self):
        data = inject_fraud(generate_rule_based(30, 5, seed=3), 0)
        self.assertEqual(int(data.fraud_flag.sum()), 0)
        self.assertTrue((data.fraud_type == "Normal").all())


if __name__ == "__main__":
    unittest.main()
