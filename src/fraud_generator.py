"""Create fraud labels only when the corresponding synthetic behavior exists."""

import random

import pandas as pd

from src.loyalty_rules import points_for_purchase, recalculate_balances

FRAUD_TYPES = ["Large redemption", "Rapid redemption", "Abnormal accumulation", "Repeated suspicious activity"]
LARGE_REDEMPTION_POINTS = 500
RAPID_INTERVAL_SECONDS = 1


def inject_fraud(df, rate=0.03, seed=7):
    """Inject suspicious customer events while preserving chronological balances.

    Each selected customer history starts with a deliberately abnormal points
    accumulation. The following rows, when available, are close-together
    redemptions. With only one available row, only accumulation is labeled.
    """
    if not 0 <= float(rate) <= 0.5:
        raise ValueError("Fraud percentage must be between 0 and 50")
    if df.empty:
        result = df.copy()
        result["fraud_flag"] = pd.Series(dtype="int64")
        result["fraud_type"] = pd.Series(dtype="object")
        return result
    result = df.copy()
    if "earning_ratio" not in result:
        result["earning_ratio"] = 1.0
    result["fraud_flag"] = 0
    result["fraud_type"] = "Normal"
    result = recalculate_balances(result)
    fraud_count = round(len(result) * float(rate))
    if not fraud_count:
        return result

    rng = random.Random(seed)
    histories = {customer: list(group.index) for customer, group in result.groupby("customer_id", sort=False)}
    customers = sorted(histories, key=lambda customer: (-len(histories[customer]), str(customer)))
    rng.shuffle(customers)
    customers.sort(key=lambda customer: -len(histories[customer]))
    remaining = fraud_count
    for customer in customers:
        if remaining <= 0:
            break
        # One customer history creates a coherent event cluster. Keeping event
        # spacing at one second makes repeated redemptions unambiguously rapid.
        rows = histories[customer][:min(remaining, 4)]
        if not rows:
            continue
        anchor_time = pd.Timestamp(result.at[rows[0], "transaction_date"])
        customer_segment = result.at[rows[0], "customer_segment"]
        category_counts = result.loc[result.customer_id == customer, "product_category"].value_counts()
        preferred_category = category_counts.index[0] if not category_counts.empty else result.at[rows[0], "product_category"]
        group_size = len(rows)

        for offset, index in enumerate(rows):
            result.at[index, "fraud_flag"] = 1
            result.at[index, "customer_segment"] = customer_segment
            result.at[index, "product_category"] = preferred_category
            result.at[index, "transaction_date"] = anchor_time + pd.Timedelta(offset * RAPID_INTERVAL_SECONDS, unit="s")
            if offset == 0:
                result.at[index, "transaction_type"] = "purchase"
                result.at[index, "purchase_amount"] = max(1.0, float(result.at[index, "purchase_amount"]))
                base_points = points_for_purchase(result.at[index, "purchase_amount"], customer_segment, result.at[index, "earning_ratio"])
                result.at[index, "points_earned"] = max(1000, base_points + LARGE_REDEMPTION_POINTS * max(1, group_size - 1))
                result.at[index, "points_redeemed"] = 0
                result.at[index, "fraud_type"] = "Abnormal accumulation"
            else:
                result.at[index, "transaction_type"] = "redeem"
                result.at[index, "purchase_amount"] = 0.0
                result.at[index, "points_earned"] = 0
                result.at[index, "points_redeemed"] = LARGE_REDEMPTION_POINTS
                if offset == 1:
                    result.at[index, "fraud_type"] = "Large redemption"
                elif offset == 2:
                    result.at[index, "fraud_type"] = "Rapid redemption"
                else:
                    result.at[index, "fraud_type"] = "Repeated suspicious activity"
        remaining -= group_size

    # Rarely, a malformed input may request more flagged rows than existing
    # customer histories cover; leave the unused share normal rather than
    # writing a label with no behavior behind it.
    result = recalculate_balances(result)
    return result
