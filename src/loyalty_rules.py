"""Small business rules used by both seed and rule-based generation."""

from datetime import datetime

import pandas as pd

EARNING_RATIOS = {"Standard": 1.0, "Premium": 1.5, "VIP": 2.0}


def points_for_purchase(amount, segment="Standard", base_ratio=1.0):
    """Award whole points; customer segment boosts the configured base ratio."""
    return max(0, int(float(amount) * float(base_ratio) * EARNING_RATIOS.get(segment, 1.0)))


def apply_points_balance(balance, earned=0, redeemed=0):
    """Return the new balance, refusing redemptions above the available balance."""
    balance, earned, redeemed = int(balance), int(earned), int(redeemed)
    if min(balance, earned, redeemed) < 0 or redeemed > balance + earned:
        raise ValueError("Points cannot be negative or redeemed beyond the available balance")
    return balance + earned - redeemed


def recalculate_balances(df):
    """Sort transactions chronologically and rebuild their customer balances.

    Purchase rows earn points and cannot redeem in the same transaction.
    Redemption rows carry no purchase or earned points and are capped at the
    customer's available balance.
    """
    result = df.copy()
    result["transaction_date"] = pd.to_datetime(result["transaction_date"], errors="raise")
    sort_columns = ["transaction_date", "customer_id"]
    if "transaction_id" in result.columns:
        sort_columns.append("transaction_id")
    result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    balances = {}
    for index, row in result.iterrows():
        customer = row["customer_id"]
        available = balances.get(customer, 0)
        if row["transaction_type"] == "purchase":
            amount = max(0.0, float(row["purchase_amount"]))
            earned = max(0, int(row["points_earned"]))
            redeemed = 0
        elif row["transaction_type"] == "redeem":
            amount, earned = 0.0, 0
            desired = max(0, int(row["points_redeemed"]))
            redeemed = min(available, desired)
        else:
            raise ValueError(f"Unknown transaction type: {row['transaction_type']}")
        balance = apply_points_balance(available, earned, redeemed)
        result.at[index, "purchase_amount"] = amount
        result.at[index, "points_earned"] = earned
        result.at[index, "points_redeemed"] = redeemed
        result.at[index, "points_balance"] = balance
        balances[customer] = balance
    return result


def transaction_id(sequence):
    return f"LS-{int(sequence):08d}"


def parse_date(value):
    """Parse an ISO date string into a date, raising a clear validation error."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must use YYYY-MM-DD format") from exc
