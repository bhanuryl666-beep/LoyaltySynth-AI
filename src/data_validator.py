"""Validate transaction schema, chronological loyalty rules, and fraud behavior."""

import pandas as pd

from src.fraud_generator import FRAUD_TYPES, LARGE_REDEMPTION_POINTS, RAPID_INTERVAL_SECONDS
from src.loyalty_rules import points_for_purchase

REQUIRED = ["transaction_id", "customer_id", "transaction_date", "transaction_type", "purchase_amount", "points_earned", "points_redeemed", "points_balance", "product_category", "payment_method", "customer_segment", "earning_ratio", "fraud_flag", "fraud_type"]


def validate_transactions(df):
    missing = [name for name in REQUIRED if name not in df.columns]
    checks = [{"name": "Required columns", "status": "Pass" if not missing else "Fail", "detail": "All expected fields present" if not missing else "Missing: " + ", ".join(missing)}]
    if missing:
        return checks
    required_frame = df[REQUIRED]
    missing_cells = int(required_frame.isna().sum().sum())
    checks.append({"name": "No missing values", "status": "Pass" if missing_cells == 0 else "Fail", "detail": f"{missing_cells} empty cells"})
    if missing_cells:
        return checks
    checks.append({"name": "Unique transaction IDs", "status": "Pass" if not df.transaction_id.duplicated().any() else "Fail", "detail": f"{int(df.transaction_id.duplicated().sum())} duplicate IDs"})
    try:
        df = df.copy()
        for column in ["purchase_amount", "points_earned", "points_redeemed", "points_balance", "earning_ratio", "fraud_flag"]:
            df[column] = pd.to_numeric(df[column], errors="raise")
    except (TypeError, ValueError):
        checks.append({"name": "Numeric transaction fields", "status": "Fail", "detail": "Amount, points, ratio, and fraud fields must be numeric"})
        return checks
    numeric = df[["purchase_amount", "points_earned", "points_redeemed", "points_balance", "earning_ratio"]]
    nonnegative = (numeric[["purchase_amount", "points_earned", "points_redeemed", "points_balance"]] >= 0).all().all() and (numeric.earning_ratio > 0).all()
    checks.append({"name": "Non-negative amounts and points", "status": "Pass" if nonnegative else "Fail", "detail": "Amounts and points are non-negative; earning ratios are positive"})
    checks.append({"name": "Valid transaction types", "status": "Pass" if df.transaction_type.isin(["purchase", "redeem"]).all() else "Fail", "detail": "Transaction type is purchase or redeem"})
    checks.append({"name": "Valid customer segments", "status": "Pass" if df.customer_segment.isin(["Standard", "Premium", "VIP"]).all() else "Fail", "detail": "Segments use Standard, Premium, or VIP"})

    try:
        chronological = df.assign(_timestamp=pd.to_datetime(df.transaction_date, errors="raise")).sort_values(["_timestamp", "customer_id", "transaction_id"], kind="stable")
        date_error = None
    except (TypeError, ValueError) as exc:
        chronological = df.copy()
        date_error = str(exc)
    checks.append({"name": "Valid transaction dates", "status": "Pass" if date_error is None else "Fail", "detail": "All transaction dates parse correctly" if date_error is None else date_error})

    balance_errors = 0
    earning_errors = 0
    redemption_errors = 0
    if date_error is None:
        running = {}
        for row in chronological.itertuples(index=False):
            previous = running.get(row.customer_id, 0)
            earned, redeemed = int(row.points_earned), int(row.points_redeemed)
            if row.transaction_type == "purchase":
                expected_earned = points_for_purchase(row.purchase_amount, row.customer_segment, row.earning_ratio)
                if row.fraud_type == "Abnormal accumulation" and int(row.fraud_flag) == 1:
                    earning_errors += int(earned <= expected_earned)
                else:
                    earning_errors += int(earned != expected_earned)
                redemption_errors += int(redeemed != 0)
            elif row.transaction_type == "redeem":
                earning_errors += int(earned != 0)
                redemption_errors += int(float(row.purchase_amount) != 0.0 or redeemed > previous)
            expected_balance = previous + earned - redeemed
            balance_errors += int(expected_balance < 0 or int(row.points_balance) != expected_balance)
            running[row.customer_id] = int(row.points_balance)
    checks.append({"name": "Points earned match purchase rules", "status": "Pass" if date_error is None and earning_errors == 0 else "Fail", "detail": f"{earning_errors} earning or redemption rows violate segment rules (abnormal accumulation labels must exceed them)"})
    checks.append({"name": "Redemptions obey transaction rules", "status": "Pass" if date_error is None and redemption_errors == 0 else "Fail", "detail": f"{redemption_errors} rows have a purchase amount on redemption, redeem on purchase, or exceed available points"})
    checks.append({"name": "Chronological balances reconcile", "status": "Pass" if date_error is None and balance_errors == 0 else "Fail", "detail": f"{balance_errors} balances differ from prior chronological balance + earned - redeemed"})

    valid_flags = df.fraud_flag.isin([0, 1]).all()
    valid_types = set(df.fraud_type.unique()).issubset(set(FRAUD_TYPES + ["Normal"]))
    matching = ((df.fraud_flag == 0) == (df.fraud_type == "Normal")).all()
    checks.append({"name": "Fraud flags and types agree", "status": "Pass" if valid_flags and valid_types and matching else "Fail", "detail": "Normal rows use flag 0; fraud rows use flag 1 and a supported type"})

    behavior_errors = 0
    if date_error is None:
        flagged = chronological[chronological.fraud_flag == 1]
        rapid_customers = set()
        repeated_customers = set()
        flagged_by_customer = {customer: events for customer, events in flagged.groupby("customer_id", sort=False)}
        for customer, events in flagged_by_customer.items():
            if len(events) >= 2:
                repeated_customers.add(customer)
            redeem_times = pd.to_datetime(events.loc[events.transaction_type == "redeem", "transaction_date"]).sort_values()
            if len(redeem_times) >= 2 and (redeem_times.diff().dropna() <= pd.Timedelta(5, unit="min")).any():
                rapid_customers.add(customer)
        for row in flagged.itertuples(index=False):
            if row.fraud_type == "Abnormal accumulation":
                expected = points_for_purchase(row.purchase_amount, row.customer_segment, row.earning_ratio)
                behavior_errors += int(row.transaction_type != "purchase" or int(row.points_earned) <= expected)
            elif row.fraud_type == "Large redemption":
                behavior_errors += int(row.transaction_type != "redeem" or int(row.points_redeemed) < LARGE_REDEMPTION_POINTS or float(row.purchase_amount) != 0.0)
            elif row.fraud_type in {"Rapid redemption", "Repeated suspicious activity"}:
                behavior_errors += int(row.customer_id not in rapid_customers or row.transaction_type != "redeem")
                if row.fraud_type == "Repeated suspicious activity":
                    behavior_errors += int(row.customer_id not in repeated_customers)
    checks.append({"name": "Fraud labels match generated behavior", "status": "Pass" if date_error is None and behavior_errors == 0 else "Fail", "detail": f"{behavior_errors} fraud labels lack their required accumulation or redemption pattern"})
    return checks
