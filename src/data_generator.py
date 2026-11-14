"""Rule-based seed/fallback and cached SDV CTGAN transaction generation."""

from datetime import timedelta
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from src.loyalty_rules import points_for_purchase, recalculate_balances, transaction_id

CATEGORIES = ["Grocery", "Electronics", "Home", "Travel", "Fashion", "Dining", "Beauty"]
PAYMENTS = ["Credit card", "Debit card", "Wallet", "Cash"]
SEGMENTS = ["Standard", "Premium", "VIP"]
SCHEMA = ["transaction_id", "customer_id", "transaction_date", "transaction_type", "purchase_amount", "points_earned", "points_redeemed", "points_balance", "product_category", "payment_method", "customer_segment", "earning_ratio", "fraud_flag", "fraud_type"]
MODEL_EPOCHS = 100


def _profiles(customers, segments, seed):
    rng = random.Random(seed)
    allowed = [segment for segment in (segments or SEGMENTS) if segment in SEGMENTS] or SEGMENTS
    profiles = {}
    for index in range(customers):
        customer_id = f"CUS-{index + 1:05d}"
        favorites = rng.sample(CATEGORIES, 2)
        profiles[customer_id] = {
            "segment": rng.choice(allowed),
            "favorites": favorites,
            "payment": rng.choice(PAYMENTS),
            # Higher-frequency members receive more transactions in a dataset.
            "frequency": rng.uniform(0.6, 1.8),
        }
    return profiles


def _allocate_customer_ids(count, profiles, rng):
    customer_ids = list(profiles)
    counts = {customer: 1 for customer in customer_ids}
    remaining = count - len(customer_ids)
    if remaining > 0:
        weights = [profiles[customer]["frequency"] for customer in customer_ids]
        for customer in rng.choices(customer_ids, weights=weights, k=remaining):
            counts[customer] += 1
    allocation = [customer for customer, total in counts.items() for _ in range(total)]
    rng.shuffle(allocation)
    return allocation


def _inclusive_date_bounds(start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end_exclusive = pd.Timestamp(end_date).normalize() + pd.Timedelta(1, unit="D")
    if start >= end_exclusive:
        raise ValueError("Start date must be on or before end date")
    return start, end_exclusive


def generate_seed_dataset(count=800, customers=120, start_date="2025-01-01", end_date="2025-12-31", earning_ratio=1.0, segments=None, seed=42):
    """Create a loyalty-shaped seed table for CTGAN training."""
    return generate_rule_based(count, customers, start_date, end_date, earning_ratio, segments, seed)


def generate_rule_based(count=500, customers=100, start_date="2025-01-01", end_date="2025-12-31", earning_ratio=1.0, segments=None, seed=None):
    """Create personalized customer transactions in chronological order."""
    count, customers = int(count), int(customers)
    if not 1 <= count <= 10000:
        raise ValueError("Transactions must be between 1 and 10,000")
    if not 1 <= customers <= count:
        raise ValueError("Customers must be between 1 and the transaction count")
    ratio = float(earning_ratio)
    if not 0.1 <= ratio <= 10:
        raise ValueError("Earning ratio must be between 0.1 and 10")
    start, end_exclusive = _inclusive_date_bounds(start_date, end_date)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    profiles = _profiles(customers, segments, seed)
    allocation = _allocate_customer_ids(count, profiles, rng)
    span_seconds = max(1, int((end_exclusive - start).total_seconds()))
    scheduled = []
    for customer in allocation:
        timestamp = start + pd.Timedelta(rng.randrange(span_seconds), unit="s")
        scheduled.append((timestamp, customer))
    scheduled.sort(key=lambda item: (item[0], item[1]))

    balances = {}
    rows = []
    for timestamp, customer in scheduled:
        profile = profiles[customer]
        balance = balances.get(customer, 0)
        is_redemption = balance >= 100 and rng.random() < 0.22
        transaction_type = "redeem" if is_redemption else "purchase"
        amount = 0.0 if is_redemption else round(max(1.0, float(np_rng.lognormal(3.5, 0.75))), 2)
        earned = 0 if is_redemption else points_for_purchase(amount, profile["segment"], ratio)
        redeemed = rng.randint(1, max(1, balance // 2)) if is_redemption else 0
        balances[customer] = balance + earned - redeemed
        favorite = rng.choices(profile["favorites"], weights=[0.75, 0.25], k=1)[0]
        category = favorite if rng.random() < 0.85 else rng.choice(CATEGORIES)
        rows.append({
            "customer_id": customer,
            "transaction_date": timestamp.isoformat(),
            "transaction_type": transaction_type,
            "purchase_amount": amount,
            "points_earned": earned,
            "points_redeemed": redeemed,
            "points_balance": balances[customer],
            "product_category": category,
            "payment_method": profile["payment"] if rng.random() < 0.9 else rng.choice(PAYMENTS),
            "customer_segment": profile["segment"],
            "earning_ratio": ratio,
            "fraud_flag": 0,
            "fraud_type": "Normal",
        })
    result = pd.DataFrame(rows)
    result.insert(0, "transaction_id", [transaction_id(i + 1) for i in range(len(result))])
    result = recalculate_balances(result)
    return result[SCHEMA]


def _model_metadata_path(model_path):
    return Path(str(model_path) + ".json")


def _read_compatible_model(model_path, expected_metadata, synthesizer_class):
    metadata_path = _model_metadata_path(model_path)
    if not Path(model_path).is_file() or not metadata_path.is_file():
        return None
    try:
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored != expected_metadata:
            return None
        return synthesizer_class.load(str(model_path))
    except Exception:
        return None


def generate_with_ctgan(count, customers, start_date, end_date, earning_ratio=1.0, segments=None, seed=None, model_path="models/loyalty_ctgan.pkl", force_retrain=False):
    """Load a compatible SDV CTGAN artifact or train/save it when needed.

    CTGAN supplies synthetic purchase amounts, transaction tendencies, dates,
    and payment patterns. Customer profiles and loyalty invariants are then
    applied deterministically to every sampled row.
    """
    from sdv import __version__ as sdv_version
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    count, customers = int(count), int(customers)
    if not 1 <= count <= 10000 or not 1 <= customers <= count:
        raise ValueError("Customers and transactions must be between 1 and 10,000, with customers no greater than transactions")
    ratio = float(earning_ratio)
    if not 0.1 <= ratio <= 10:
        raise ValueError("Earning ratio must be between 0.1 and 10")
    start, end_exclusive = _inclusive_date_bounds(start_date, end_date)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    seed_df = generate_seed_dataset(max(500, min(1000, count)), max(50, min(customers, 200)), start, end_exclusive - pd.Timedelta(days=1), ratio, segments, seed or 42)
    seed_df["transaction_date"] = pd.to_datetime(seed_df["transaction_date"])
    expected_metadata = {"schema": SCHEMA, "sdv_version": sdv_version, "epochs": MODEL_EPOCHS}
    model = None if force_retrain else _read_compatible_model(model_path, expected_metadata, CTGANSynthesizer)
    model_reused = model is not None

    def train_and_save():
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(seed_df)
        metadata.update_column("transaction_id", sdtype="id")
        fitted = CTGANSynthesizer(metadata, epochs=MODEL_EPOCHS, verbose=False)
        fitted.fit(seed_df)
        fitted.save(str(model_path))
        _model_metadata_path(model_path).write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
        return fitted

    if model is None:
        model = train_and_save()
    try:
        synthetic = model.sample(num_rows=count)
    except Exception:
        if not model_reused:
            raise
        model = train_and_save()
        synthetic = model.sample(num_rows=count)
        model_reused = False

    start_utc = start
    last_second = end_exclusive - pd.Timedelta(seconds=1)
    sampled_dates = pd.to_datetime(synthetic["transaction_date"], errors="coerce", utc=True).dt.tz_convert(None)
    synthetic["transaction_date"] = sampled_dates.fillna(start_utc).clip(lower=start_utc, upper=last_second)
    synthetic = synthetic.sort_values("transaction_date", kind="stable").reset_index(drop=True)
    profiles = _profiles(customers, segments, seed)
    rng = random.Random(seed)
    customer_allocation = _allocate_customer_ids(count, profiles, rng)
    candidate_types = synthetic.get("transaction_type", pd.Series(["purchase"] * count)).astype(str).tolist()
    candidate_amounts = pd.to_numeric(synthetic.get("purchase_amount", 0), errors="coerce")
    if not isinstance(candidate_amounts, pd.Series):
        candidate_amounts = pd.Series([candidate_amounts] * count)
    candidate_redeems = pd.to_numeric(synthetic.get("points_redeemed", 0), errors="coerce")
    if not isinstance(candidate_redeems, pd.Series):
        candidate_redeems = pd.Series([candidate_redeems] * count)
    balances = {}
    rows = []
    for index in range(count):
        customer = customer_allocation[index]
        profile = profiles[customer]
        available = balances.get(customer, 0)
        requested_redeem = str(candidate_types[index]).lower() == "redeem" and available >= 100
        if requested_redeem:
            kind, amount, earned = "redeem", 0.0, 0
            redeemed = min(available, max(1, int(candidate_redeems.iloc[index] if pd.notna(candidate_redeems.iloc[index]) else 1)))
        else:
            kind = "purchase"
            raw_amount = candidate_amounts.iloc[index]
            amount = round(max(1.0, float(raw_amount) if pd.notna(raw_amount) else 1.0), 2)
            earned = points_for_purchase(amount, profile["segment"], ratio)
            redeemed = 0
        balances[customer] = available + earned - redeemed
        favorite = rng.choices(profile["favorites"], weights=[0.75, 0.25], k=1)[0]
        category = favorite if rng.random() < 0.85 else rng.choice(CATEGORIES)
        payment = synthetic.iloc[index].get("payment_method", profile["payment"])
        if payment not in PAYMENTS:
            payment = profile["payment"]
        rows.append({
            "customer_id": customer,
            "transaction_date": synthetic.iloc[index]["transaction_date"].isoformat(),
            "transaction_type": kind,
            "purchase_amount": amount,
            "points_earned": earned,
            "points_redeemed": redeemed,
            "points_balance": balances[customer],
            "product_category": category,
            "payment_method": payment,
            "customer_segment": profile["segment"],
            "earning_ratio": ratio,
            "fraud_flag": 0,
            "fraud_type": "Normal",
        })
    result = pd.DataFrame(rows)
    result.insert(0, "transaction_id", [transaction_id(i + 1) for i in range(len(result))])
    result = recalculate_balances(result)
    result = result[SCHEMA]
    result.attrs["ctgan_model_reused"] = model_reused
    return result
