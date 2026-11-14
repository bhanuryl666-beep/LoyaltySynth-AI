"""Dashboard summaries, Plotly-ready series, and leakage-safe demo classifier."""

import pandas as pd


def summarize_transactions(df):
    rows = len(df)
    fraud_count = int(df["fraud_flag"].astype(int).sum()) if rows and "fraud_flag" in df else 0
    return {
        "transactions": rows,
        "customers": int(df.customer_id.nunique()) if rows and "customer_id" in df else 0,
        "revenue": float(df.purchase_amount.sum()) if rows and "purchase_amount" in df else 0.0,
        "points_earned": int(df.points_earned.sum()) if rows and "points_earned" in df else 0,
        "points_redeemed": int(df.points_redeemed.sum()) if rows and "points_redeemed" in df else 0,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_count / rows * 100 if rows else 0.0,
        "fraud_types": df[df.fraud_flag == 1].groupby("fraud_type").size().to_dict() if rows and {"fraud_flag", "fraud_type"}.issubset(df.columns) else {},
        "categories": df.groupby("product_category").purchase_amount.sum().sort_values(ascending=False).to_dict() if rows and {"product_category", "purchase_amount"}.issubset(df.columns) else {},
        "segments": df.groupby("customer_segment").size().to_dict() if rows and "customer_segment" in df else {},
        "monthly_redemptions": df.assign(month=pd.to_datetime(df.transaction_date, errors="coerce").dt.to_period("M").astype(str)).groupby("month").points_redeemed.sum().tail(12).to_dict() if rows and {"transaction_date", "points_redeemed"}.issubset(df.columns) else {},
    }


def evaluate_fraud_model(df):
    """Train/test a small classifier; label columns are never used as features."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    if df.empty or "fraud_flag" not in df or df.fraud_flag.nunique() < 2:
        return {"available": False, "message": "Generate a dataset with some fraud labels to train the demonstration classifier."}
    feature_cols = ["transaction_type", "purchase_amount", "points_earned", "points_redeemed", "points_balance", "product_category", "payment_method", "customer_segment"]
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    y = df.fraud_flag.astype(int)
    if y.value_counts().min() < 2:
        return {"available": False, "message": "At least two normal and two flagged rows are needed for a train/test split."}
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    transform = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    model = Pipeline([("features", transform), ("classifier", RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=42))])
    groups = df.customer_id.astype(str) if "customer_id" in df else pd.Series([str(i) for i in range(len(df))], index=df.index)
    split = None
    # Retry deterministic grouped splits to find one with both labels in train
    # and test. No customer's rows can then leak across the evaluation boundary.
    splitter = GroupShuffleSplit(n_splits=50, test_size=0.25, random_state=42)
    for train_indices, test_indices in splitter.split(X, y, groups):
        if y.iloc[train_indices].nunique() == 2 and y.iloc[test_indices].nunique() == 2:
            split = (train_indices, test_indices)
            break
    if split is None:
        return {"available": False, "message": "A customer-grouped train/test split with both labels was not possible. Generate more customers and fraud examples."}
    train_indices, test_indices = split
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
    return {"available": True, "accuracy": accuracy_score(y_test, predicted), "precision": precision_score(y_test, predicted, zero_division=0), "recall": recall_score(y_test, predicted, zero_division=0), "f1": f1_score(y_test, predicted, zero_division=0), "confusion": matrix, "feature_columns": list(X.columns), "grouped_split": True}
