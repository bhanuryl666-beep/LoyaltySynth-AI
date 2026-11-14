"""Flask entry point for LoyaltySynth AI."""

from pathlib import Path
import os
import secrets

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from src.analytics import evaluate_fraud_model, summarize_transactions
from src.data_generator import SCHEMA, generate_rule_based, generate_with_ctgan
from src.data_validator import validate_transactions
from src.fraud_generator import inject_fraud
from src.loyalty_rules import parse_date, points_for_purchase, recalculate_balances

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LOYALTYSYNTH_DATA_DIR", ROOT / "data")).resolve()
MODEL_DIR = Path(os.environ.get("LOYALTYSYNTH_MODEL_DIR", ROOT / "models")).resolve()
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "seed_transactions.csv"
app = Flask(__name__)
app.secret_key = os.environ.get("LOYALTYSYNTH_SECRET_KEY") or secrets.token_hex(32)


def load_transactions():
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        inject_fraud(generate_rule_based(500, 100), 0.03).to_csv(DATA_FILE, index=False)
        return pd.read_csv(DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    # Migrate earlier generated datasets once, preserving their transactions
    # and estimating the configured ratio from ordinary purchase rows.
    checks = validate_transactions(df)
    if not set(SCHEMA).issubset(df.columns) or any(check["status"] != "Pass" for check in checks):
        core_schema = set(SCHEMA) - {"earning_ratio", "fraud_flag", "fraud_type"}
        if not core_schema.issubset(df.columns):
            df = inject_fraud(generate_rule_based(500, 100), 0.03)
        else:
            segment_multipliers = {"Standard": 1.0, "Premium": 1.5, "VIP": 2.0}
            ordinary = df[df.fraud_flag == 0] if "fraud_flag" in df else df
            ordinary = ordinary[(ordinary.transaction_type == "purchase") & (ordinary.purchase_amount > 0)] if not ordinary.empty else ordinary
            estimates = [int(row.points_earned) / (float(row.purchase_amount) * segment_multipliers.get(row.customer_segment, 1.0)) for row in ordinary.itertuples()]
            ratio = round(float(pd.Series(estimates).median()), 1) if estimates else 1.0
            ratio = min(10.0, max(0.1, ratio))
            df["earning_ratio"] = ratio
            old_rate = float(df["fraud_flag"].mean()) if "fraud_flag" in df else 0.03
            for index, row in df.iterrows():
                if row.transaction_type == "purchase":
                    df.at[index, "points_earned"] = points_for_purchase(row.purchase_amount, row.customer_segment, ratio)
                    df.at[index, "points_redeemed"] = 0
                elif row.transaction_type == "redeem":
                    df.at[index, "purchase_amount"] = 0.0
                    df.at[index, "points_earned"] = 0
            df = inject_fraud(recalculate_balances(df), old_rate)
        df.to_csv(DATA_FILE, index=False)
    return df


def preview_rows(df, limit=12):
    view = df.head(limit).copy()
    view = view.where(pd.notna(view), "")
    return {"columns": list(view.columns), "rows": view.astype(str).values.tolist()}


@app.route("/")
def index():
    df = load_transactions()
    return render_template("index.html", summary=summarize_transactions(df), preview=preview_rows(df, 6))


@app.route("/generator", methods=["GET", "POST"])
def generator():
    if request.method == "POST":
        try:
            count = int(request.form.get("count", "500"))
            customers = int(request.form.get("customers", "100"))
            start_date = parse_date(request.form.get("start_date", "2025-01-01")).isoformat()
            end_date = parse_date(request.form.get("end_date", "2025-12-31")).isoformat()
            ratio = float(request.form.get("earning_ratio", "1"))
            fraud_percent = float(request.form.get("fraud_percent", "3"))
            segments = request.form.getlist("segments")
            if not segments:
                raise ValueError("Select at least one customer segment")
            mode = request.form.get("mode", "rules")
            if mode not in {"rules", "ctgan"}:
                raise ValueError("Choose a supported generation engine")
            options = (count, customers, start_date, end_date, ratio, segments)
            generation_method = "rule-based synthetic data"
            if mode == "ctgan":
                try:
                    force_retrain = request.form.get("retrain_model") == "on"
                    df = generate_with_ctgan(*options, seed=42, model_path=str(MODEL_DIR / "loyalty_ctgan.pkl"), force_retrain=force_retrain)
                    generation_method = "SDV CTGAN-generated data (trained a new model)" if not df.attrs.get("ctgan_model_reused") else "SDV CTGAN-generated data (reused the saved model)"
                except Exception as exc:  # Optional dependency/runtime failure: keep the app usable.
                    app.logger.exception("CTGAN generation failed; using rule-based generator")
                    df = generate_rule_based(*options)
                    flash(f"CTGAN was unavailable ({type(exc).__name__}); generated this dataset with business rules instead.", "warning")
            else:
                df = generate_rule_based(*options)
            df = inject_fraud(df, fraud_percent / 100)
            df.to_csv(DATA_FILE, index=False)
            flash(f"Created {len(df):,} rows using {generation_method}.", "success")
            return redirect(url_for("generator"))
        except (TypeError, ValueError) as exc:
            flash(str(exc), "error")
        except Exception:
            app.logger.exception("Unexpected error while handling POST /generator")
            flash("Generation failed unexpectedly. Your current dataset was not replaced; check the application log for details.", "error")
    return render_template("generator.html", summary=summarize_transactions(load_transactions()))


@app.route("/analytics")
def analytics():
    df = load_transactions()
    return render_template("analytics.html", summary=summarize_transactions(df), model=evaluate_fraud_model(df))


@app.route("/data-quality")
def data_quality():
    df = load_transactions()
    return render_template("data_quality.html", report=validate_transactions(df), row_count=len(df))


@app.route("/download")
def download():
    load_transactions()
    return send_file(DATA_FILE, as_attachment=True, download_name="loyaltysynth_transactions.csv", mimetype="text/csv")


@app.route("/download/fraud")
def download_fraud():
    df = load_transactions()
    output = DATA_DIR / "fraud_labeled_transactions.csv"
    df.to_csv(output, index=False)
    return send_file(output, as_attachment=True, download_name="loyaltysynth_fraud_labeled.csv", mimetype="text/csv")


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_enabled)
