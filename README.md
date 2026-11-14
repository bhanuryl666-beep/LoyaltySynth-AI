# LoyaltySynth AI

**Synthetic Loyalty Transaction Generator**

LoyaltySynth AI is a Flask application for generating fictional retail loyalty data for analytics, fraud-detection experiments, and personalization prototypes. It combines configurable business rules with optional SDV CTGAN synthesis and provides validation, interactive charts, and CSV export.

> All records and fraud behaviors are synthetic. They are intended for development and education, not for evaluating or making decisions about real customers.

## Features

- Generate transactions with configurable volume, customers, date range, earning ratio, segments, and simulated fraud rate.
- Choose a fast rule-based generator or SDV CTGAN.
- Reuse a compatible saved CTGAN model, or explicitly retrain it.
- Apply customer-specific shopping frequency and product-category preferences.
- Enforce loyalty point rules and rebuild balances chronologically.
- Simulate labeled accumulation and redemption patterns for fraud analytics.
- Review data-quality checks, charts, classifier metrics, and a transaction preview.
- Export the active dataset or the fraud-labeled dataset as CSV.

## Technology

| Area | Technology |
| --- | --- |
| Web application | Python, Flask, Jinja2 |
| Synthetic generation | SDV CTGAN and rule-based fallback |
| Data processing | Pandas, NumPy |
| Demonstration classifier | scikit-learn Random Forest |
| Charts | Plotly.js |
| Optional persistence | SQLAlchemy with PyMySQL |

## Quick start on Windows

Open the project folder in VS Code. In the integrated PowerShell terminal, create and activate an environment, install the dependencies, and start the app:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in your browser. Keep the terminal open while using the app; press **Ctrl+C** there to stop the server.

If PowerShell does not allow environment activation, run the environment's Python executable directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

## Deploy to Render

The repository includes a Render Blueprint in `render.yaml`. It configures a Python web service, installs `requirements.txt`, starts the app with Gunicorn on Render's assigned port, disables debug mode, and asks Render to generate the Flask secret key.

1. Create a GitHub repository for this project.
2. From the project folder, initialize Git and push the project:

   ```powershell
   git init
   git add .
   git commit -m "Prepare LoyaltySynth AI for Render"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/loyaltysynth-ai.git
   git push -u origin main
   ```

3. In Render, choose **New → Blueprint**, connect the GitHub repository, and apply the `render.yaml` configuration.
4. Wait for the build and deployment to finish, then open the `onrender.com` URL shown on the service page.

The Blueprint uses Render's Free web service plan. Free services have an ephemeral filesystem, so generated CSV changes and newly trained CTGAN files can be lost when the service restarts or spins down. The committed seed data and any committed CTGAN artifact are available on deploy. For persistent generated data, use a paid persistent disk or connect durable database storage. The app accepts `LOYALTYSYNTH_DATA_DIR` and `LOYALTYSYNTH_MODEL_DIR` if you attach a disk and want those files stored there.

Render free services can spin down after inactivity, so the first request after idle may take longer while the service starts. CTGAN training is also resource intensive; use the saved model when compatible and avoid selecting **Retrain and replace** unless you intend to wait for training.

## Using the application

1. Open **Generator** and choose the transaction count, customer count, date range, earning ratio, segments, and fraud percentage.
2. Select **Business rules** for fast generation or **SDV CTGAN** for model-based synthesis.
3. Review the dashboard, analytics, and data-quality pages.
4. Download the generated CSV from the header or use the generator page's fraud-labeled export.

The first run creates a 500-row demo dataset in `data/seed_transactions.csv`. Generating a new dataset replaces the active CSV.

## Generation engines

### Business rules

The rule-based generator is the default and fallback. It creates chronological transactions, assigns consistent customer segments and shopping preferences, awards points by purchase amount and segment, and prevents redemptions from exceeding available points. The selected end date is included.

### SDV CTGAN

When CTGAN is selected, the application creates a rule-based seed table and fits an SDV `CTGANSynthesizer` if no compatible saved model is available. The model is stored at `models/loyalty_ctgan.pkl`, with a JSON sidecar containing its schema and SDV version. Compatible models are reused; select **Retrain and replace** to train a fresh model.

CTGAN samples are post-processed to apply customer profiles and enforce transaction, earning, redemption, and chronological balance rules. If CTGAN cannot load, train, or sample, the application logs the error and reports when it falls back to rule-based generation. Model training can take several minutes.

## Project structure

```text
app.py                      Flask routes and application setup
src/
  analytics.py              Dashboard summaries and classifier evaluation
  data_generator.py         Rule-based and CTGAN generation
  data_validator.py         Schema, loyalty, and fraud behavior checks
  database.py               Optional MySQL persistence helper
  fraud_generator.py        Synthetic fraud event generation
  loyalty_rules.py          Points and balance rules
templates/                  Jinja2 pages
static/
  css/style.css             Application styles
  images/                   Brand assets
  js/script.js              Form validation and Plotly charts
data/                       Active transaction CSV files
models/                     Saved CTGAN model and metadata
tests/                      Automated tests
```

## Configuration and limitations

- `LOYALTYSYNTH_SECRET_KEY` can be set to configure Flask's session secret. If unset, the app generates a random secret for the current process.
- Debug mode is disabled by default. Set `FLASK_DEBUG=1` only for local development.
- Plotly.js and the web fonts are loaded from public CDNs, so those visual assets require an internet connection.
- Fraud patterns and classifier metrics are demonstrations based on synthetic labels. Metrics do not represent real-world fraud detection performance.
- The active dataset is stored in a local CSV shared by the app. MySQL is optional and is not connected automatically.

## Run the tests

From the project root, run:

```powershell
python -m unittest discover -v
```

The tests cover points calculations, redemption limits, chronological balances, inclusive date ranges, customer preferences and frequency, fraud behavior, and CSV export.

## Optional MySQL adapter

To use the persistence helper, set `LOYALTYSYNTH_DATABASE_URL` to a SQLAlchemy URL, for example:

```powershell
$env:LOYALTYSYNTH_DATABASE_URL = "mysql+pymysql://user:password@localhost/loyaltysynth"
```

Application code can then call `save_to_mysql(dataframe, "transactions")` from `src.database`. The web app does not connect to MySQL unless code explicitly invokes the helper.
