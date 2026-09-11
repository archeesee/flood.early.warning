# Flood Risk Early-Warning Broadcast System

A regression-model-powered flood risk predictor wired into an **automated SMS alert app**,
instead of a dashboard people have to remember to check.

## What's in here
- `Flood_Risk_Regression_Model.ipynb` — full ML workflow: EDA → preprocessing → Linear Regression
  baseline vs Ridge/Lasso/Random Forest/XGBoost → validation (train/test + 5-fold CV, MAE/MSE/RMSE/R²)
  → feature importance → exports the trained model.
- `app.py` — Flask backend: subscriber registration, region risk scoring, automatic alert broadcast
  when predicted risk crosses a threshold (real SMS via Twilio if credentials are set, otherwise a
  clearly-logged simulation — great for a live demo without a paid account).
- `templates/index.html` — dashboard: register test subscribers, punch in factor values for a
  region, watch the alert fire and log in real time.

## Setup

1. **Get the dataset.** Download `train.csv` from Kaggle — search "Regression with a Flood
   Prediction Dataset" (Playground Series S4E5) or "Flood Prediction Factors" — and place it in
   this folder (or upload it into Colab).

2. **Run the notebook** (Colab or Jupyter) top to bottom. It trains the models, prints the
   comparison table + validation metrics + feature importance, and saves:
   ```
   model_artifacts/flood_model.pkl
   model_artifacts/scaler.pkl
   model_artifacts/feature_columns.pkl
   model_artifacts/uses_scaling.pkl
   ```
   Copy the `model_artifacts/` folder next to `app.py`.

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **(Optional) Enable real SMS.** Set these environment variables with a free Twilio trial
   account before starting the app — this is the "wow" moment for your presentation:
   ```
   export TWILIO_ACCOUNT_SID=xxxx
   export TWILIO_AUTH_TOKEN=xxxx
   export TWILIO_FROM_NUMBER=+1xxxxxxxxxx
   ```
   Without these set, the app still works fully — alerts are logged and shown on the dashboard as
   "simulated," which is completely fine for the assignment (just say so in your report/Appendix).

5. **Run the app:**
   ```
   python app.py
   ```
   Open http://127.0.0.1:5000, register a subscriber with your own phone number and your area,
   then use "Fill Extreme Values (demo)" + "Score Region & Check Alert" to trigger a real alert.

## For your report (Section 5–8 of the brief)
- **Model Development / Validation:** pull the metrics table and residual plots straight from the
  notebook output.
- **Model Interpretation:** use the feature-importance chart — this is your "important predictors"
  discussion.
- **Deployment:** screenshot the dashboard mid-alert (risk score + alert log populated) plus, if you
  enabled Twilio, a screenshot of the actual SMS received on a phone.
- **Limitations/Future Scope:** see the notebook's final markdown cell — already drafted for you,
  edit to taste.
