# 🎤 How to explain this project in interviews

You will be asked about every line you publish. Read this sheet, run the project yourself, and change the wording to your own voice.

## "Walk me through your project" (30-second pitch)
"I built an end-to-end churn prediction system for a telecom dataset. I compared three models with cross-validated AUC, and the key feature is explainability — for each customer the app shows the top SHAP factors driving their churn risk, because a retention team needs to know *why*, not just *who*."

## "Why AUC and not accuracy?"
The classes are imbalanced: 73% stayed, 27% churned. A model predicting 'stays' for everyone gets 73% accuracy while catching zero churners. AUC measures ranking quality across all thresholds, which is what you want for churn.

## "What is SHAP?"
SHAP values come from game theory (Shapley values): each feature gets a share of the prediction, positive = pushes toward churn, negative = toward retention, and the shares sum up to the prediction's deviation from the average. That makes every single prediction explainable.

## "Why XGBoost over logistic regression?"
I compared both by cross-validated AUC — that comparison is in the training script. Trees handle non-linear effects (e.g. tenure vs churn) and the tenure × charges interaction without manual feature engineering. If asked honestly: logistic regression is competitive on this dataset and more interpretable; I chose based on measured AUC, not fashion.

## "Hardest bug you hit?"
The dataset stores TotalCharges as a string column with ~11 blank values. A naive model.fit() crashes or silently corrupts training. Fix: pd.to_numeric(errors='coerce') + fill blanks with MonthlyCharges × tenure.

## "Why class_weight='balanced'?"
Without it, models optimize for the majority class and barely detect churners — the class we actually care about. Balanced weights penalize misses on the minority class.

## "How would you take this to production?"
Batch-score nightly instead of real-time (churn decisions aren't urgent), monitor input drift and AUC decay over time, and pick the alert threshold from retention economics (offer cost vs customer lifetime value), not 0.5.

## "What would you improve next?"
RandomizedSearchCV for hyperparameters, probability calibration, per-segment models, and a cost-based threshold instead of 0.5.

## One rule
If you can't explain it, don't claim it. Everything above maps to real code in this repo — read src/train.py and app.py before the interview.
