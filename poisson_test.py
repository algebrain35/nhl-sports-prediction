from scipy.stats import poisson
from sklearn.metrics import log_loss
import numpy as np
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def get_features(fpath):
    rows = None
    with open(fpath, 'r') as f:
        rows = json.load(f)
    return [row for row in rows]

data = pd.read_csv("all_games_preproc.csv")
feats = get_features("all_games_preproc_features.json")

subset = data[feats + ["goalsFor", "goalsAgainst"]].copy()
subset[feats] = subset[feats].fillna(0)
subset = subset.dropna(subset=["goalsFor", "goalsAgainst"])

X = subset[feats]
y_gf = subset["goalsFor"]
y_ga = subset["goalsAgainst"]

X_train, X_test, y_train_gf, y_test_gf, y_train_ga, y_test_ga = train_test_split(
    X, y_gf, y_ga, test_size=0.2, random_state=42
)

model_gf = xgb.XGBRegressor(objective="count:poisson", n_estimators=300,
                              max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7)
model_ga = xgb.XGBRegressor(objective="count:poisson", n_estimators=300,
                              max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7)

model_gf.fit(X_train, y_train_gf)
model_ga.fit(X_train, y_train_ga)

lambda_gf = model_gf.predict(X_test)
lambda_ga = model_ga.predict(X_test)
lambda_total = lambda_gf + lambda_ga

# ── Evaluate as an O/U model, not a point predictor ──────────────
y_total_test = y_gf.loc[X_test.index] + y_ga.loc[X_test.index]

for line in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]:
    # Model's predicted P(over)
    if line % 1 == 0.5:
        p_over = 1 - poisson.cdf(int(line - 0.5), lambda_total)
    else:
        p_over = 1 - poisson.cdf(int(line), lambda_total)

    # Actual outcomes
    actual_over = (y_total_test > line).astype(int).values

    # Log loss — how well calibrated are the probabilities?
    ll = log_loss(actual_over, p_over)

    # Accuracy — if we bet over when p > 0.5
    pred_over = (p_over > 0.5).astype(int)
    acc = (pred_over == actual_over).mean()

    # Calibration — when model says ~55% over, does it hit ~55%?
    mean_pred = p_over.mean()
    mean_actual = actual_over.mean()

    print(f"Line {line}: acc={acc:.3f}  log_loss={ll:.4f}  "
          f"pred_over_rate={mean_pred:.3f}  actual_over_rate={mean_actual:.3f}")
