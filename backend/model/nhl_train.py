import os
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error,
)
from scipy.stats import uniform, randint, poisson, norm

# Import Skellam functions from inference module for consistency
from nhl_model import skellam_pmf, skellam_cdf, spread_probability, ml_probability_from_poisson

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EVENTS          = ("ml", "spread", "ou", "poisson")
N_FEATURES      = 40
N_SPLITS_CV     = 5
EARLY_STOP      = 50
MAX_BOOST_ROUND = 1000

DEFAULT_PARAMS: dict[str, dict] = {
    "ml": {
        "objective":    "binary:logistic",
        "eval_metric":  "logloss",
        "booster":      "gbtree",
        "learning_rate": 0.02,
        "max_depth":    3,
    },
    "spread": {
        "objective":    "binary:logistic",
        "eval_metric":  "logloss",
        "booster":      "gbtree",
        "learning_rate": 0.05,
        "max_depth":    2,
    },
    "ou": {
        "objective":    "binary:logistic",
        "eval_metric":  "logloss",
        "booster":      "gbtree",
        "learning_rate": 0.04,
        "max_depth":    2,
        "min_child_weight": 3,
        "reg_alpha":    0.4,
        "subsample":    0.9,
    },
    "poisson_gf": {
        "objective":    "count:poisson",
        "eval_metric":  "poisson-nloglik",
        "booster":      "gbtree",
        "learning_rate": 0.03,
        "max_depth":    4,
        "subsample":    0.8,
        "colsample_bytree": 0.7,
    },
    "poisson_ga": {
        "objective":    "count:poisson",
        "eval_metric":  "poisson-nloglik",
        "booster":      "gbtree",
        "learning_rate": 0.03,
        "max_depth":    4,
        "subsample":    0.8,
        "colsample_bytree": 0.7,
    },
    # Kept for optional comparison — inference no longer requires this model
    "spread_reg": {
        "objective":    "reg:squarederror",
        "eval_metric":  "rmse",
        "booster":      "gbtree",
        "learning_rate": 0.03,
        "max_depth":    4,
        "subsample":    0.8,
        "colsample_bytree": 0.7,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _class_weight(y: pd.Series) -> float:
    """Ratio of negative to positive samples for scale_pos_weight."""
    return (y == 0).sum() / max((y == 1).sum(), 1)


def _save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ── Trainer ───────────────────────────────────────────────────────────────────

class NHLModelTrainer:
    def __init__(self, data_path=None, model_save_path=None):
        self.data_path       = data_path or os.getenv("DATA_PATH", "all_games_preproc.csv")
        self.model_save_path = model_save_path or os.getenv("MODEL_PATH", "./backend/model/models")
        self.params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}

    # ── I/O ───────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, low_memory=False)
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, self.data_path)
        return df

    def _params_path(self, event: str, value=None) -> str:
        if event == "ml":
            return os.path.join(self.model_save_path, "ml", "hyperparameters_ml.json")
        return os.path.join(self.model_save_path, event, str(value), f"hyperparameters_{event}_{value}.json")

    def _model_dir(self, event: str, value=None) -> str:
        if value is not None:
            return os.path.join(self.model_save_path, event, str(value))
        return os.path.join(self.model_save_path, event)

    def load_params(self, event: str, value=None) -> None:
        saved = _load_json(self._params_path(event, value))
        if saved:
            self.params[event].update(saved)
            self.params[event]["objective"] = "binary:logistic"
            self.params[event].pop("n_estimators", None)
            self.params[event].pop("use_label_encoder", None)
            logger.info("Loaded saved params for %s", event)

    # ── Feature selection (inside-split, no leakage) ──────────────────────

    @staticmethod
    def _select_features(X: pd.DataFrame, y: pd.Series, n: int = N_FEATURES) -> list[str]:
        """Fit a fast XGB classifier on training split, return top-n features."""
        n = min(n, X.shape[1])
        selector = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0,
        )
        selector.fit(X, y)
        imp = pd.Series(selector.feature_importances_, index=X.columns)
        return imp.nlargest(n).index.tolist()

    @staticmethod
    def _select_features_regression(X: pd.DataFrame, y: pd.Series, n: int = N_FEATURES) -> list[str]:
        """Feature selection using XGBRegressor for Poisson targets."""
        n = min(n, X.shape[1])
        selector = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=100, max_depth=3, learning_rate=0.1,
            verbosity=0,
        )
        selector.fit(X, y)
        imp = pd.Series(selector.feature_importances_, index=X.columns)
        return imp.nlargest(n).index.tolist()

    @staticmethod
    def _select_features_regression_sq(X: pd.DataFrame, y: pd.Series, n: int = N_FEATURES) -> list[str]:
        """Feature selection using XGBRegressor for squared error targets."""
        n = min(n, X.shape[1])
        selector = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100, max_depth=3, learning_rate=0.1,
            verbosity=0,
        )
        selector.fit(X, y)
        imp = pd.Series(selector.feature_importances_, index=X.columns)
        return imp.nlargest(n).index.tolist()

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _load_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Load saved feature list or fall back to whitelist selection."""
        feature_path = self.data_path.replace(".csv", "_features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                feature_cols = json.load(f)
            feature_cols = [c for c in feature_cols if c in df.columns]
            logger.info("Loaded %d feature columns from %s", len(feature_cols), feature_path)
            return feature_cols

        logger.warning("Feature list not found at %s — using fallback", feature_path)
        SAFE_CONTEXTUAL = {
            "is_home", "elo_diff", "pdo",
            "eloFor", "eloAgainst", "eloExpectedFor", "eloExpectedAgainst",
            "days_rest", "is_back_to_back",
        }
        feature_cols = []
        for c in df.columns:
            is_safe = (
                "_diff_ema_" in c
                or "_momentum" in c
                or c in SAFE_CONTEXTUAL
                or ("_seasonal_ema_span_" in c and ("_span_5" in c or "_span_13" in c))
            )
            if is_safe:
                feature_cols.append(c)
        logger.info("Fallback selected %d feature columns", len(feature_cols))
        return feature_cols

    def preprocess(self, event: str, df: pd.DataFrame, value=None, team=None):
        """
        Prepare features + target for classification events (ml, spread, ou).
        """
        if event not in EVENTS:
            raise ValueError(f"event must be one of {EVENTS}")

        self.load_params(event, value)

        df = df.dropna(axis=1, how="all")
        df = df.drop(
            columns=df.columns[df.columns.str.contains("winner_seasonal_ema_span")],
            errors="ignore",
        )
        df = df[df["goalDiffFor"] != 0]

        if team:
            df = df[df["team"] == team]

        feature_cols = self._load_feature_cols(df)

        # Target variable
        if event == "ml":
            y = df["winner"]
        elif event == "spread":
            if value is None:
                raise ValueError("spread value required")
            df[f"spread_{value}"] = np.where(df["goalDiffFor"] > -value, 1.0, 0.0)
            y = df[f"spread_{value}"]
        elif event == "ou":
            if value is None:
                raise ValueError("ou value required")
            df["total_goals"] = df["goalsFor"] + df["goalsAgainst"]
            df[f"ou_{value}"] = np.where(df["total_goals"] > value, 1.0, 0.0)
            y = df[f"ou_{value}"]

        X = df[feature_cols].copy()
        X = X.dropna(axis=1, how="all")
        y = y.loc[X.index]

        logger.info("Final feature matrix: %d rows × %d cols", *X.shape)
        return X, y

    # ══════════════════════════════════════════════════════════════════════
    # Classification training (ML / binary spread / binary OU)
    # ══════════════════════════════════════════════════════════════════════

    def _train(
        self,
        event:  str,
        X:      pd.DataFrame,
        y:      pd.Series,
        value=  None,
        n_iter: int = 10,
    ) -> tuple[xgb.Booster, float]:
        """Walk-forward training for binary classification events."""
        tscv   = TimeSeriesSplit(n_splits=n_iter)
        params = {k: v for k, v in self.params[event].items()
                  if k not in ("n_estimators", "use_label_encoder")}
        acc_results = []
        final_model = None

        for fold, (train_idx, test_idx) in enumerate(
            tqdm(tscv.split(X), total=n_iter, desc=f"Training [{event}]")
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            top_features = self._select_features(X_train, y_train)
            X_tr = X_train[top_features]
            X_te = X_test[top_features]

            weight = _class_weight(y_train)
            dtrain = xgb.DMatrix(X_tr, label=y_train)
            dtest  = xgb.DMatrix(X_te, label=y_test)

            model = xgb.train(
                {**params, "scale_pos_weight": weight},
                dtrain,
                num_boost_round=MAX_BOOST_ROUND,
                evals=[(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )

            preds  = model.predict(dtest)
            y_pred = (preds > 0.5).astype(int)
            acc    = round(accuracy_score(y_test, y_pred) * 100, 2)
            acc_results.append(acc)
            logger.info("[%s] fold %d  acc=%.2f%%  (rounds=%d)",
                        event, fold, acc, model.best_iteration)
            final_model = model

        final_acc = acc_results[-1]
        self._save_model(final_model, event, final_acc, value)
        self.evaluate(y_test, y_pred)
        self._save_selected_features(top_features, event, value)

        mean_acc = float(np.mean(acc_results))
        logger.info(
            "[%s] CV accuracy: %.2f%% ± %.2f%%  (last fold: %.2f%%)",
            event, mean_acc, float(np.std(acc_results)), final_acc,
        )
        return final_model, mean_acc

    def _save_model(self, model: xgb.Booster, event: str, acc: float, value=None) -> None:
        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"XGBoost_{acc:.2f}%_{event}.json")
        model.save_model(path)
        logger.info("Saved model → %s", path)

    def _save_selected_features(self, features: list[str], event: str, value=None) -> None:
        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"selected_features_{event}.json")
        _save_json(features, path)
        logger.info("Saved %d selected features → %s", len(features), path)

    def train_ml(self, X, y, n_iter=10, team=None):
        return self._train("ml", X, y, n_iter=n_iter)

    def train_spread(self, X, y, spread, n_iter=10):
        return self._train("spread", X, y, value=spread, n_iter=n_iter)

    def train_ou(self, X, y, ou, n_iter=10):
        return self._train("ou", X, y, value=ou, n_iter=n_iter)

    def train_event(self, event, X, y, value=None, n_iter=10):
        if event == "ml":     return self.train_ml(X, y, n_iter=n_iter)
        if event == "spread": return self.train_spread(X, y, value, n_iter=n_iter)
        if event == "ou":     return self.train_ou(X, y, value, n_iter=n_iter)
        raise ValueError(f"Unknown event: {event}")

    # ══════════════════════════════════════════════════════════════════════
    # Poisson training — the unified model
    #
    # Trains dual Poisson regressors (goalsFor + goalsAgainst) and evaluates
    # ALL THREE markets per fold:
    #   - O/U via Poisson CDF on λ_total
    #   - Spread via Skellam CDF on (λ_gf, λ_ga)
    #   - ML via Skellam P(diff > 0)
    # ══════════════════════════════════════════════════════════════════════

    def preprocess_poisson(self, df: pd.DataFrame):
        """
        Prepare data for the dual Poisson model.
        Returns X, y_gf, y_ga, y_winner (for ML evaluation).
        """
        feature_path = self.data_path.replace(".csv", "_features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                feature_cols = json.load(f)
            feature_cols = [c for c in feature_cols if c in df.columns]
            logger.info("Poisson: loaded %d features from %s", len(feature_cols), feature_path)
        else:
            raise FileNotFoundError(f"Feature list required: {feature_path}")

        needed = ["goalsFor", "goalsAgainst", "goalDiffFor", "winner"]
        available = [c for c in needed if c in df.columns]
        subset = df[feature_cols + available].copy()
        subset[feature_cols] = subset[feature_cols].fillna(0)
        subset = subset.dropna(subset=["goalsFor", "goalsAgainst"])

        X      = subset[feature_cols]
        y_gf   = subset["goalsFor"]
        y_ga   = subset["goalsAgainst"]

        logger.info("Poisson: %d rows × %d features", *X.shape)
        return X, y_gf, y_ga, subset

    def train_poisson(
        self,
        X:      pd.DataFrame,
        y_gf:   pd.Series,
        y_ga:   pd.Series,
        df_full: pd.DataFrame = None,
        n_iter: int = 10,
    ) -> tuple[xgb.Booster, xgb.Booster, dict]:
        """
        Walk-forward training for dual Poisson models.
        Evaluates all three markets (ML, O/U, spread) per fold using Skellam.
        """
        tscv = TimeSeriesSplit(n_splits=n_iter)

        params_gf = {k: v for k, v in self.params["poisson_gf"].items()
                     if k not in ("n_estimators", "use_label_encoder")}
        params_ga = {k: v for k, v in self.params["poisson_ga"].items()
                     if k not in ("n_estimators", "use_label_encoder")}

        final_model_gf = None
        final_model_ga = None
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
            tqdm(tscv.split(X), total=n_iter, desc="Training [poisson]")
        ):
            X_train, X_test         = X.iloc[train_idx], X.iloc[test_idx]
            y_train_gf, y_test_gf   = y_gf.iloc[train_idx], y_gf.iloc[test_idx]
            y_train_ga, y_test_ga   = y_ga.iloc[train_idx], y_ga.iloc[test_idx]

            # Feature selection on train split
            top_features = self._select_features_regression(X_train, y_train_gf)
            X_tr = X_train[top_features]
            X_te = X_test[top_features]

            # ── Train goalsFor model ─────────────────────────────────
            dtrain_gf = xgb.DMatrix(X_tr, label=y_train_gf)
            dtest_gf  = xgb.DMatrix(X_te, label=y_test_gf)
            model_gf = xgb.train(
                params_gf, dtrain_gf,
                num_boost_round=MAX_BOOST_ROUND,
                evals=[(dtrain_gf, "train"), (dtest_gf, "eval")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )

            # ── Train goalsAgainst model ─────────────────────────────
            dtrain_ga = xgb.DMatrix(X_tr, label=y_train_ga)
            dtest_ga  = xgb.DMatrix(X_te, label=y_test_ga)
            model_ga = xgb.train(
                params_ga, dtrain_ga,
                num_boost_round=MAX_BOOST_ROUND,
                evals=[(dtrain_ga, "train"), (dtest_ga, "eval")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )

            # ── Predict λ values ─────────────────────────────────────
            lambda_gf = model_gf.predict(dtest_gf)
            lambda_ga = model_ga.predict(dtest_ga)

            # ── Get actual outcomes for test fold ────────────────────
            y_gf_actual = y_test_gf.values
            y_ga_actual = y_test_ga.values
            y_total     = y_gf_actual + y_ga_actual
            y_diff      = y_gf_actual - y_ga_actual

            # Winner: 1 if home won (goalDiff > 0), else 0
            # NHL has no ties — OT/SO always produces a winner
            if df_full is not None and "winner" in df_full.columns:
                y_winner = df_full.iloc[test_idx]["winner"].values
            else:
                y_winner = (y_diff > 0).astype(float)

            # ── Evaluate all three markets ───────────────────────────
            fold_eval = self._evaluate_poisson_fold_unified(
                lambda_gf, lambda_ga, y_total, y_diff, y_winner
            )
            fold_metrics.append(fold_eval)

            logger.info(
                "[poisson] fold %d  gf_rounds=%d  ga_rounds=%d  |  "
                "ML_acc=%.1f%%  OU_5.5_acc=%.1f%%  SPR_-1.5_acc=%.1f%%",
                fold, model_gf.best_iteration, model_ga.best_iteration,
                fold_eval["ml_accuracy"] * 100,
                fold_eval.get("ou_5.5_acc", 0) * 100,
                fold_eval.get("spread_-1.5_acc", 0) * 100,
            )

            final_model_gf = model_gf
            final_model_ga = model_ga

        # ── Save ─────────────────────────────────────────────────────
        self._save_poisson_models(final_model_gf, final_model_ga)
        self._save_selected_features(top_features, "poisson")

        # ── Aggregate metrics ────────────────────────────────────────
        avg_metrics = self._aggregate_metrics(fold_metrics)

        logger.info("")
        logger.info("=" * 70)
        logger.info("POISSON UNIFIED CV RESULTS  (%d folds)", n_iter)
        logger.info("=" * 70)

        # Regression quality
        logger.info("  λ_gf  MSE: %.4f ± %.4f", avg_metrics["mean_mse_gf"], avg_metrics.get("std_mse_gf", 0))
        logger.info("  λ_ga  MSE: %.4f ± %.4f", avg_metrics["mean_mse_ga"], avg_metrics.get("std_mse_ga", 0))
        logger.info("  λ_tot MAE: %.4f ± %.4f", avg_metrics["mean_mae_total"], avg_metrics.get("std_mae_total", 0))

        # ML accuracy
        logger.info("  ML accuracy:      %.2f%% ± %.2f%%",
                     avg_metrics["mean_ml_accuracy"] * 100, avg_metrics.get("std_ml_accuracy", 0) * 100)

        # O/U accuracy at key lines
        for line in [5.0, 5.5, 6.0, 6.5]:
            key = f"mean_ou_{line}_acc"
            if key in avg_metrics:
                logger.info("  O/U %s accuracy:  %.2f%%", line, avg_metrics[key] * 100)

        # Spread accuracy at key lines
        for line in [-1.5, 1.5, -2.5, 2.5]:
            key = f"mean_spread_{line}_acc"
            if key in avg_metrics:
                logger.info("  Spread %s accuracy: %.2f%%", line, avg_metrics[key] * 100)

        logger.info("=" * 70)

        return final_model_gf, final_model_ga, avg_metrics

    # ── Unified fold evaluation (all three markets) ──────────────────────

    def _evaluate_poisson_fold_unified(
        self,
        lambda_gf:  np.ndarray,
        lambda_ga:  np.ndarray,
        y_total:    np.ndarray,
        y_diff:     np.ndarray,
        y_winner:   np.ndarray,
    ) -> dict:
        """
        Evaluate a single fold across all three markets using the Poisson λ pair.
        Uses Skellam for spread + ML, Poisson CDF for O/U.
        """
        lambda_total = lambda_gf + lambda_ga
        metrics = {}

        # ── Regression quality ───────────────────────────────────────
        metrics["mse_gf"]      = float(mean_squared_error(y_diff + lambda_ga, lambda_gf))  # approx
        metrics["mse_ga"]      = float(mean_squared_error(y_total - (y_diff + lambda_ga) + lambda_ga, lambda_ga))  # approx
        # Simpler: just compute directly
        metrics["mse_gf"]      = float(mean_squared_error(
            (y_total + y_diff) / 2, lambda_gf  # goalsFor = (total + diff) / 2
        ))
        metrics["mse_ga"]      = float(mean_squared_error(
            (y_total - y_diff) / 2, lambda_ga  # goalsAgainst = (total - diff) / 2
        ))
        metrics["mse_total"]   = float(mean_squared_error(y_total, lambda_total))
        metrics["mae_total"]   = float(mean_absolute_error(y_total, lambda_total))
        metrics["mean_lambda"] = float(np.mean(lambda_total))
        metrics["mean_actual"] = float(np.mean(y_total))

        # ── ML: Skellam P(diff > 0) ──────────────────────────────────
        ml_probs = ml_probability_from_poisson(lambda_gf, lambda_ga)
        p_home = np.asarray(ml_probs["home"])
        pred_winner = (p_home > 0.5).astype(int)
        metrics["ml_accuracy"] = float(accuracy_score(y_winner.astype(int), pred_winner))

        # ML log loss
        p_home_clip = np.clip(p_home, 1e-7, 1 - 1e-7)
        metrics["ml_logloss"] = float(-np.mean(
            y_winner * np.log(p_home_clip) +
            (1 - y_winner) * np.log(1 - p_home_clip)
        ))

        # ── O/U: Poisson CDF at standard lines ──────────────────────
        for line in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
            probs = self._ou_probability(lambda_total, line)
            p_over = np.asarray(probs["over"])

            if line % 1 == 0.5:
                actual_over = (y_total > line).astype(int)
            else:
                mask = y_total != line
                actual_over = (y_total[mask] > line).astype(int)
                p_over = p_over[mask]

            pred_over = (p_over > 0.5).astype(int)
            metrics[f"ou_{line}_acc"] = float(np.mean(pred_over == actual_over))

            p_clip = np.clip(p_over, 1e-7, 1 - 1e-7)
            metrics[f"ou_{line}_logloss"] = float(-np.mean(
                actual_over * np.log(p_clip) + (1 - actual_over) * np.log(1 - p_clip)
            ))

        # ── Spread: Skellam at standard lines ────────────────────────
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            threshold = -line
            probs = spread_probability(lambda_gf, lambda_ga, line)
            p_cover = np.asarray(probs["cover"])

            if line % 1 == 0.5 or threshold % 1 == 0.5:
                actual_cover = (y_diff > threshold).astype(int)
            else:
                mask = y_diff != threshold
                actual_cover = (y_diff[mask] > threshold).astype(int)
                p_cover = p_cover[mask]

            pred_cover = (p_cover > 0.5).astype(int)
            metrics[f"spread_{line}_acc"] = float(np.mean(pred_cover == actual_cover))

            p_clip = np.clip(p_cover, 1e-7, 1 - 1e-7)
            metrics[f"spread_{line}_logloss"] = float(-np.mean(
                actual_cover * np.log(p_clip) + (1 - actual_cover) * np.log(1 - p_clip)
            ))

        return metrics

    @staticmethod
    def _ou_probability(lambda_total, line):
        """P(over), P(under), P(push) via Poisson CDF."""
        if line % 1 == 0.5:
            k = int(line - 0.5)
            p_under = poisson.cdf(k, lambda_total)
            return {"over": 1 - p_under, "under": p_under, "push": 0.0}
        else:
            k = int(line)
            p_under = poisson.cdf(k - 1, lambda_total)
            p_push  = poisson.pmf(k, lambda_total)
            return {"over": 1 - p_under - p_push, "under": p_under, "push": p_push}

    def _save_poisson_models(self, model_gf, model_ga) -> None:
        save_dir = self._model_dir("poisson")
        os.makedirs(save_dir, exist_ok=True)
        path_gf = os.path.join(save_dir, "poisson_goalsFor.json")
        path_ga = os.path.join(save_dir, "poisson_goalsAgainst.json")
        model_gf.save_model(path_gf)
        model_ga.save_model(path_ga)
        logger.info("Saved Poisson models → %s, %s", path_gf, path_ga)

    # ══════════════════════════════════════════════════════════════════════
    # Spread Regression (optional comparison — no longer used for inference)
    # ══════════════════════════════════════════════════════════════════════

    def preprocess_spread_reg(self, df: pd.DataFrame):
        """Prepare data for the optional spread regression model."""
        feature_cols = self._load_feature_cols(df)
        df = df[df["goalDiffFor"] != 0].copy()
        subset = df[feature_cols + ["goalDiffFor"]].copy()
        subset[feature_cols] = subset[feature_cols].fillna(0)
        subset = subset.dropna(subset=["goalDiffFor"])
        X      = subset[feature_cols]
        y_diff = subset["goalDiffFor"]
        logger.info("Spread reg: %d rows × %d features", *X.shape)
        return X, y_diff

    def train_spread_reg(
        self, X: pd.DataFrame, y_diff: pd.Series, n_iter: int = 10,
    ) -> tuple[xgb.Booster, float, dict]:
        """
        Walk-forward training for goal differential regression.
        OPTIONAL — kept for comparison against Skellam-based spreads.
        Inference no longer requires this model.
        """
        tscv   = TimeSeriesSplit(n_splits=n_iter)
        params = {k: v for k, v in self.params["spread_reg"].items()
                  if k not in ("n_estimators", "use_label_encoder")}

        final_model = None
        final_sigma = None
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
            tqdm(tscv.split(X), total=n_iter, desc="Training [spread_reg]")
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_diff.iloc[train_idx], y_diff.iloc[test_idx]

            top_features = self._select_features_regression_sq(X_train, y_train)
            X_tr = X_train[top_features]
            X_te = X_test[top_features]

            dtrain = xgb.DMatrix(X_tr, label=y_train)
            dtest  = xgb.DMatrix(X_te, label=y_test)

            model = xgb.train(
                params, dtrain,
                num_boost_round=MAX_BOOST_ROUND,
                evals=[(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )

            train_preds = model.predict(dtrain)
            sigma = float(np.std(y_train.values - train_preds))
            pred_diff = model.predict(dtest)

            fold_eval = self._evaluate_spread_reg_fold(pred_diff, y_test.values, sigma)
            fold_metrics.append(fold_eval)

            logger.info(
                "[spread_reg] fold %d  rounds=%d  sigma=%.3f  "
                "line_1.5_acc=%.3f  line_-1.5_acc=%.3f",
                fold, model.best_iteration, sigma,
                fold_eval.get("line_1.5_acc", 0),
                fold_eval.get("line_-1.5_acc", 0),
            )

            final_model = model
            final_sigma = sigma

        self._save_spread_reg_model(final_model, final_sigma)
        self._save_selected_features(top_features, "spread_reg")

        avg_metrics = self._aggregate_metrics(fold_metrics)
        logger.info("[spread_reg] CV Results (COMPARISON ONLY — not used for inference):")
        for key, val in avg_metrics.items():
            logger.info("  %s: %.4f", key, val)

        return final_model, final_sigma, avg_metrics

    def _evaluate_spread_reg_fold(self, pred_diff, y_actual, sigma) -> dict:
        """Evaluate spread_reg fold using normal CDF (for comparison)."""
        metrics = {
            "mse":  float(mean_squared_error(y_actual, pred_diff)),
            "mae":  float(mean_absolute_error(y_actual, pred_diff)),
            "rmse": float(np.sqrt(mean_squared_error(y_actual, pred_diff))),
            "sigma": sigma,
        }
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            threshold = -line
            p_cover = 1 - norm.cdf(threshold, loc=pred_diff, scale=sigma)

            if threshold % 1 == 0.5 or line % 1 == 0.5:
                actual_cover = (y_actual > threshold).astype(int)
            else:
                mask = y_actual != threshold
                actual_cover = (y_actual[mask] > threshold).astype(int)
                p_cover = p_cover[mask]

            pred_cover = (np.asarray(p_cover) > 0.5).astype(int)
            metrics[f"line_{line}_acc"] = float(np.mean(pred_cover == actual_cover))

        return metrics

    def _save_spread_reg_model(self, model, sigma) -> None:
        save_dir = self._model_dir("spread_reg")
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, "spread_reg_goalDiff.json"))
        _save_json({"sigma": sigma}, os.path.join(save_dir, "spread_reg_meta.json"))
        logger.info("Saved spread reg model (sigma=%.3f)", sigma)

    # ══════════════════════════════════════════════════════════════════════
    # Hyperparameter tuning
    # ══════════════════════════════════════════════════════════════════════

    def tune_hyperparameters(
        self, event, X, y, value=None, param_dist=None, n_iter=100,
    ) -> xgb.XGBClassifier:
        if param_dist is None:
            param_dist = {
                "n_estimators":     randint(100, 500),
                "max_depth":        [2, 3, 4, 5],
                "learning_rate":    uniform(0.01, 0.09),
                "reg_alpha":        uniform(0, 1),
                "reg_lambda":       uniform(0, 1),
                "subsample":        uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5),
                "min_child_weight": [1, 3, 5, 7],
            }

        tscv = TimeSeriesSplit(n_splits=5)
        split = list(tscv.split(X))
        train_idx, _ = split[-1]
        top_features = self._select_features(X.iloc[train_idx], y.iloc[train_idx])
        X_sel = X[top_features]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=_class_weight(y),
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        search = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter, cv=tscv,
            scoring=["neg_log_loss", "roc_auc", "balanced_accuracy"],
            refit="balanced_accuracy", verbose=2, n_jobs=-1,
            random_state=42, return_train_score=True,
        )
        search.fit(X_sel, y)

        best_params = search.best_params_
        best_params["objective"] = "binary:logistic"
        self.params[event].update(best_params)

        _save_json(best_params, self._params_path(event, value))

        score = round(100 * search.best_score_, 2)
        logger.info("[%s] Tuned CV balanced_accuracy: %.2f%%", event, score)

        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        search.best_estimator_.save_model(os.path.join(save_dir, f"XGBoost_tuned_{score:.2f}.json"))
        return search.best_estimator_

    # ══════════════════════════════════════════════════════════════════════
    # Calibration
    # ══════════════════════════════════════════════════════════════════════

    def calibrate(self, event, X, y, value=None, method="isotonic"):
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        base_params = {k: v for k, v in self.params[event].items()
                       if k not in ("eval_metric", "n_estimators")}
        base = xgb.XGBClassifier(
            **base_params, n_estimators=300,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        cal = CalibratedClassifierCV(base, method=method, cv=tscv)
        cal.fit(X, y)

        import joblib
        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(cal, os.path.join(save_dir, f"calibrated_{event}.pkl"))
        logger.info("[%s] Calibrated model saved", event)
        return cal

    # ══════════════════════════════════════════════════════════════════════
    # Evaluation + visualization
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(self, y_true, y_pred, y_pred_proba=None) -> dict:
        metrics = {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "ROC AUC":   roc_auc_score(y_true, y_pred),
        }
        if y_pred_proba is not None:
            try: metrics["Log Loss"] = log_loss(y_true, y_pred_proba)
            except: pass
            try: metrics["ROC AUC"] = roc_auc_score(y_true, y_pred_proba)
            except: pass

        for k, v in metrics.items():
            logger.info("%s: %s", k, v)
        return metrics

    def report_feature_importance(self, model, feature_names, top_n=20):
        try:
            importance = model.get_score(importance_type="gain")
            imp_df = pd.DataFrame(
                [(k, v) for k, v in importance.items()],
                columns=["feature", "gain"]
            ).sort_values("gain", ascending=False)
            logger.info("Top %d features by gain:\n%s", top_n, imp_df.head(top_n).to_string())
            return imp_df
        except:
            logger.warning("Feature importance report failed", exc_info=True)
            return None

    # ══════════════════════════════════════════════════════════════════════
    # Metric aggregation
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _aggregate_metrics(fold_metrics: list[dict]) -> dict:
        """Average all numeric metrics across folds."""
        all_keys = fold_metrics[0].keys()
        avg = {}
        for key in all_keys:
            vals = [m[key] for m in fold_metrics if key in m]
            avg[f"mean_{key}"] = float(np.mean(vals))
            if len(vals) > 1:
                avg[f"std_{key}"] = float(np.std(vals))
        return avg

    # ══════════════════════════════════════════════════════════════════════
    # Pipeline
    # ══════════════════════════════════════════════════════════════════════

    def run_pipeline(
        self,
        event:     str,
        value=     None,
        n_iter:    int  = 10,
        tune:      bool = False,
        calibrate: bool = True,
    ) -> tuple:
        df = self.load_data()

        if event == "poisson":
            X, y_gf, y_ga, df_full = self.preprocess_poisson(df)
            model_gf, model_ga, metrics = self.train_poisson(
                X, y_gf, y_ga, df_full=df_full, n_iter=n_iter
            )
            return model_gf, model_ga, metrics

        if event == "spread_reg":
            X, y_diff = self.preprocess_spread_reg(df)
            model, sigma, metrics = self.train_spread_reg(X, y_diff, n_iter=n_iter)
            return model, sigma, metrics

        X, y = self.preprocess(event, df, value)

        if tune:
            self.tune_hyperparameters(event, X, y, value=value)

        model, mean_acc = self.train_event(event, X, y, value=value, n_iter=n_iter)
        self.report_feature_importance(model, X.columns.tolist())

        if calibrate:
            self.calibrate(event, X, y, value=value)

        return model, mean_acc


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    trainer = NHLModelTrainer(None, None)

    # Primary: train Poisson models → evaluates ML + O/U + Spread
    trainer.run_pipeline("poisson", n_iter=60)

    # Optional: train standalone ML classifier for comparison
    # trainer.run_pipeline("ml", tune=True)

    # Optional: train spread_reg for comparison against Skellam
    # trainer.run_pipeline("spread_reg", n_iter=20)
