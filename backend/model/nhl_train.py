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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, confusion_matrix, roc_auc_score, roc_curve,
)
from scipy.stats import uniform, randint, poisson, norm

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


def _save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _should_keep_span(col_name: str) -> bool:
    """Return True if column either has no span suffix or has a kept span."""
    for span in ("3", "5", "8", "13"):
        if f"_span_{span}" in col_name:
            return span in KEEP_SPANS
    return True  # no span suffix → keep


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
            # Remove sklearn-only keys that break native xgb.train()
            self.params[event].pop("n_estimators", None)
            self.params[event].pop("use_label_encoder", None)
            logger.info("Loaded saved params for %s", event)

    # ── Feature selection (inside-split, no leakage) ──────────────────────

    @staticmethod
    def _select_features(X: pd.DataFrame, y: pd.Series, n: int = N_FEATURES) -> list[str]:
        """
        Fit a fast XGB on the training split only and return top-n feature names.
        """
        n = min(n, X.shape[1])  # can't select more features than exist
        selector = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0,
        )
        selector.fit(X, y)
        imp = pd.Series(selector.feature_importances_, index=X.columns)
        return imp.nlargest(n).index.tolist()

    # ── Preprocessing ─────────────────────────────────────────────────────

    def preprocess(self, event: str, df: pd.DataFrame, value=None, team=None):
        """
        Consumes the already-engineered DataFrame from the preprocessor.
        Loads the saved feature list (or falls back to whitelist selection).
        Only handles target creation and final filtering here.
        """
        if event not in EVENTS:
            raise ValueError(f"event must be one of {EVENTS}")

        self.load_params(event, value)

        # ── Shared cleaning ──────────────────────────────────────────────
        df = df.dropna(axis=1, how="all")
        df = df.drop(
            columns=df.columns[df.columns.str.contains("winner_seasonal_ema_span")],
            errors="ignore",
        )
        df = df[df["goalDiffFor"] != 0]

        if team:
            df = df[df["team"] == team]

        # ── Load feature list from preprocessor ──────────────────────────
        feature_path = self.data_path.replace(".csv", "_features.json")
        feature_cols = None
        if os.path.exists(feature_path):
            import json
            with open(feature_path, "r") as f:
                feature_cols = json.load(f)
            # Only keep columns that exist in this df
            feature_cols = [c for c in feature_cols if c in df.columns]
            logger.info("Loaded %d feature columns from %s", len(feature_cols), feature_path)
        else:
            # Fallback: whitelist-based selection
            logger.warning("Feature list not found at %s — using fallback selection", feature_path)
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

        # ── Target variable ──────────────────────────────────────────────
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

        # Drop any columns that are all NaN
        X = X.dropna(axis=1, how="all")
        y = y.loc[X.index]

        logger.info("Final feature matrix: %d rows × %d cols", *X.shape)
        return X, y

    # ── Core training (walk-forward) ──────────────────────────────────────

    def _train(
        self,
        event:    str,
        X:        pd.DataFrame,
        y:        pd.Series,
        value=    None,
        n_iter:   int = 10,
    ) -> tuple[xgb.Booster, float]:
        """
        Walk-forward training over n_iter TimeSeriesSplit folds.
        Feature selection runs inside each fold to prevent leakage.
        Returns (final_fold_model, mean_cv_accuracy).

        Key changes from v1:
        - Uses eval-set early stopping instead of nested xgb.cv
        - Always returns the LAST fold's model (trained on most data)
        - Consistent num_boost_round handling
        """
        tscv        = TimeSeriesSplit(n_splits=n_iter)
        params      = {k: v for k, v in self.params[event].items()
                       if k not in ("n_estimators", "use_label_encoder")}
        acc_results = []
        final_model = None

        for fold, (train_idx, test_idx) in enumerate(
            tqdm(tscv.split(X), total=n_iter, desc=f"Training [{event}]")
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Feature selection on train split only
            top_features = self._select_features(X_train, y_train)
            X_tr = X_train[top_features]
            X_te = X_test[top_features]

            weight = _class_weight(y_train)
            dtrain = xgb.DMatrix(X_tr, label=y_train)
            dtest  = xgb.DMatrix(X_te, label=y_test)

            # Single early-stopping pass — no nested CV
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

            # Always keep the latest (largest training set) fold model
            final_model = model

        # Save the final model
        final_acc = acc_results[-1]
        self._save_model(final_model, event, final_acc, value)

        # Evaluate on last fold
        self.evaluate(y_test, y_pred)

        mean_acc = float(np.mean(acc_results))
        logger.info(
            "[%s] CV accuracy: %.2f%% ± %.2f%%  (last fold: %.2f%%)",
            event, mean_acc, float(np.std(acc_results)), final_acc,
        )

        # Store selected features from last fold for inference
        self._save_selected_features(top_features, event, value)

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

    # ── Public training API ───────────────────────────────────────────────

    def train_ml(self, X, y, n_iter: int = 10, team=None):
        return self._train("ml", X, y, n_iter=n_iter)

    def train_spread(self, X, y, spread, n_iter: int = 10):
        return self._train("spread", X, y, value=spread, n_iter=n_iter)

    def train_ou(self, X, y, ou, n_iter: int = 10):
        return self._train("ou", X, y, value=ou, n_iter=n_iter)

    def train_event(self, event: str, X, y, value=None, n_iter: int = 10):
        if event == "ml":
            return self.train_ml(X, y, n_iter=n_iter)
        if event == "spread":
            return self.train_spread(X, y, value, n_iter=n_iter)
        if event == "ou":
            return self.train_ou(X, y, value, n_iter=n_iter)
        raise ValueError(f"Unknown event: {event}")

    # ── Hyperparameter tuning ─────────────────────────────────────────────

    def tune_hyperparameters(
        self,
        event:      str,
        X:          pd.DataFrame,
        y:          pd.Series,
        value=      None,
        param_dist= None,
        n_iter:     int = 100,
    ) -> xgb.XGBClassifier:
        if param_dist is None:
            param_dist = {
                "n_estimators":    randint(100, 500),
                "max_depth":       [2, 3, 4, 5],
                "learning_rate":   uniform(0.01, 0.09),
                "reg_alpha":       uniform(0, 1),
                "reg_lambda":      uniform(0, 1),
                "subsample":       uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5),  # added: helps with correlated features
                "min_child_weight": [1, 3, 5, 7],
            }

        tscv = TimeSeriesSplit(n_splits=5)

        # Feature selection on last fold's train split
        split = list(tscv.split(X))
        train_idx, _ = split[-1]
        top_features = self._select_features(X.iloc[train_idx], y.iloc[train_idx])
        X_sel = X[top_features]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=_class_weight(y),
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring=["neg_log_loss", "roc_auc", "balanced_accuracy"],
            refit="balanced_accuracy",
            verbose=2,
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
        )
        search.fit(X_sel, y)

        best_params = search.best_params_
        best_params["objective"] = "binary:logistic"
        self.params[event].update(best_params)
        logger.info("[%s] Best params: %s", event, json.dumps(best_params, indent=2))

        _save_json(best_params, self._params_path(event, value))

        # Report CV score only — no misleading full-set accuracy
        score = round(100 * search.best_score_, 2)
        logger.info("[%s] Tuned CV score (balanced_accuracy): %.2f%%", event, score)

        # Also report CV log loss for calibration insight
        cv_results = pd.DataFrame(search.cv_results_)
        best_idx = search.best_index_
        cv_logloss = -cv_results.loc[best_idx, "mean_test_neg_log_loss"]
        cv_auc = cv_results.loc[best_idx, "mean_test_roc_auc"]
        logger.info("[%s] Tuned CV log_loss: %.4f, ROC AUC: %.4f", event, cv_logloss, cv_auc)

        best_model = search.best_estimator_
        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        best_model.save_model(os.path.join(save_dir, f"XGBoost_tuned_{score:.2f}.json"))

        return best_model

    # ── Calibration ───────────────────────────────────────────────────────

    def calibrate(
        self,
        event: str,
        X:     pd.DataFrame,
        y:     pd.Series,
        value= None,
        method: str = "isotonic",
    ) -> CalibratedClassifierCV:
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        base_params = {k: v for k, v in self.params[event].items()
                       if k not in ("eval_metric", "n_estimators")}
        base = xgb.XGBClassifier(
            **base_params,
            n_estimators=300,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        cal = CalibratedClassifierCV(base, method=method, cv=tscv)
        cal.fit(X, y)

        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        import joblib
        joblib.dump(cal, os.path.join(save_dir, f"calibrated_{event}.pkl"))
        logger.info("[%s] Calibrated model saved", event)
        return cal

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, y_true, y_pred, y_pred_proba=None) -> dict:
        metrics = {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "ROC AUC":   roc_auc_score(y_true, y_pred),
            "Log Loss":  None,
            "Confusion Matrix": confusion_matrix(y_true, y_pred),
        }

        if y_pred_proba is not None:
            try:
                metrics["Log Loss"] = log_loss(y_true, y_pred_proba)
            except Exception:
                logger.warning("Log loss computation failed", exc_info=True)
            try:
                metrics["ROC AUC"] = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                logger.warning("ROC AUC computation failed", exc_info=True)

        for k, v in metrics.items():
            if k == "Confusion Matrix":
                logger.info("%s:\n%s", k, v)
            else:
                logger.info("%s: %s", k, v)
        return metrics

    # ── Visualizations ────────────────────────────────────────────────────

    def save_visualizations(self, event: str, X, y, model, value=None) -> None:
        save_dir = self._model_dir(event, value)
        os.makedirs(save_dir, exist_ok=True)
        prefix = os.path.join(save_dir, f"{event}_{value}")

        # SHAP — skip if it causes memory issues (known xgb.Booster + SHAP bug)
        try:
            # Only run SHAP on a sample to avoid memory issues
            sample_size = min(500, len(X))
            X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
            explainer   = shap.Explainer(model)
            shap_values = explainer(X_sample)
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(f"{prefix}_shap.png", bbox_inches="tight", dpi=100)
            plt.close("all")
            logger.info("SHAP plot → %s_shap.png", prefix)
        except Exception:
            logger.warning("SHAP plot skipped (memory/compatibility issue)", exc_info=False)
            plt.close("all")

        try:
            y_score = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_score)
            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y, y_score):.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
            plt.savefig(f"{prefix}_roc.png", bbox_inches="tight", dpi=100)
            plt.close("all")
            logger.info("ROC plot → %s_roc.png", prefix)
        except Exception:
            logger.warning("ROC plot failed", exc_info=True)
            plt.close("all")

    # ── Feature importance report ─────────────────────────────────────────

    def report_feature_importance(self, model, feature_names: list[str], top_n: int = 20):
        """Log top features by importance from the trained booster."""
        try:
            importance = model.get_score(importance_type="gain")
            imp_df = pd.DataFrame(
                [(k, v) for k, v in importance.items()],
                columns=["feature", "gain"]
            ).sort_values("gain", ascending=False)
            logger.info("Top %d features by gain:\n%s", top_n, imp_df.head(top_n).to_string())
            return imp_df
        except Exception:
            logger.warning("Feature importance report failed", exc_info=True)
            return None

    # ── Poisson Regression ───────────────────────────────────────────────

    def preprocess_poisson(self, df: pd.DataFrame):
        """
        Prepare data for the dual Poisson model (goalsFor + goalsAgainst).
        Returns X, y_gf, y_ga with aligned indices.
        """
        # Load feature list
        feature_path = self.data_path.replace(".csv", "_features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                feature_cols = json.load(f)
            feature_cols = [c for c in feature_cols if c in df.columns]
            logger.info("Poisson: loaded %d features from %s", len(feature_cols), feature_path)
        else:
            raise FileNotFoundError(f"Feature list required: {feature_path}")

        subset = df[feature_cols + ["goalsFor", "goalsAgainst"]].copy()
        subset[feature_cols] = subset[feature_cols].fillna(0)
        subset = subset.dropna(subset=["goalsFor", "goalsAgainst"])

        X    = subset[feature_cols]
        y_gf = subset["goalsFor"]
        y_ga = subset["goalsAgainst"]

        logger.info("Poisson: %d rows × %d features", *X.shape)
        return X, y_gf, y_ga

    def train_poisson(
        self,
        X:      pd.DataFrame,
        y_gf:   pd.Series,
        y_ga:   pd.Series,
        n_iter: int = 10,
    ) -> tuple[xgb.Booster, xgb.Booster, dict]:
        """
        Walk-forward training for dual Poisson models (goalsFor & goalsAgainst).
        Returns (model_gf, model_ga, eval_metrics).
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
            X_train, X_test     = X.iloc[train_idx], X.iloc[test_idx]
            y_train_gf, y_test_gf = y_gf.iloc[train_idx], y_gf.iloc[test_idx]
            y_train_ga, y_test_ga = y_ga.iloc[train_idx], y_ga.iloc[test_idx]

            # Feature selection on train split
            # Use goalsFor as proxy target for feature ranking
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

            # ── Evaluate this fold ───────────────────────────────────
            lambda_gf = model_gf.predict(dtest_gf)
            lambda_ga = model_ga.predict(dtest_ga)
            lambda_total = lambda_gf + lambda_ga
            y_total_test = y_test_gf.values + y_test_ga.values

            fold_eval = self._evaluate_poisson_fold(
                lambda_total, y_total_test, lambda_gf, lambda_ga,
                y_test_gf.values, y_test_ga.values
            )
            fold_metrics.append(fold_eval)
            logger.info(
                "[poisson] fold %d  gf_rounds=%d  ga_rounds=%d  "
                "line_5.5_acc=%.3f  line_5.5_logloss=%.4f",
                fold, model_gf.best_iteration, model_ga.best_iteration,
                fold_eval.get("line_5.5_acc", 0),
                fold_eval.get("line_5.5_logloss", 0),
            )

            final_model_gf = model_gf
            final_model_ga = model_ga

        # ── Save models ──────────────────────────────────────────────
        self._save_poisson_models(final_model_gf, final_model_ga)
        self._save_selected_features(top_features, "poisson")

        # ── Aggregate metrics ────────────────────────────────────────
        avg_metrics = self._aggregate_poisson_metrics(fold_metrics)
        logger.info("[poisson] CV Results:")
        for key, val in avg_metrics.items():
            logger.info("  %s: %.4f", key, val)

        return final_model_gf, final_model_ga, avg_metrics

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
    def ou_probability(lambda_total, line):
        """
        Compute P(over), P(under), and P(push) for a given O/U line.
        Works with both scalar and array inputs.
        """
        if line % 1 == 0.5:
            # Half-line: no push possible
            k = int(line - 0.5)  # e.g., 5.5 → CDF at 5
            p_under = poisson.cdf(k, lambda_total)
            p_over  = 1 - p_under
            return {"over": p_over, "under": p_under, "push": 0.0}
        else:
            # Whole-line: push possible when total == line
            k = int(line)
            p_under = poisson.cdf(k - 1, lambda_total)  # P(X < line)
            p_push  = poisson.pmf(k, lambda_total)       # P(X == line)
            p_over  = 1 - p_under - p_push                # P(X > line)
            return {"over": p_over, "under": p_under, "push": p_push}

    def _evaluate_poisson_fold(
        self, lambda_total, y_total,
        lambda_gf, lambda_ga, y_gf, y_ga
    ) -> dict:
        """Evaluate a single fold across multiple O/U lines."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        metrics = {
            "mse_total":  float(mean_squared_error(y_total, lambda_total)),
            "mae_total":  float(mean_absolute_error(y_total, lambda_total)),
            "mse_gf":     float(mean_squared_error(y_gf, lambda_gf)),
            "mse_ga":     float(mean_squared_error(y_ga, lambda_ga)),
            "mean_lambda": float(np.mean(lambda_total)),
            "mean_actual": float(np.mean(y_total)),
        }

        # O/U accuracy and log loss at standard lines
        for line in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
            probs = self.ou_probability(lambda_total, line)
            p_over = probs["over"]

            if line % 1 == 0.5:
                actual_over = (y_total > line).astype(int)
            else:
                # Exclude pushes for accuracy calculation on whole lines
                mask = y_total != line
                actual_over = (y_total[mask] > line).astype(int)
                p_over = p_over[mask] if isinstance(p_over, np.ndarray) else p_over

            pred_over = (np.asarray(p_over) > 0.5).astype(int)
            acc = float(np.mean(pred_over == actual_over))

            # Log loss (clip to avoid log(0))
            p_clipped = np.clip(np.asarray(p_over), 1e-7, 1 - 1e-7)
            ll = float(-np.mean(
                actual_over * np.log(p_clipped) +
                (1 - actual_over) * np.log(1 - p_clipped)
            ))

            metrics[f"line_{line}_acc"] = acc
            metrics[f"line_{line}_logloss"] = ll
            metrics[f"line_{line}_pred_over_rate"] = float(np.mean(np.asarray(p_over) > 0.5))
            metrics[f"line_{line}_actual_over_rate"] = float(np.mean(actual_over))

        return metrics

    def _aggregate_poisson_metrics(self, fold_metrics: list[dict]) -> dict:
        """Average metrics across all folds."""
        all_keys = fold_metrics[0].keys()
        avg = {}
        for key in all_keys:
            vals = [m[key] for m in fold_metrics if key in m]
            avg[f"mean_{key}"] = float(np.mean(vals))
            if len(vals) > 1:
                avg[f"std_{key}"] = float(np.std(vals))
        return avg

    def _save_poisson_models(
        self, model_gf: xgb.Booster, model_ga: xgb.Booster
    ) -> None:
        save_dir = self._model_dir("poisson")
        os.makedirs(save_dir, exist_ok=True)

        path_gf = os.path.join(save_dir, "poisson_goalsFor.json")
        path_ga = os.path.join(save_dir, "poisson_goalsAgainst.json")
        model_gf.save_model(path_gf)
        model_ga.save_model(path_ga)
        logger.info("Saved Poisson models → %s, %s", path_gf, path_ga)

    # ── Spread Regression ─────────────────────────────────────────────────

    def preprocess_spread_reg(self, df: pd.DataFrame):
        """
        Prepare data for the spread regression model.
        Target = goalDiffFor (continuous, from home team perspective).
        Returns X, y_diff with aligned indices.
        """
        feature_path = self.data_path.replace(".csv", "_features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                feature_cols = json.load(f)
            feature_cols = [c for c in feature_cols if c in df.columns]
            logger.info("Spread reg: loaded %d features from %s", len(feature_cols), feature_path)
        else:
            raise FileNotFoundError(f"Feature list required: {feature_path}")

        # Filter out ties (goalDiffFor == 0) since NHL games go to OT
        df = df[df["goalDiffFor"] != 0].copy()

        subset = df[feature_cols + ["goalDiffFor"]].copy()
        subset[feature_cols] = subset[feature_cols].fillna(0)
        subset = subset.dropna(subset=["goalDiffFor"])

        X      = subset[feature_cols]
        y_diff = subset["goalDiffFor"]

        logger.info("Spread reg: %d rows × %d features", *X.shape)
        return X, y_diff

    def train_spread_reg(
        self,
        X:      pd.DataFrame,
        y_diff: pd.Series,
        n_iter: int = 10,
    ) -> tuple[xgb.Booster, float, dict]:
        """
        Walk-forward training for goal differential regression.
        One model, any spread line via normal CDF post-prediction.
        Returns (model, residual_sigma, eval_metrics).
        """
        tscv = TimeSeriesSplit(n_splits=n_iter)

        params = {k: v for k, v in self.params["spread_reg"].items()
                  if k not in ("n_estimators", "use_label_encoder")}

        final_model = None
        final_sigma = None
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
            tqdm(tscv.split(X), total=n_iter, desc="Training [spread_reg]")
        ):
            X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test   = y_diff.iloc[train_idx], y_diff.iloc[test_idx]

            # Feature selection on train split
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

            # Predict and compute residual std on train set
            train_preds = model.predict(dtrain)
            residuals = y_train.values - train_preds
            sigma = float(np.std(residuals))

            # Evaluate on test fold
            pred_diff = model.predict(dtest)

            fold_eval = self._evaluate_spread_fold(
                pred_diff, y_test.values, sigma
            )
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

        # Save model and sigma
        self._save_spread_reg_model(final_model, final_sigma)
        self._save_selected_features(top_features, "spread_reg")

        # Aggregate metrics
        avg_metrics = self._aggregate_spread_metrics(fold_metrics)
        logger.info("[spread_reg] CV Results:")
        for key, val in avg_metrics.items():
            logger.info("  %s: %.4f", key, val)

        return final_model, final_sigma, avg_metrics

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

    @staticmethod
    def spread_probability(pred_diff, sigma, line):
        """
        Compute P(home covers spread) using normal distribution.

        pred_diff: predicted goal differential (home - away)
        sigma:     residual std from training
        line:      spread line (e.g., -1.5 means home favored by 1.5)

        For home to cover a spread of -1.5, they need goalDiff > 1.5.
        The "line" from the user is the home spread, so:
          cover_threshold = -line  (e.g., spread=-1.5 → need diff > 1.5)

        Half-lines (1.5, 2.5): no push
        Whole-lines (1, 2): push possible
        """
        cover_threshold = -line

        if cover_threshold % 1 == 0.5 or line % 1 == 0.5:
            # Half-line: no push
            p_cover = 1 - norm.cdf(cover_threshold, loc=pred_diff, scale=sigma)
            return {"cover": p_cover, "not_cover": 1 - p_cover, "push": 0.0}
        else:
            # Whole-line: approximate push probability
            # P(diff == exactly threshold) ≈ density at that point × small window
            # For practical purposes in NHL, use a ±0.5 window around the integer
            p_push = (
                norm.cdf(cover_threshold + 0.5, loc=pred_diff, scale=sigma) -
                norm.cdf(cover_threshold - 0.5, loc=pred_diff, scale=sigma)
            )
            p_not_cover = norm.cdf(cover_threshold - 0.5, loc=pred_diff, scale=sigma)
            p_cover = 1 - p_not_cover - p_push
            return {"cover": p_cover, "not_cover": p_not_cover, "push": p_push}

    def _evaluate_spread_fold(self, pred_diff, y_actual, sigma) -> dict:
        """Evaluate a single fold across standard spread lines."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        metrics = {
            "mse":          float(mean_squared_error(y_actual, pred_diff)),
            "mae":          float(mean_absolute_error(y_actual, pred_diff)),
            "rmse":         float(np.sqrt(mean_squared_error(y_actual, pred_diff))),
            "sigma":        sigma,
            "mean_pred":    float(np.mean(pred_diff)),
            "mean_actual":  float(np.mean(y_actual)),
        }

        # Evaluate at standard NHL spread lines
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            cover_threshold = -line

            # Model's predicted P(cover)
            p_cover = 1 - norm.cdf(cover_threshold, loc=pred_diff, scale=sigma)

            # Actual outcomes
            if cover_threshold % 1 == 0.5 or line % 1 == 0.5:
                actual_cover = (y_actual > cover_threshold).astype(int)
            else:
                mask = y_actual != cover_threshold
                actual_cover = (y_actual[mask] > cover_threshold).astype(int)
                p_cover = p_cover[mask] if isinstance(p_cover, np.ndarray) else p_cover

            pred_cover = (np.asarray(p_cover) > 0.5).astype(int)
            acc = float(np.mean(pred_cover == actual_cover))

            # Log loss
            p_clipped = np.clip(np.asarray(p_cover), 1e-7, 1 - 1e-7)
            ll = float(-np.mean(
                actual_cover * np.log(p_clipped) +
                (1 - actual_cover) * np.log(1 - p_clipped)
            ))

            metrics[f"line_{line}_acc"] = acc
            metrics[f"line_{line}_logloss"] = ll
            metrics[f"line_{line}_pred_cover_rate"] = float(np.mean(np.asarray(p_cover) > 0.5))
            metrics[f"line_{line}_actual_cover_rate"] = float(np.mean(actual_cover))

        return metrics

    def _aggregate_spread_metrics(self, fold_metrics: list[dict]) -> dict:
        """Average spread metrics across all folds."""
        all_keys = fold_metrics[0].keys()
        avg = {}
        for key in all_keys:
            vals = [m[key] for m in fold_metrics if key in m]
            avg[f"mean_{key}"] = float(np.mean(vals))
            if len(vals) > 1:
                avg[f"std_{key}"] = float(np.std(vals))
        return avg

    def _save_spread_reg_model(self, model: xgb.Booster, sigma: float) -> None:
        save_dir = self._model_dir("spread_reg")
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "spread_reg_goalDiff.json")
        model.save_model(model_path)

        # Save sigma alongside the model — needed at inference time
        meta_path = os.path.join(save_dir, "spread_reg_meta.json")
        _save_json({"sigma": sigma}, meta_path)

        logger.info("Saved spread reg model → %s (sigma=%.3f)", model_path, sigma)

    # ── Pipeline ──────────────────────────────────────────────────────────

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
            X, y_gf, y_ga = self.preprocess_poisson(df)
            model_gf, model_ga, metrics = self.train_poisson(X, y_gf, y_ga, n_iter=n_iter)
            return model_gf, model_ga, metrics

        if event == "spread_reg":
            X, y_diff = self.preprocess_spread_reg(df)
            model, sigma, metrics = self.train_spread_reg(X, y_diff, n_iter=n_iter)
            return model, sigma, metrics

        X, y = self.preprocess(event, df, value)

        if tune:
            self.tune_hyperparameters(event, X, y, value=value)

        model, mean_acc = self.train_event(event, X, y, value=value, n_iter=n_iter)

        # Report what the model actually learned
        self.report_feature_importance(model, X.columns.tolist())

        if calibrate:
            self.calibrate(event, X, y, value=value)

        #self.save_visualizations(event, X, y, model, value=value)
        return model, mean_acc


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    trainer = NHLModelTrainer(None, None)

    trainer.run_pipeline("ml", tune=True)
    
    trainer.run_pipeline("poisson", n_iter=10)

    trainer.run_pipeline("spread_reg", n_iter=10)


