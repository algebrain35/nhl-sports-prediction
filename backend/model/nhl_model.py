import os
import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import sys

sys.path.insert(1, "./backend/preprocess")
import elo

from scipy.stats import poisson, norm

# ── Constants (must match preprocessor) ───────────────────────────────────────

EMA_SPANS = [5, 13]

DIFFERENTIAL_METRICS = [
    "xOnGoal", "xGoals", "xRebounds", "xFreeze",
    "xPlayStopped", "xPlayContinuedInZone", "xPlayContinuedOutsideZone",
    "flurryAdjustedxGoals", "scoreVenueAdjustedxGoals",
    "flurryScoreVenueAdjustedxGoals",
    "shotsOnGoal", "missedShots", "blockedShotAttempts", "shotAttempts",
    "rebounds", "reboundGoals", "freeze",
    "playStopped", "playContinuedInZone", "playContinuedOutsideZone",
    "savedShotsOnGoal", "savedUnblockedShotAttempts",
    "penalties", "penalityMinutes", "faceOffsWon",
    "hits", "takeaways", "giveaways",
    "lowDangerShots", "mediumDangerShots", "highDangerShots",
    "lowDangerxGoals", "mediumDangerxGoals", "highDangerxGoals",
    "lowDangerGoals", "mediumDangerGoals", "highDangerGoals",
    "scoreAdjustedShotsAttempts", "unblockedShotAttempts",
    "scoreAdjustedUnblockedShotAttempts",
    "dZoneGiveaways",
    "xGoalsFromxReboundsOfShots", "xGoalsFromActualReboundsOfShots",
    "reboundxGoals", "totalShotCredit",
    "scoreAdjustedTotalShotCredit", "scoreFlurryAdjustedTotalShotCredit",
    "goals",
]

MOMENTUM_METRICS = [
    "xGoals", "highDangerxGoals", "goals", "shotsOnGoal",
    "scoreVenueAdjustedxGoals", "corsiPercentage", "fenwickPercentage",
    "xGoalsPercentage", "winRateFor",
]

# Model filenames (stable, not accuracy-stamped)
GF_POISSON_FILE = "poisson_goalsFor.json"
GA_POISSON_FILE = "poisson_goalsAgainst.json"
SPREAD_REG_FILE = "spread_reg_goalDiff.json"
SPREAD_META_FILE = "spread_reg_meta.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_objective(booster: xgb.Booster) -> str:
    return json.loads(booster.save_config())["learner"]["objective"]["name"]


def _load_booster(path: str) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(path)
    return model


def _best_scored_file(directory: str) -> str:
    """Find the file with the highest numeric score in its filename."""
    scores = []
    for f in os.listdir(directory):
        match = re.search(r"\d+\.\d+", f)
        if match:
            scores.append((f, float(match.group())))
    if not scores:
        raise ValueError(f"No scored models in: {directory}")
    best, _ = max(scores, key=lambda x: x[1])
    return os.path.join(directory, best)


def best_model_path(event: str, model_dir: str) -> str | tuple[str, str]:
    """
    Resolve model path(s) for a given event type.
      ml     → str  (path to best accuracy-stamped .json)
      ou     → tuple[str, str]  (goalsFor path, goalsAgainst path)
      spread → str  (path to spread_reg_goalDiff.json)
    """
    try:
        if event == "ml":
            ml_dir = os.path.join(model_dir, "ml")
            if not os.path.exists(ml_dir):
                raise FileNotFoundError(f"Directory not found: {ml_dir}")
            return _best_scored_file(ml_dir)

        elif event == "ou":
            p_dir = os.path.join(model_dir, "poisson")
            if not os.path.exists(p_dir):
                raise FileNotFoundError(f"Directory not found: {p_dir}")
            return (
                os.path.join(p_dir, GF_POISSON_FILE),
                os.path.join(p_dir, GA_POISSON_FILE),
            )

        elif event == "spread":
            s_dir = os.path.join(model_dir, "spread_reg")
            if not os.path.exists(s_dir):
                raise FileNotFoundError(f"Directory not found: {s_dir}")
            return os.path.join(s_dir, SPREAD_REG_FILE)

        else:
            raise ValueError(f"Unknown event: {event}")

    except Exception as e:
        print(f"Error finding model path for '{event}': {e}")
        return ""


# ── Probability helpers ───────────────────────────────────────────────────────

def ou_probability(lambda_total, line):
    """
    P(over), P(under), P(push) for a given O/U line from Poisson λ.
    Works with both scalar and array inputs.
    """
    if line % 1 == 0.5:
        k = int(line - 0.5)
        p_under = poisson.cdf(k, lambda_total)
        return {"over": 1 - p_under, "under": p_under, "push": 0.0}
    else:
        k = int(line)
        p_under = poisson.cdf(k - 1, lambda_total)
        p_push  = poisson.pmf(k, lambda_total)
        return {"over": 1 - p_under - p_push, "under": p_under, "push": p_push}


def spread_probability(pred_diff, sigma, line):
    """
    P(home covers spread) using normal CDF on predicted goal differential.
    line: home spread (e.g., -1.5 means home favored by 1.5)
    """
    threshold = -line
    if threshold % 1 == 0.5 or line % 1 == 0.5:
        p_cover = 1 - norm.cdf(threshold, loc=pred_diff, scale=sigma)
        return {"cover": p_cover, "not_cover": 1 - p_cover, "push": 0.0}
    else:
        p_push = (
            norm.cdf(threshold + 0.5, loc=pred_diff, scale=sigma) -
            norm.cdf(threshold - 0.5, loc=pred_diff, scale=sigma)
        )
        p_not_cover = norm.cdf(threshold - 0.5, loc=pred_diff, scale=sigma)
        return {"cover": 1 - p_not_cover - p_push, "not_cover": p_not_cover, "push": p_push}


# ── Feature engineering (inference) ───────────────────────────────────────────

def engineer_match_features(row: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features for a single inference row.
    Must stay in sync with Preprocessor.engineer_features().
    """
    df = row.copy()

    # is_home
    if "home_or_away" in df.columns:
        df["is_home"] = (df["home_or_away"] == "HOME").astype(int)
    else:
        df["is_home"] = 1

    # PDO from lagged EMAs
    gf_col  = "goalsForPerGame_seasonal_ema_span_5"
    ga_col  = "goalsAgainstPerGame_seasonal_ema_span_5"
    sog_for = "shotsOnGoalFor_seasonal_ema_span_5"
    sog_ag  = "shotsOnGoalAgainst_seasonal_ema_span_5"
    if all(c in df.columns for c in [gf_col, ga_col, sog_for, sog_ag]):
        shooting_pct = df[gf_col] / df[sog_for].replace(0, np.nan)
        save_pct     = 1 - (df[ga_col] / df[sog_ag].replace(0, np.nan))
        df["pdo"] = shooting_pct + save_pct

    # For−Against differentials
    for metric in DIFFERENTIAL_METRICS:
        for span in EMA_SPANS:
            for_col = f"{metric}For_seasonal_ema_span_{span}"
            ag_col  = f"{metric}Against_seasonal_ema_span_{span}"
            if for_col in df.columns and ag_col in df.columns:
                df[f"{metric}_diff_ema_{span}"] = df[for_col] - df[ag_col]

    # Momentum: EMA_5 − EMA_13
    for metric in MOMENTUM_METRICS:
        col_5  = f"{metric}_diff_ema_5"
        col_13 = f"{metric}_diff_ema_13"
        if col_5 in df.columns and col_13 in df.columns:
            df[f"{metric}_momentum"] = df[col_5] - df[col_13]
            continue
        col_5  = f"{metric}_seasonal_ema_span_5"
        col_13 = f"{metric}_seasonal_ema_span_13"
        if col_5 in df.columns and col_13 in df.columns:
            df[f"{metric}_momentum"] = df[col_5] - df[col_13]

    # Elo differential + expected
    if "eloFor" in df.columns and "eloAgainst" in df.columns:
        df["elo_diff"] = df["eloFor"] - df["eloAgainst"]
        df["eloExpectedFor"] = 1 / (1 + 10 ** ((df["eloAgainst"] - df["eloFor"]) / 400))
        df["eloExpectedAgainst"] = 1 - df["eloExpectedFor"]

    # Days rest / back-to-back defaults for inference
    if "days_rest" not in df.columns:
        df["days_rest"] = 2.0
    if "is_back_to_back" not in df.columns:
        df["is_back_to_back"] = 0

    return df


# ── NHLModel ──────────────────────────────────────────────────────────────────

class NHLModel:
    """
    Unified inference model for three event types:

      ml:     XGBoost classifier          → P(home win)
      ou:     Dual Poisson (GF + GA)      → P(over/under) at any line
      spread: Regression on goalDiffFor   → P(cover) at any spread line

    Init patterns (matches app.py usage):
      NHLModel("ml",     model_path=best_model_path("ml", MODEL_DIR))
      NHLModel("ou",     model_paths=best_model_path("ou", MODEL_DIR))
      NHLModel("spread", model_path=best_model_path("spread", MODEL_DIR))
    """

    def __init__(self, event: str, model_path=None, model_paths=None):
        self.event = event.lower()

        # ML classifier
        self.model = None

        # Poisson dual models
        self.poisson_gf = None
        self.poisson_ga = None

        # Spread regression + residual sigma
        self.spread_model = None
        self.spread_sigma = None

        if self.event == "ml":
            if model_path is None:
                raise ValueError("ml requires model_path")
            self.model = _load_booster(model_path)
            print(f"[ml] Loaded: {model_path}")

        elif self.event == "ou":
            if model_paths is None:
                raise ValueError("ou requires model_paths=(gf_path, ga_path)")
            gf_path, ga_path = model_paths
            self.poisson_gf = _load_booster(gf_path)
            self.poisson_ga = _load_booster(ga_path)
            print(f"[ou] Loaded Poisson: {gf_path}, {ga_path}")

        elif self.event == "spread":
            if model_path is None:
                raise ValueError("spread requires model_path")
            self.spread_model = _load_booster(model_path)
            # Load sigma from meta file next to the model
            meta_path = os.path.join(os.path.dirname(model_path), SPREAD_META_FILE)
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                self.spread_sigma = meta["sigma"]
            else:
                self.spread_sigma = 2.5  # reasonable NHL default
                print(f"[spread] Warning: {meta_path} not found, using sigma={self.spread_sigma}")
            print(f"[spread] Loaded: {model_path} (sigma={self.spread_sigma:.3f})")

        else:
            raise ValueError(f"Unknown event: {self.event}")

    # ── DMatrix creation ──────────────────────────────────────────────

    @staticmethod
    def _to_dmatrix(booster: xgb.Booster, match_df: pd.DataFrame) -> xgb.DMatrix:
        """Create DMatrix using only features the booster expects."""
        features = booster.feature_names
        missing = [f for f in features if f not in match_df.columns]
        if missing:
            print(f"Warning: filling {len(missing)} missing features with 0")
            for f in missing:
                match_df[f] = 0.0
        mat = match_df[features].apply(pd.to_numeric, errors="coerce").astype(float)
        return xgb.DMatrix(mat)

    # ── Prediction dispatch ───────────────────────────────────────────

    def predict(self, match_df: pd.DataFrame, threshold=None) -> np.ndarray:
        """
        Unified predict method.
          ml:     → [[P(away), P(home)]]
          ou:     → [[P(under), P(over)]]       (threshold = O/U line)
          spread: → [[P(not_cover), P(cover)]]   (threshold = spread line)
        """
        if self.event == "ml":
            dmat = self._to_dmatrix(self.model, match_df)
            pred = self.model.predict(dmat)
            return np.stack([1 - pred, pred], axis=1)

        if threshold is None:
            raise ValueError(f"{self.event} prediction requires a threshold")

        if self.event == "ou":
            dmat_gf = self._to_dmatrix(self.poisson_gf, match_df)
            dmat_ga = self._to_dmatrix(self.poisson_ga, match_df)
            lambda_total = self.poisson_gf.predict(dmat_gf) + self.poisson_ga.predict(dmat_ga)
            probs = ou_probability(lambda_total, float(threshold))
            return np.stack([probs["under"], probs["over"]], axis=1)

        if self.event == "spread":
            dmat = self._to_dmatrix(self.spread_model, match_df)
            pred_diff = self.spread_model.predict(dmat)
            probs = spread_probability(pred_diff, self.spread_sigma, float(threshold))
            return np.stack([probs["not_cover"], probs["cover"]], axis=1)

    # ── Match construction ────────────────────────────────────────────

    def create_match(self, df: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
        """
        Build a single-row feature DataFrame for home vs away.
        Orients perspectives, merges, updates Elo, engineers features.
        """
        home_df = df[(df["team"] == home) | (df["opposingTeam"] == home)].tail(1).copy()
        away_df = df[(df["team"] == away) | (df["opposingTeam"] == away)].tail(1).copy()

        if home_df.empty or away_df.empty:
            raise ValueError(f"No data for: home={home}, away={away}")

        if home_df.iloc[0]["team"] != home:
            home_df = self._flip_perspective(home_df)
        if away_df.iloc[0]["team"] != away:
            away_df = self._flip_perspective(away_df)

        match_row = self._merge_perspectives(home_df, away_df)
        self._update_elo(match_row, home_df, away_df)
        match_row["home_or_away"] = "HOME"
        return engineer_match_features(match_row)

    @staticmethod
    def _flip_perspective(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for col in df.columns:
            if "Against" in col:
                rename[col] = col.replace("Against", "For")
            elif "For" in col:
                rename[col] = col.replace("For", "Against")
        return df.rename(columns=rename)

    @staticmethod
    def _merge_perspectives(home_df: pd.DataFrame, away_df: pd.DataFrame) -> pd.DataFrame:
        row = home_df.iloc[[0]].copy()
        for col in away_df.columns:
            if "For" in col and "_seasonal_ema_" in col:
                ag = col.replace("For", "Against")
                if ag in row.columns:
                    row[ag] = away_df.iloc[0][col]
        if "eloFor" in away_df.columns:
            row["eloAgainst"] = away_df.iloc[0]["eloFor"]
        return row

    @staticmethod
    def _update_elo(match_row, home_df, away_df):
        try:
            scorer = elo.Elo(50, 0.05)
            h_team = home_df.iloc[0]["team"]
            a_team = away_df.iloc[0]["team"]
            scorer[h_team] = home_df.iloc[0].get("eloFor", 1500)
            scorer[a_team] = away_df.iloc[0].get("eloFor", 1500)

            for tdf in [home_df, away_df]:
                r = tdf.iloc[0]
                margin = scorer.get_margin_factor(
                    r.get("goalsFor", 0) - r.get("goalsAgainst", 0)
                )
                w = r["team"] if r.get("winner", 0) == 1.0 else r.get("opposingTeam", "")
                l = r.get("opposingTeam", "") if r.get("winner", 0) == 1.0 else r["team"]
                if w and l and w in scorer.ratings and l in scorer.ratings:
                    inf = scorer.get_inflation_factor(scorer[w], scorer[l])
                    scorer.update_ratings(w, l, 50, margin, inf)

            match_row["eloFor"]     = scorer[h_team]
            match_row["eloAgainst"] = scorer[a_team]
        except Exception as e:
            print(f"Elo update warning: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def get_team_prediction(self, home: str, away: str, threshold=None):
        """
        End-to-end: load CSV → build match → engineer features → predict.

          ml:     → [[P(away), P(home)]]
          ou:     → [[P(under), P(over)]]       threshold = O/U line (e.g. 5.5)
          spread: → [[P(not_cover), P(cover)]]   threshold = spread line (e.g. -1.5)
        """
        try:
            df = pd.read_csv("all_games_preproc.csv")
            match_df = self.create_match(df, home, away)
            return self.predict(match_df, threshold)
        except Exception as e:
            print(f"Prediction error ({self.event}): {e}")
            import traceback
            traceback.print_exc()
            return None


# ── Utility functions ─────────────────────────────────────────────────────────

def kelly_fraction(p: float, a: float, b: float) -> float:
    return p / a - (1 - p) / b


def kelly_criterion_result(bankroll: float, prob: float, ret: float) -> float:
    return bankroll * kelly_fraction(prob, 1.0, ret)


def american_to_decimal(odds: int):
    return (100 + -odds) / -odds if odds < 0 else (100 + odds) / 100


if __name__ == "__main__":
    MODEL_DIR = "./backend/model/models"

    ml = NHLModel("ml", model_path=best_model_path("ml", MODEL_DIR))
    print("ML:", ml.get_team_prediction("PIT", "FLA"))

    ou = NHLModel("ou", model_paths=best_model_path("ou", MODEL_DIR))
    print("OU 5.5:", ou.get_team_prediction("PIT", "FLA", threshold=5.5))

    sp = NHLModel("spread", model_path=best_model_path("spread", MODEL_DIR))
    print("Spread -1.5:", sp.get_team_prediction("PIT", "FLA", threshold=-1.5))
