import os
import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import sys

sys.path.insert(1, "./backend/preprocess")
import elo

from scipy.stats import poisson
from scipy.special import ive as log_scaled_bessel  # exponentially-scaled Bessel

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

# Model filenames
GF_POISSON_FILE = "poisson_goalsFor.json"
GA_POISSON_FILE = "poisson_goalsAgainst.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

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

      ml     → str           (path to best accuracy-stamped .json classifier)
      ou     → (str, str)    (goalsFor path, goalsAgainst path)
      spread → (str, str)    (same Poisson models — spread is Skellam-derived)
    """
    try:
        if event == "ml":
            ml_dir = os.path.join(model_dir, "ml")
            if not os.path.exists(ml_dir):
                raise FileNotFoundError(f"Directory not found: {ml_dir}")
            return _best_scored_file(ml_dir)

        elif event in ("ou", "spread"):
            # Both O/U and spread are powered by the same Poisson pair
            p_dir = os.path.join(model_dir, "poisson")
            if not os.path.exists(p_dir):
                raise FileNotFoundError(f"Directory not found: {p_dir}")
            return (
                os.path.join(p_dir, GF_POISSON_FILE),
                os.path.join(p_dir, GA_POISSON_FILE),
            )

        else:
            raise ValueError(f"Unknown event: {event}")

    except Exception as e:
        print(f"Error finding model path for '{event}': {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Skellam Distribution
#
# The difference of two independent Poisson RVs X ~ Pois(μ₁), Y ~ Pois(μ₂)
# follows a Skellam distribution. The PMF uses the modified Bessel function
# of the first kind:
#
#   P(X - Y = k) = e^{-(μ₁+μ₂)} · (μ₁/μ₂)^{k/2} · I_{|k|}(2√(μ₁μ₂))
#
# This gives us EXACT discrete probabilities for goal differential —
# no normal approximation needed.
# ══════════════════════════════════════════════════════════════════════════════

def skellam_pmf(k, mu1, mu2):
    """
    P(X - Y = k) where X ~ Poisson(mu1), Y ~ Poisson(mu2).

    Uses the numerically stable form via exponentially-scaled Bessel:
      ive(v, z) = I_v(z) · e^{-|Re(z)|}
    so  I_v(z) = ive(v, z) · e^{z}

    log P(k) = -(mu1 + mu2)  +  (k/2) · ln(mu1/mu2)  +  ln(ive(|k|, z))  +  z
    where z = 2·sqrt(mu1·mu2)
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    z = 2.0 * np.sqrt(mu1 * mu2)
    abs_k = np.abs(k)

    # ive(v, z) = I_v(z) · e^{-z}  →  log I_v(z) = log(ive(v, z)) + z
    ive_val = log_scaled_bessel(abs_k, z)

    # Guard against ive returning 0 for large |k|
    ive_val = np.maximum(ive_val, 1e-300)

    log_pmf = (
        -(mu1 + mu2)
        + (k / 2.0) * np.log(np.maximum(mu1 / mu2, 1e-15))
        + np.log(ive_val)
        + z  # un-scale the exponentially-scaled Bessel
    )
    return np.exp(np.minimum(log_pmf, 0.0))  # clamp to avoid > 1 from float noise


def skellam_cdf(k, mu1, mu2, max_goals=12):
    """
    P(X - Y ≤ k) — CDF of Skellam distribution.
    Computed by summing PMF from -max_goals to floor(k).

    For NHL: goal differentials beyond ±12 have negligible probability.
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    k_floor = int(np.floor(k))

    # Vectorized sum over all diff values at once
    diffs = np.arange(-max_goals, k_floor + 1)
    total = np.zeros_like(mu1, dtype=np.float64)
    for d in diffs:
        total += skellam_pmf(d, mu1, mu2)

    return np.clip(total, 0.0, 1.0)


# ── O/U probabilities (Poisson on total goals — unchanged) ───────────────────

def ou_probability(lambda_total, line):
    """
    P(over), P(under), P(push) for a given O/U line.
    Uses Poisson CDF on total expected goals (λ_gf + λ_ga).
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


# ── Spread probabilities (Skellam on goal differential) ──────────────────────

def spread_probability(lambda_gf, lambda_ga, line):
    """
    P(home covers spread) using the Skellam distribution.

    This is the key improvement: instead of fitting a separate regression
    model and using a normal CDF approximation, we derive spread probabilities
    directly from the same Poisson λ values used for O/U.

    The Skellam distribution gives EXACT discrete probabilities for
    (goals_for - goals_against), which is inherently an integer.

    Args:
        lambda_gf: predicted expected goals for home team
        lambda_ga: predicted expected goals against home team
        line:      home spread (e.g., -1.5 means home favored by 1.5)

    Cover condition: goalDiff > -line  (home needs diff to exceed threshold)
    """
    threshold = -line  # e.g., spread=-1.5 → home needs diff > 1.5

    if line % 1 == 0.5 or threshold % 1 == 0.5:
        # Half-line: no push possible
        # P(cover) = P(diff > threshold) = 1 - P(diff ≤ floor(threshold))
        p_not_cover = skellam_cdf(threshold, lambda_gf, lambda_ga)
        p_cover = 1.0 - p_not_cover
        return {"cover": p_cover, "not_cover": p_not_cover, "push": 0.0}
    else:
        # Whole-line: push when diff == threshold exactly
        k = int(threshold)
        p_push = skellam_pmf(k, lambda_gf, lambda_ga)
        p_at_or_below = skellam_cdf(k, lambda_gf, lambda_ga)
        p_not_cover = p_at_or_below - p_push  # P(diff < threshold)
        p_cover = 1.0 - p_at_or_below          # P(diff > threshold)
        return {
            "cover":     np.clip(p_cover, 0.0, 1.0),
            "not_cover": np.clip(p_not_cover, 0.0, 1.0),
            "push":      np.clip(p_push, 0.0, 1.0),
        }


# ── ML probabilities (Skellam on goal differential) ──────────────────────────

def ml_probability_from_poisson(lambda_gf, lambda_ga):
    """
    Derive moneyline probabilities from Poisson λ values via Skellam.

      P(home win in regulation) = P(diff > 0) = 1 - P(diff ≤ 0)
      P(away win in regulation) = P(diff < 0) = P(diff ≤ -1)
      P(regulation draw → OT)   = P(diff == 0)

    NHL games that are tied after regulation go to OT/SO.
    Historical OT home win rate is ~52%, so we split draws accordingly.
    """
    OT_HOME_EDGE = 0.52

    p_draw    = skellam_pmf(0, lambda_gf, lambda_ga)
    p_home_reg = 1.0 - skellam_cdf(0, lambda_gf, lambda_ga)   # P(diff > 0)
    p_away_reg = skellam_cdf(-1, lambda_gf, lambda_ga)         # P(diff ≤ -1) = P(diff < 0)

    p_home = p_home_reg + p_draw * OT_HOME_EDGE
    p_away = p_away_reg + p_draw * (1.0 - OT_HOME_EDGE)

    return {"home": p_home, "away": p_away, "ot_probability": p_draw}


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


# ══════════════════════════════════════════════════════════════════════════════
# NHLModel — Unified inference
# ══════════════════════════════════════════════════════════════════════════════

class NHLModel:
    """
    Unified inference model for three betting markets.

    Architecture
    ────────────
    The dual Poisson models (goalsFor + goalsAgainst) are the SINGLE SOURCE
    OF TRUTH. They predict λ_gf and λ_ga, from which we derive:

      ML:     P(home win) via Skellam   →  P(diff > 0)
      O/U:    P(over/under) via Poisson →  CDF on λ_total = λ_gf + λ_ga
      Spread: P(cover) via Skellam      →  P(diff > threshold)

    The standalone ML classifier is kept as an optional alternative when
    initialized with model_path instead of model_paths.

    Why Skellam > normal approximation for spreads
    ───────────────────────────────────────────────
    NHL goal differentials are discrete integers. The Skellam distribution
    gives the EXACT probability mass for each integer value, while the normal
    approximation smears probability across a continuous range. This matters
    most at whole-number spread lines (±1, ±2) where the discrete push
    probability is significant (~18-22% for puck line at ±1).

    Init patterns
    ─────────────
      NHLModel("ml",     model_path=best_model_path("ml", DIR))      # classifier
      NHLModel("ml",     model_paths=best_model_path("ou", DIR))     # Poisson-derived ML
      NHLModel("ou",     model_paths=best_model_path("ou", DIR))
      NHLModel("spread", model_paths=best_model_path("spread", DIR)) # = same Poisson pair
    """

    def __init__(self, event: str, model_path=None, model_paths=None):
        self.event = event.lower()

        # Standalone ML classifier (optional)
        self.model = None

        # Poisson dual models — shared across all Poisson-based predictions
        self.poisson_gf = None
        self.poisson_ga = None

        if self.event == "ml":
            if model_paths is not None:
                # Poisson-based moneyline
                self._load_poisson_pair(model_paths)
            elif model_path is not None:
                # Standalone classifier
                self.model = _load_booster(model_path)
                print(f"[ml] Loaded classifier: {model_path}")
            else:
                raise ValueError("ml requires model_path or model_paths")

        elif self.event in ("ou", "spread"):
            if model_paths is None:
                raise ValueError(f"{self.event} requires model_paths=(gf_path, ga_path)")
            self._load_poisson_pair(model_paths)

        else:
            raise ValueError(f"Unknown event: {self.event}")

    def _load_poisson_pair(self, model_paths: tuple[str, str]):
        """Load the shared Poisson GF/GA model pair."""
        gf_path, ga_path = model_paths
        self.poisson_gf = _load_booster(gf_path)
        self.poisson_ga = _load_booster(ga_path)
        print(f"[{self.event}] Loaded Poisson pair: {gf_path}, {ga_path}")

    # ── Properties ────────────────────────────────────────────────────

    @property
    def uses_poisson(self) -> bool:
        """True if this model instance uses Poisson dual models."""
        return self.poisson_gf is not None

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

    # ── Core: Poisson λ prediction (single forward pass) ──────────────

    def _predict_lambdas(self, match_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict λ_gf and λ_ga from the Poisson dual models.
        This single forward pass is the foundation for ALL three markets.
        """
        dmat_gf = self._to_dmatrix(self.poisson_gf, match_df)
        dmat_ga = self._to_dmatrix(self.poisson_ga, match_df)
        lambda_gf = self.poisson_gf.predict(dmat_gf)
        lambda_ga = self.poisson_ga.predict(dmat_ga)
        return lambda_gf, lambda_ga

    # ── Prediction dispatch ───────────────────────────────────────────

    def predict(self, match_df: pd.DataFrame, threshold=None) -> np.ndarray:
        """
        Unified predict method.
          ml:     → [[P(away), P(home)]]
          ou:     → [[P(under), P(over)]]       (threshold = O/U line)
          spread: → [[P(not_cover), P(cover)]]   (threshold = spread line)
        """
        if self.event == "ml":
            return self._predict_ml(match_df)

        if threshold is None:
            raise ValueError(f"{self.event} prediction requires a threshold")

        if self.event == "ou":
            return self._predict_ou(match_df, float(threshold))

        if self.event == "spread":
            return self._predict_spread(match_df, float(threshold))

    def _predict_ml(self, match_df: pd.DataFrame) -> np.ndarray:
        if self.uses_poisson:
            lambda_gf, lambda_ga = self._predict_lambdas(match_df)
            probs = ml_probability_from_poisson(lambda_gf, lambda_ga)
            return np.stack([probs["away"], probs["home"]], axis=1)
        else:
            dmat = self._to_dmatrix(self.model, match_df)
            pred = self.model.predict(dmat)
            return np.stack([1 - pred, pred], axis=1)

    def _predict_ou(self, match_df: pd.DataFrame, line: float) -> np.ndarray:
        lambda_gf, lambda_ga = self._predict_lambdas(match_df)
        lambda_total = lambda_gf + lambda_ga
        probs = ou_probability(lambda_total, line)
        return np.stack([probs["under"], probs["over"]], axis=1)

    def _predict_spread(self, match_df: pd.DataFrame, line: float) -> np.ndarray:
        """
        Spread via Skellam distribution on (λ_gf, λ_ga).

        Previously this used a separate spread_reg model + normal CDF.
        Now it derives spread probabilities directly from the same Poisson
        λ values, giving exact discrete probabilities.
        """
        lambda_gf, lambda_ga = self._predict_lambdas(match_df)
        probs = spread_probability(lambda_gf, lambda_ga, line)
        return np.stack([probs["not_cover"], probs["cover"]], axis=1)

    # ── Batch prediction (all markets, one forward pass) ──────────────

    def predict_all(self, match_df: pd.DataFrame, ou_line=5.5, spread_line=-1.5) -> dict:
        """
        Predict all three markets from a SINGLE Poisson forward pass.
        Only works when the model is Poisson-based.

        Returns a dict with λ values and probabilities for all markets.
        This is what the /api/nhl/predict endpoint should use.
        """
        if not self.uses_poisson:
            raise ValueError("predict_all requires Poisson-based model (pass model_paths)")

        # One forward pass → two λ values → three markets
        lambda_gf, lambda_ga = self._predict_lambdas(match_df)
        lambda_total = lambda_gf + lambda_ga

        ml_probs = ml_probability_from_poisson(lambda_gf, lambda_ga)
        ou_probs = ou_probability(lambda_total, ou_line)
        sp_probs = spread_probability(lambda_gf, lambda_ga, spread_line)

        def _scalar(x):
            """Extract scalar from numpy array or return float directly."""
            return float(x[0]) if hasattr(x, '__len__') else float(x)

        return {
            "lambda_gf":    float(lambda_gf[0]),
            "lambda_ga":    float(lambda_ga[0]),
            "expected_diff": float(lambda_gf[0] - lambda_ga[0]),
            "lambda_total": float(lambda_total[0]),
            "ml": {
                "home":           float(ml_probs["home"][0]),
                "away":           float(ml_probs["away"][0]),
                "ot_probability": float(ml_probs["ot_probability"][0]),
            },
            "ou": {
                "line":  ou_line,
                "over":  _scalar(ou_probs["over"]),
                "under": _scalar(ou_probs["under"]),
                "push":  _scalar(ou_probs["push"]),
            },
            "spread": {
                "line":      spread_line,
                "cover":     _scalar(sp_probs["cover"]),
                "not_cover": _scalar(sp_probs["not_cover"]),
                "push":      _scalar(sp_probs["push"]),
            },
        }

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
            scorer = elo.Elo(50, 0.2)
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
        End-to-end: load CSV → build match → predict.
        Convenience method for standalone usage.
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_DIR = "./backend/model/models"
    poisson_paths = best_model_path("ou", MODEL_DIR)

    print("=" * 70)
    print("POISSON-UNIFIED PREDICTIONS  (single model pair → all 3 markets)")
    print("=" * 70)

    # Load once, predict everything
    model = NHLModel("ou", model_paths=poisson_paths)
    df = pd.read_csv("all_games_preproc.csv")
    match_df = model.create_match(df, "PIT", "FLA")

    results = model.predict_all(match_df, ou_line=5.5, spread_line=-1.5)

    print(f"\nPIT (home) vs FLA (away)")
    print(f"  λ_gf = {results['lambda_gf']:.3f}   (expected home goals)")
    print(f"  λ_ga = {results['lambda_ga']:.3f}   (expected away goals)")
    print(f"  E[diff] = {results['expected_diff']:+.3f}")
    print(f"  λ_total = {results['lambda_total']:.3f}")
    print()
    print(f"  ML:       P(home) = {results['ml']['home']:.1%}   "
          f"P(away) = {results['ml']['away']:.1%}   "
          f"P(OT) = {results['ml']['ot_probability']:.1%}")
    print(f"  O/U 5.5:  P(over) = {results['ou']['over']:.1%}   "
          f"P(under) = {results['ou']['under']:.1%}")
    print(f"  Spread -1.5: P(cover) = {results['spread']['cover']:.1%}   "
          f"P(not_cover) = {results['spread']['not_cover']:.1%}")

    # Also test individual event interfaces
    print("\n" + "-" * 70)
    print("Individual event models (same Poisson pair under the hood):")
    print("-" * 70)

    ml = NHLModel("ml", model_paths=poisson_paths)
    print(f"\nML:         {ml.get_team_prediction('PIT', 'FLA')}")

    ou = NHLModel("ou", model_paths=poisson_paths)
    print(f"OU 5.5:     {ou.get_team_prediction('PIT', 'FLA', threshold=5.5)}")

    sp = NHLModel("spread", model_paths=poisson_paths)
    print(f"Spread -1.5: {sp.get_team_prediction('PIT', 'FLA', threshold=-1.5)}")

    # Compare with standalone ML classifier if available
    print("\n" + "-" * 70)
    print("Standalone ML classifier (for comparison):")
    print("-" * 70)
    try:
        ml_cls = NHLModel("ml", model_path=best_model_path("ml", MODEL_DIR))
        print(f"ML classifier: {ml_cls.get_team_prediction('PIT', 'FLA')}")
    except Exception as e:
        print(f"Not available: {e}")
