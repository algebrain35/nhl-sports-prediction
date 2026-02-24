import os
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import elo as elo

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../backend/data")

# ── Constants ─────────────────────────────────────────────────────────────────

EMA_SPANS = [5, 13]
SEASONS   = list(range(2008, 2026))

REGULAR_URL = "https://moneypuck.com/moneypuck/playerData/teamGameByGame/{season}/regular/{team}.csv"
PLAYOFF_URL = "https://moneypuck.com/moneypuck/playerData/teamGameByGame/{season}/playoffs/{team}.csv"

DOWNLOAD_TIMEOUT_MS = 15_000
NAV_TIMEOUT_MS      = 30_000

EMA_EXCLUDE = {
    "gameId", "gameDate", "season", "iceTime",
    "winner", "goalDiffFor", "totalGoals",
    "goalsFor", "goalsAgainst",
}

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

KEEP_AS_IS_EMA = [
    "xGoalsPercentage", "corsiPercentage", "fenwickPercentage",
    "winRateFor", "goalsForPerGame", "goalsAgainstPerGame",
    "totalGoalsPerGame", "ryderExpFor", "ryderProbFor",
]


# ── Standalone utilities ──────────────────────────────────────────────────────

def get_float_features(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.float64]).columns.tolist()


def calculate_seasonal_ema(df: pd.DataFrame, col: str, span: int) -> np.ndarray:
    """
    Lag-1 EMA: value at row i uses only rows 0..i-1.
    shift(1) guarantees zero lookahead.
    """
    if df[col].dtype != np.float64:
        raise TypeError(f"Expected float64 for '{col}', got {df[col].dtype}")
    return df[col].ewm(span=span, adjust=False).mean().shift(1).to_numpy()


def _is_html(content: bytes) -> bool:
    return b"<!DOCTYPE" in content[:200] or b"<html" in content[:200]


# ── Scraper ───────────────────────────────────────────────────────────────────

class Scraper:
    SUPPORTED_SPORTS = {"NHL"}

    def __init__(self, sport: str, headless: bool = True):
        if sport not in self.SUPPORTED_SPORTS:
            raise ValueError(f"Unsupported sport '{sport}'")
        self.sport    = sport
        self.headless = headless
        self._pw      = None
        self._browser = None

    def _load_teams(self, path: str = "./team_files") -> list:
        teams_file = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        if not os.path.exists(teams_file):
            teams_file = path
        with open(teams_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _save_path(self, team: str, season: int, game_type: str) -> str:
        return os.path.join(
            DATA_PATH, self.sport, "teams", game_type, str(season), f"{team}.csv"
        )

    # ── Browser lifecycle ─────────────────────────────────────────────────

    def _start_browser(self) -> None:
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        logger.info("Browser started (headless=%s)", self.headless)

    def _stop_browser(self) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
        self._browser = None
        self._pw      = None
        logger.info("Browser stopped")

    # ── Single file download ──────────────────────────────────────────────

    @staticmethod
    def _validate_csv(path: str) -> bool:
        """Check that a file looks like a valid CSV, not HTML."""
        try:
            with open(path, "rb") as f:
                head = f.read(500)
            if _is_html(head):
                return False
            first_line = head.split(b"\n")[0]
            return b"," in first_line and len(head) > 50
        except Exception:
            return False

    def _download_http(self, url: str, save_path: str) -> bool:
        """
        Tier 1: Direct HTTP request using the browser's network context.
        Fastest approach — no page rendering needed.
        Uses a fresh context so we get proper cookie/header handling.
        """
        context = None
        try:
            context = self._browser.new_context()
            response = context.request.get(url, timeout=DOWNLOAD_TIMEOUT_MS)

            if response.status != 200:
                logger.debug("HTTP %d: %s", response.status, url)
                return False

            body = response.body()
            if _is_html(body[:200]):
                logger.debug("HTML response (Cloudflare?): %s", url)
                return False

            # Check it looks like CSV
            first_line = body[:500].split(b"\n")[0]
            if b"," not in first_line or len(body) < 50:
                logger.debug("Not CSV content: %s", url)
                return False

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(body)
            logger.debug("Saved (HTTP): %s", save_path)
            return True

        except Exception as e:
            logger.debug("HTTP download failed (%s): %s", type(e).__name__, url)
            return False
        finally:
            try:
                if context:
                    context.close()
            except Exception:
                pass

    def _download_browser(self, url: str, save_path: str) -> bool:
        """
        Tier 2: Browser-based download with expect_download + text fallback.
        Used when direct HTTP fails (e.g., Cloudflare challenge).

        Strategy:
          a) Navigate and check if a download event fires.
          b) If no download, read the page body as text (CSV rendered inline).
        """
        context = None
        page    = None
        tmp     = save_path + ".tmp"

        try:
            context = self._browser.new_context(accept_downloads=True)
            page    = context.new_page()

            # Navigate first, then check what happened
            response = page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_MS)

            # If the server returned a non-200 status, bail
            if response and response.status >= 400:
                logger.debug("HTTP %d (browser): %s", response.status, url)
                return False

            # ── Check for download event ─────────────────────────────────
            # After navigation, check if any downloads started.
            # Use a short timeout — if it's a download, it would have
            # triggered during navigation already.
            try:
                with page.expect_download(timeout=3_000) as download_info:
                    pass  # download should already be in-flight from goto
                download = download_info.value
                download.save_as(tmp)

                if self._validate_csv(tmp):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    os.replace(tmp, save_path)
                    logger.debug("Saved (browser download): %s", save_path)
                    return True
                else:
                    logger.warning("Invalid CSV from download: %s", url)
                    return False

            except PlaywrightTimeout:
                pass  # No download — content is rendered in page

            # ── Fallback: read page body as text ─────────────────────────
            body_text = page.inner_text("body").strip()

            if not body_text or len(body_text) < 50:
                logger.debug("Empty/short page body: %s", url)
                return False

            first_line = body_text.split("\n")[0]
            if "," not in first_line:
                logger.debug("Not CSV content in body: %s", url)
                return False

            csv_bytes = body_text.encode("utf-8")
            if _is_html(csv_bytes[:200]):
                logger.debug("HTML in body text: %s", url)
                return False

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(csv_bytes)
            logger.debug("Saved (browser text): %s", save_path)
            return True

        except Exception:
            logger.exception("Browser download failed: %s", url)
            return False

        finally:
            try:
                if page and not page.is_closed():
                    page.close()
            except Exception:
                pass
            try:
                if context:
                    context.close()
            except Exception:
                pass
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def _download_file(self, url: str, save_path: str) -> bool:
        """
        Download a CSV file with a tiered strategy:
          1. Direct HTTP via Playwright API context (fast, no rendering)
          2. Full browser with expect_download + text fallback (handles CF)
        """
        # Tier 1: direct HTTP — works for most MoneyPuck URLs
        if self._download_http(url, save_path):
            return True

        # Tier 2: full browser rendering — handles Cloudflare, JS challenges
        return self._download_browser(url, save_path)

    # ── Public download API ───────────────────────────────────────────────

    def download_nhl_team_data(
        self,
        regular:  bool = True,
        playoff:  bool = True,
        seasons:  list = None,
    ) -> None:
        """
        Download one CSV per team per season per game type.
        Skips files already on disk so interrupted runs can resume.
        """
        teams   = self._load_teams()
        seasons = seasons or SEASONS

        self._start_browser()
        try:
            for season in seasons:
                for team in tqdm(teams, desc=f"Season {season}"):
                    if regular:
                        url  = REGULAR_URL.format(season=season, team=team)
                        path = self._save_path(team, season, "regular")
                        if not os.path.exists(path):
                            self._download_file(url, path)

                    if playoff:
                        url  = PLAYOFF_URL.format(season=season, team=team)
                        path = self._save_path(team, season, "playoff")
                        if not os.path.exists(path):
                            self._download_file(url, path)

                logger.info("Season %d complete", season)
        finally:
            self._stop_browser()


# ── Preprocessor ─────────────────────────────────────────────────────────────

class Preprocessor:
    def __init__(self, sport: str):
        if sport != "NHL":
            raise ValueError(f"Unsupported sport: {sport}")
        self.sport   = sport
        self.subject = "teams"

    # ── I/O ───────────────────────────────────────────────────────────────

    def load_all_csvs(self) -> pd.DataFrame:
        sport_dir = os.path.join(DATA_PATH, self.sport, self.subject)
        if not os.path.exists(sport_dir):
            raise FileNotFoundError(f"Data directory not found: {sport_dir}")

        frames = []
        for game_type in ("regular", "playoff"):
            type_dir = os.path.join(sport_dir, game_type)
            if not os.path.isdir(type_dir):
                logger.warning("Missing directory: %s", type_dir)
                continue
            for season_dir in sorted(os.listdir(type_dir)):
                season_path = os.path.join(type_dir, season_dir)
                if not os.path.isdir(season_path):
                    continue
                try:
                    season = int(season_dir)
                except ValueError:
                    logger.warning("Skipping non-season dir: %s", season_path)
                    continue
                for fname in os.listdir(season_path):
                    if not fname.endswith(".csv"):
                        continue
                    fpath = os.path.join(season_path, fname)
                    try:
                        with open(fpath, "rb") as f:
                            if _is_html(f.read(200)):
                                logger.warning("Skipping HTML file: %s", fpath)
                                continue
                        df = pd.read_csv(fpath)
                        df["season"]    = season
                        df["game_type"] = game_type
                        frames.append(df)
                    except Exception:
                        logger.exception("Failed to read %s", fpath)

        if not frames:
            raise RuntimeError(
                f"No valid CSVs found under {sport_dir}. "
                "Run Scraper.download_nhl_team_data() first."
            )

        combined = pd.concat(frames, ignore_index=True).sort_values("gameId")
        logger.info("Loaded %d rows from %d files", len(combined), len(frames))
        return combined

    # ── Cleaning ──────────────────────────────────────────────────────────

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["situation"] == "all"].copy()
        if df.empty:
            return df

        df["home_or_away"] = df["home_or_away"].str.strip().str.upper()

        # Targets (never used as features)
        df["winner"]      = np.where(df["goalsFor"] > df["goalsAgainst"], 1.0, 0.0)
        df["goalDiffFor"] = df["goalsFor"] - df["goalsAgainst"]
        df["totalGoals"]  = df["goalsFor"] + df["goalsAgainst"]

        # Lagged expanding averages — shift(1) prevents lookahead
        for src_col, new_col in [
            ("winner",       "winRateFor"),
            ("goalsFor",     "goalsForPerGame"),
            ("goalsAgainst", "goalsAgainstPerGame"),
            ("totalGoals",   "totalGoalsPerGame"),
        ]:
            df[new_col] = (
                df.groupby(["team", "season"])[src_col]
                .transform(lambda x: x.expanding().mean().shift(1))
            )

        # Ryder — using LAGGED averages only (no current-game leakage)
        exp = (df["goalsForPerGame"] + df["goalsAgainstPerGame"]).pow(0.458)
        gf_lagged = df["goalsForPerGame"].pow(exp)
        ga_lagged = df["goalsAgainstPerGame"].pow(exp)
        df["ryderExpFor"]  = exp
        df["ryderProbFor"] = gf_lagged / (gf_lagged + ga_lagged)

        return df

    # ── EMA ───────────────────────────────────────────────────────────────

    def _ema_for_group(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("gameId").copy()
        for col in get_float_features(df):
            if "_seasonal_ema_" in col or col in EMA_EXCLUDE:
                continue
            for span in EMA_SPANS:
                df[f"{col}_seasonal_ema_span_{span}"] = calculate_seasonal_ema(df, col, span)
        return df

    def apply_seasonal_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_df = (
            df.groupby(["team", "season"], group_keys=False)
            .apply(self._ema_for_group)
            .sort_values("gameId")
            .reset_index(drop=True)
        )
        new_cols = [c for c in ema_df.columns if c not in df.columns]
        return pd.concat(
            [df.sort_values("gameId").reset_index(drop=True),
             ema_df[new_cols]],
            axis=1,
        )

    # ── Elo ───────────────────────────────────────────────────────────────

    def _run_elo_pass(self, df: pd.DataFrame, K: float, decay: float) -> pd.DataFrame:
        """Returns Elo mapping for ALL rows (both home and away perspectives)."""
        home_df  = df[df["home_or_away"] == "HOME"].sort_values("gameId")
        scorer   = elo.Elo(K, decay)
        team_set = set(home_df["team"].unique()) | set(home_df["opposingTeam"].unique())
        for team in team_set:
            scorer.add_team(team)

        n            = len(home_df)
        elo_home     = np.zeros(n)
        elo_away     = np.zeros(n)
        elo_expected = np.zeros(n)
        current_season = None

        for i, row in enumerate(home_df.itertuples()):
            if current_season is None or row.season != current_season:
                scorer.set_season(row.season, team_set)
                current_season = row.season

            home, away        = row.team, row.opposingTeam
            elo_home[i]       = scorer[home]
            elo_away[i]       = scorer[away]
            elo_expected[i]   = scorer.get_expect_result(elo_home[i], elo_away[i])

            margin = scorer.get_margin_factor(row.goalsFor - row.goalsAgainst)
            if row.winner == 1.0:
                winner, loser = home, away
                inflation = scorer.get_inflation_factor(elo_home[i], elo_away[i])
            else:
                winner, loser = away, home
                inflation = scorer.get_inflation_factor(elo_away[i], elo_home[i])

            scorer.update_ratings(winner, loser, K, margin, inflation)

        # Home perspective
        home_map = pd.DataFrame({
            "gameId": home_df["gameId"].values,
            "team":   home_df["team"].values,
            "eloFor": elo_home, "eloAgainst": elo_away,
            "eloExpectedFor": elo_expected, "eloExpectedAgainst": 1 - elo_expected,
        })
        # Away perspective (flipped)
        away_map = pd.DataFrame({
            "gameId": home_df["gameId"].values,
            "team":   home_df["opposingTeam"].values,
            "eloFor": elo_away, "eloAgainst": elo_home,
            "eloExpectedFor": 1 - elo_expected, "eloExpectedAgainst": elo_expected,
        })
        return pd.concat([home_map, away_map], ignore_index=True)

    def _elo_log_loss(self, df: pd.DataFrame, K: float, decay: float) -> float:
        home_df = df[df["home_or_away"] == "HOME"].sort_values("gameId")
        elo_map = self._run_elo_pass(df, K, decay)
        home_elo = elo_map[elo_map["gameId"].isin(home_df["gameId"])]
        home_elo = home_elo.drop_duplicates(subset=["gameId", "team"], keep="first")
        home_elo = home_elo[home_elo["team"].isin(home_df["team"].values)].sort_values("gameId")
        expected = np.clip(home_elo["eloExpectedFor"].values[:len(home_df)], 1e-7, 1 - 1e-7)
        actual   = home_df["winner"].values[:len(expected)]
        errors   = -(actual * np.log(expected) + (1 - actual) * np.log(1 - expected))
        return float(np.mean(errors))

    def apply_elo_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        space = [Real(10, 50, name="K"), Real(0.0, 0.2, name="decay")]

        @use_named_args(space)
        def objective(K, decay):
            return self._elo_log_loss(df, K, decay)

        logger.info("Optimizing Elo hyperparameters (30 calls)...")
        result = gp_minimize(objective, space, n_calls=30, random_state=42)
        best_K, best_decay = result.x
        logger.info("Best Elo: K=%.2f  decay=%.4f  loss=%.4f", best_K, best_decay, result.fun)

        elo_map = self._run_elo_pass(df, best_K, best_decay)
        df = df.merge(elo_map, on=["gameId", "team"], how="left")
        return df

    # ── Feature Engineering ───────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "home_or_away" in df.columns:
            df["is_home"] = (df["home_or_away"] == "HOME").astype(int)

        # PDO from lagged EMAs only
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

        # Elo differential
        if "eloFor" in df.columns and "eloAgainst" in df.columns:
            df["elo_diff"] = df["eloFor"] - df["eloAgainst"]

        # Back-to-back / days rest
        if "gameDate" in df.columns:
            try:
                df["gameDate_dt"] = pd.to_datetime(df["gameDate"])
                df = df.sort_values(["team", "gameDate_dt"])
                df["days_rest"] = (
                    df.groupby("team")["gameDate_dt"]
                    .diff().dt.days.fillna(3).clip(upper=7)
                )
                df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
                df = df.drop(columns=["gameDate_dt"])
            except Exception:
                logger.warning("Back-to-back detection failed", exc_info=True)

        return df

    def select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        SAFE_CONTEXTUAL = {
            "is_home", "elo_diff", "pdo",
            "eloFor", "eloAgainst", "eloExpectedFor", "eloExpectedAgainst",
            "days_rest", "is_back_to_back",
        }

        feature_cols = []
        feature_cols += [c for c in df.columns if "_diff_ema_" in c]
        feature_cols += [c for c in df.columns if "_momentum" in c]
        feature_cols += [c for c in SAFE_CONTEXTUAL if c in df.columns]

        for metric in KEEP_AS_IS_EMA:
            for span in EMA_SPANS:
                col = f"{metric}_seasonal_ema_span_{span}"
                if col in df.columns:
                    feature_cols.append(col)

        for col in ["eloFor", "eloAgainst"]:
            if col in df.columns:
                feature_cols.append(col)

        # Whitelist validation
        validated = []
        seen = set()
        for c in feature_cols:
            if c in seen:
                continue
            seen.add(c)
            if ("ema" in c or "_diff_" in c or "_momentum" in c or c in SAFE_CONTEXTUAL):
                validated.append(c)
            else:
                logger.warning("BLOCKED potential leakage feature: %s", c)

        logger.info("Selected %d feature columns", len(validated))
        return validated

    # ── Master pipeline ───────────────────────────────────────────────────

    def update_csv(self, output_path: str = "all_games_preproc.csv") -> pd.DataFrame:
        raw = self.load_all_csvs()
        logger.info("Raw rows: %d  columns: %d", *raw.shape)

        df = self.clean_dataframe(raw)
        if df.empty:
            raise RuntimeError("DataFrame empty after cleaning.")
        logger.info("Rows after cleaning: %d", len(df))

        df = self.apply_seasonal_ema(df)
        logger.info("Rows after EMA: %d", len(df))

        df = self.apply_elo_rating(df)

        df = self.engineer_features(df)
        logger.info("Rows after feature engineering: %d, cols: %d", *df.shape)

        feature_cols = self.select_feature_columns(df)

        df = df.sort_values("gameId").set_index("gameId")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.to_csv(output_path)
        logger.info("Saved %s: %d rows  %d cols", output_path, *df.shape)

        feature_path = output_path.replace(".csv", "_features.json")
        with open(feature_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        logger.info("Saved feature list → %s (%d features)", feature_path, len(feature_cols))

        return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    scraper = Scraper("NHL")
    scraper.download_nhl_team_data()

    preproc = Preprocessor("NHL")
    preproc.update_csv()
