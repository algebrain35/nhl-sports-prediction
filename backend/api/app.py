from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import sys
import os
import gc
import logging

sys.path.insert(1, "backend/model")
sys.path.insert(2, "backend/db")

from dotenv import load_dotenv
from nhl_model import NHLModel, best_model_path
from db import NHLPipeline, create_table_map

load_dotenv()

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5000"
).split(",")

CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS,
     allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")

# ── Models (loaded once at startup) ───────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_DIR", "./backend/model/models")
DATA_PATH = os.getenv("DATA_PATH", "all_games_preproc.csv")

# Lazy-load models to reduce cold start memory spike
_models = {}

def _get_model(event: str) -> NHLModel:
    """Lazy-load and cache models on first request."""
    if event not in _models:
        if event == "ml":
            _models[event] = NHLModel("ml", model_path=best_model_path("ml", MODEL_DIR))
        elif event == "ou":
            _models[event] = NHLModel("ou", model_paths=best_model_path("ou", MODEL_DIR))
        elif event == "spread":
            _models[event] = NHLModel("spread", model_path=best_model_path("spread", MODEL_DIR))
        gc.collect()  # free any transient allocs from model loading
    return _models[event]


# ── Shared CSV loader (cached) ────────────────────────────────────────────────

_df_cache = {"df": None, "mtime": 0}

def _get_data() -> pd.DataFrame:
    """Load CSV once, reload only if file changed."""
    try:
        mtime = os.path.getmtime(DATA_PATH)
        if _df_cache["df"] is None or mtime > _df_cache["mtime"]:
            _df_cache["df"] = pd.read_csv(DATA_PATH, low_memory=False)
            _df_cache["mtime"] = mtime
            logger.info("Loaded %d rows from %s", len(_df_cache["df"]), DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    return _df_cache["df"]


# ── Prediction endpoints (no auth required) ──────────────────────────────────

@app.route("/api/nhl/ml/predict", methods=["GET"])
def predict_ml():
    try:
        home = request.args.get("home")
        away = request.args.get("away")
        if not home or not away:
            return jsonify({"error": "home and away params required"}), 400

        model = _get_model("ml")
        df = _get_data()
        match_df = model.create_match(df, home, away)
        preds = model.predict(match_df)
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        logger.exception("ML prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/nhl/ou/predict", methods=["GET"])
def predict_ou():
    try:
        home = request.args.get("home")
        away = request.args.get("away")
        ou = request.args.get("ou")
        if not home or not away or not ou:
            return jsonify({"error": "home, away, and ou params required"}), 400

        model = _get_model("ou")
        df = _get_data()
        match_df = model.create_match(df, home, away)
        preds = model.predict(match_df, threshold=float(ou))
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        logger.exception("OU prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/nhl/spread/predict", methods=["GET"])
def predict_spread():
    try:
        home = request.args.get("home")
        away = request.args.get("away")
        spread = request.args.get("spread")
        if not home or not away or not spread:
            return jsonify({"error": "home, away, and spread params required"}), 400

        model = _get_model("spread")
        df = _get_data()
        match_df = model.create_match(df, home, away)
        preds = model.predict(match_df, threshold=float(spread))
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        logger.exception("Spread prediction failed")
        return jsonify({"error": str(e)}), 500


# ── Batch prediction (all 3 events in one call) ──────────────────────────────

@app.route("/api/nhl/predict", methods=["GET"])
def predict_all():
    """
    Single endpoint returning ML + OU + spread for one matchup.
    ?home=PIT&away=FLA&ou=5.5&spread=-1.5
    """
    try:
        home = request.args.get("home")
        away = request.args.get("away")
        ou = request.args.get("ou", "5.5")
        spread = request.args.get("spread", "-1.5")
        if not home or not away:
            return jsonify({"error": "home and away params required"}), 400

        df = _get_data()

        # Build match once, reuse for all models
        ml_model = _get_model("ml")
        match_df = ml_model.create_match(df, home, away)

        ml_preds = ml_model.predict(match_df)
        ou_preds = _get_model("ou").predict(match_df, threshold=float(ou))
        sp_preds = _get_model("spread").predict(match_df, threshold=float(spread))

        return jsonify({
            "home": home,
            "away": away,
            "ml":     ml_preds.tolist()[0],
            "ou":     {"line": float(ou), "probs": ou_preds.tolist()[0]},
            "spread": {"line": float(spread), "probs": sp_preds.tolist()[0]},
        }), 200
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": str(e)}), 500


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(_models.keys()),
        "data_rows": len(_df_cache["df"]) if _df_cache["df"] is not None else 0,
    })


# ── Init ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
