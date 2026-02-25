# Project Radon

NHL prediction engine using XGBoost regression models for moneyline, over/under, and spread betting markets.

## What It Does

Radon takes ~20,000 historical NHL games, engineers 100+ features (rolling EMAs, Elo ratings, shot quality differentials, momentum signals), and trains three XGBoost models that produce calibrated probabilities for any matchup:

| Model | Method | Input | Output |
|---|---|---|---|
| **Moneyline** | Binary classifier | Team features | P(home win) |
| **Over/Under** | Dual Poisson regression (GF + GA) | Team features + O/U line | P(over), P(under) |
| **Spread** | Normal regression on goal differential | Team features + spread line | P(cover), P(not cover) |

The Poisson and spread regression models generalize to any betting line without retraining — they predict continuous values (expected goals, expected goal differential) and convert to probabilities via Poisson CDF and normal CDF respectively.

## Project Structure

```
project-radon/
├── app.py                          # Flask API (prediction endpoints)
├── Dockerfile                      # Production container
├── railway.toml                    # Railway deployment config
├── requirements.txt
├── all_games_preproc.csv           # Preprocessed game data
│
├── backend/
│   ├── model/
│   │   ├── nhl_model.py            # Inference: load models, build matches, predict
│   │   ├── nhl_train.py            # Training: walk-forward CV, feature selection
│   │   └── models/
│   │       ├── ml/                 # XGBoost_57.8%_ml.json
│   │       ├── poisson/            # poisson_goalsFor.json, poisson_goalsAgainst.json
│   │       └── spread_reg/         # spread_reg_goalDiff.json, spread_reg_meta.json
│   ├── preprocess/
│   │   ├── nhl_preprocess.py       # Feature engineering pipeline
│   │   └── elo.py                  # Elo rating system
│   ├── db/
│   │   └── db.py                   # Data pipeline
│   └── api/
│
└── web/                            # React frontend (Vite)
    ├── src/
    │   ├── App.jsx
    │   ├── pages/
    │   │   ├── GamesPage.jsx       # Today's games via ESPN API
    │   │   ├── PredictPage.jsx     # Manual matchup predictions
    │   │   └── AboutPage.jsx
    │   ├── components/
    │   └── hooks/
    ├── vite.config.js
    └── package.json
```

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Flask dev server
python app.py
# → http://localhost:5000
```

### Frontend

```bash
cd web
npm install
npm run dev
# → http://localhost:3000 (proxies /api to :5000)
```

### Training Models

```python
from backend.model.nhl_train import NHLModelTrainer

trainer = NHLModelTrainer()
df = trainer.load_data()

# Moneyline classifier
X, y = trainer.preprocess("ml", df)
trainer.train_ml(X, y)

# Poisson O/U (dual model)
X, y_gf, y_ga = trainer.preprocess_poisson(df)
trainer.train_poisson(X, y_gf, y_ga)

# Spread regression
X, y_diff = trainer.preprocess_spread_reg(df)
trainer.train_spread_reg(X, y_diff)
```

## API Endpoints

All prediction endpoints are open — no authentication required.

```
GET /api/nhl/ml/predict?home=PIT&away=FLA
    → {"predictions": [[0.42, 0.58]]}
       [P(away_win), P(home_win)]

GET /api/nhl/ou/predict?home=PIT&away=FLA&ou=5.5
    → {"predictions": [[0.45, 0.55]]}
       [P(under), P(over)]

GET /api/nhl/spread/predict?home=PIT&away=FLA&spread=-1.5
    → {"predictions": [[0.62, 0.38]]}
       [P(not_cover), P(cover)]

GET /api/nhl/predict?home=PIT&away=FLA&ou=5.5&spread=-1.5
    → All three in one call (builds match row once)

GET /api/health
    → {"status": "ok", "models_loaded": [...], "data_rows": 20000}
```

## How the Models Work

**Moneyline** — standard XGBoost binary classifier trained on win/loss outcomes with walk-forward time-series cross-validation and feature selection per fold.

**Over/Under** — two separate Poisson regressors predict expected goals-for (λ_gf) and goals-against (λ_ga). Their sum λ_total is passed through the Poisson CDF to get P(over) and P(under) at any line. Half-lines (5.5) have no push; whole-lines (5.0) account for push probability via the PMF.

**Spread** — a single regressor predicts the expected goal differential. The residual standard deviation σ (~2.5 for NHL) is computed from training residuals and saved alongside the model. At inference, the predicted differential and σ are passed through the normal CDF to get P(cover) at any spread line.

All three models share the same feature engineering pipeline: EMA-smoothed stats at spans of 5 and 13 games, for-against differentials across 45+ metrics, momentum signals (EMA5 − EMA13), Elo ratings with margin-of-victory adjustments, PDO, days rest, and back-to-back flags.

## Data Sources

- Game stats: [MoneyPuck](https://moneypuck.com) (2008–2025)
- Live scores: ESPN Scoreboard API (frontend only)
