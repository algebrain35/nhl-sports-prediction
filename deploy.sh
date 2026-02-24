#!/usr/bin/env bash
set -euo pipefail

PROD_FLAG=""
if [[ "${1:-}" == "--prod" ]]; then
    PROD_FLAG="--prod"
    echo "==> Deploying to PRODUCTION"
else
    echo "==> Deploying to PREVIEW"
fi

echo ""
echo "── Checking model files ──"
MODEL_DIR="./backend/model/models"

REQUIRED_FILES=(
    "$MODEL_DIR/poisson/poisson_goalsFor.json"
    "$MODEL_DIR/poisson/poisson_goalsAgainst.json"
    "$MODEL_DIR/spread_reg/spread_reg_goalDiff.json"
    "$MODEL_DIR/spread_reg/spread_reg_meta.json"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "   MISSING: $f"
        MISSING=1
    else
        SIZE=$(du -sh "$f" | cut -f1)
        echo "   OK: $f ($SIZE)"
    fi
done

ML_DIR="$MODEL_DIR/ml"
if [[ -d "$ML_DIR" ]]; then
    ML_COUNT=$(find "$ML_DIR" -name "*.json" | wc -l)
    if [[ "$ML_COUNT" -eq 0 ]]; then
        echo "   MISSING: No ML model files in $ML_DIR"
        MISSING=1
    else
        echo "   OK: $ML_COUNT ML model(s) in $ML_DIR"
    fi
else
    echo "   MISSING: $ML_DIR directory"
    MISSING=1
fi

if [[ "$MISSING" -eq 1 ]]; then
    echo ""
    echo "ERROR: Missing model files. Train models first:"
    echo "  python -m backend.model.nhl_train  (or your training script)"
    exit 1
fi

if [[ ! -f "all_games_preproc.csv" ]]; then
    echo "   MISSING: all_games_preproc.csv"
    echo "ERROR: Run preprocessing first to generate all_games_preproc.csv"
    exit 1
else
    ROWS=$(wc -l < all_games_preproc.csv)
    echo "   OK: all_games_preproc.csv ($ROWS rows)"
fi

echo ""
echo "── Checking deploy size ──"
TOTAL_SIZE=$(du -sh --exclude=node_modules --exclude=.git --exclude=__pycache__ . | cut -f1)
echo "   Total project size: $TOTAL_SIZE"
echo "   (Vercel limit: 250MB per serverless function)"


echo ""
echo "── Building frontend ──"
cd web
if [[ ! -d "node_modules" ]]; then
    echo "   Installing dependencies..."
    npm install --silent
fi
echo "   Running vite build..."
npm run build
cd ..
echo "   Built to web/dist/"


echo ""
echo "── Deploying to Vercel ──"

if ! command -v vercel &> /dev/null; then
    echo "   Installing Vercel CLI..."
    npm i -g vercel
fi

vercel deploy $PROD_FLAG

echo ""
echo "── Done ──"
