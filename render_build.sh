#!/usr/bin/env bash
# Exit on error
set -o errexit

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Build React Frontend
cd web
npm install
npm run build
cd ..
