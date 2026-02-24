# Project Radon — React Frontend

## Directory Structure

```
web/
├── index.html                  # Single HTML entry (Stripe.js loaded here)
├── package.json
├── vite.config.js              # Dev server on :3000, proxies /api → Flask :5000
├── public/
│   └── images/
│       └── favicon.svg
└── src/
    ├── main.jsx                # ReactDOM entry
    ├── App.jsx                 # Router + AuthProvider wrapper
    ├── index.css               # Global styles + CSS variables
    ├── api/
    │   ├── client.js           # Shared fetch wrapper (credentials, base URL)
    │   └── constants.js        # Teams, ESPN mappings, Stripe price ID
    ├── hooks/
    │   └── useAuth.jsx         # Auth context (login/register/logout/refresh)
    ├── components/
    │   ├── Nav.jsx             # Sticky nav with route links + auth button
    │   ├── AuthDropdown.jsx    # Login/register/account dropdown
    │   ├── GameCard.jsx        # ESPN game card with inline predictions
    │   ├── ProbBar.jsx         # Probability bar visualization
    │   └── StatusBadge.jsx     # LIVE / FINAL / SCHEDULED badge
    └── pages/
        ├── GamesPage.jsx       # Today's NHL games from ESPN API
        ├── PredictPage.jsx     # Manual matchup predictions (ML/spread/OU)
        ├── PricingPage.jsx     # Stripe subscription flow
        └── AboutPage.jsx       # About + how the model works
```

## Setup Instructions

### Prerequisites
- Node.js 18+ installed (https://nodejs.org)
- Your Flask backend running on `http://localhost:5000`

### 1. Replace your current `web/` directory

```bash
# From your project root
rm -rf web/               # remove old static files
mv web-react web/         # or unzip into web/
```

### 2. Install dependencies

```bash
cd web
npm install
```

This installs React, React Router, and Vite.

### 3. Start the dev server

```bash
npm run dev
```

This starts Vite on **http://localhost:3000**. All `/api/*` requests are
automatically proxied to your Flask backend on `:5000`, so you don't need
the old Express proxy server (`server.js`) anymore.

### 4. Make sure Flask allows CORS from localhost:3000

In your Flask `server.py`, ensure you have:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
```

The `supports_credentials=True` is required for session cookies to work
across the proxy.

### 5. Open in browser

Navigate to **http://localhost:3000**. You should see today's NHL games.

## Production Build

```bash
npm run build
```

This outputs static files to `web/dist/`. To serve them:

**Option A — Flask serves the built files:**

```python
# In server.py
from flask import send_from_directory

app = Flask(__name__, static_folder="../web/dist", static_url_path="/")

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")
```

**Option B — Nginx (recommended for production):**

```nginx
server {
    listen 80;

    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
    }

    location / {
        root /path/to/web/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

## API Endpoints Used

All existing Flask endpoints are preserved:

| Method | Path                          | Description           |
|--------|-------------------------------|-----------------------|
| GET    | /api/user                     | Get session user      |
| POST   | /api/login                    | Login                 |
| POST   | /api/register                 | Register              |
| POST   | /api/logout                   | Logout                |
| GET    | /api/nhl/ml/predict           | Moneyline prediction  |
| GET    | /api/nhl/ou/predict           | Over/under prediction |
| GET    | /api/nhl/spread/predict       | Spread prediction     |
| POST   | /api/stripe/subscription      | Create subscription   |
| POST   | /api/stripe/cancel            | Cancel subscription   |

## Configuration

Edit `src/api/constants.js` to update:
- `STRIPE_PRICE_ID` — your Stripe price ID
- `ESPN_TO_BACKEND` — team abbreviation mapping if teams change

For a custom API base URL (e.g., production), create a `.env` file:

```
VITE_API_BASE=https://api.yourdomain.com
```

## What Changed from the Static Site

- **Removed**: `history.html`, `history.js`, Express `server.js` proxy
- **Added**: React Router for SPA navigation, ESPN live game integration
- **Auth**: Dropdown in nav instead of duplicated forms on every page
- **Predictions**: Inline predictions on game cards + manual prediction page
- **Stripe**: Same flow, integrated into React component lifecycle
