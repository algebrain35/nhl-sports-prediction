import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../api/client";
import { TEAMS } from "../api/constants";
import { useAuth } from "../hooks/useAuth";
import ProbBar from "../components/ProbBar";

export default function PredictPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [ou, setOu] = useState("");
  const [spread, setSpread] = useState("");
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState({});

 /* useEffect(() => {
    if (!user?.is_subscribed) navigate("/pricing");
  }, [user, navigate]); */

  const predict = async (type) => {
    if (!home || !away) return;
    setLoading((l) => ({ ...l, [type]: true }));
    try {
      let url;
      if (type === "ml") url = `/api/nhl/ml/predict?home=${home}&away=${away}`;
      else if (type === "ou")
        url = `/api/nhl/ou/predict?home=${home}&away=${away}&ou=${ou}`;
      else
        url = `/api/nhl/spread/predict?home=${home}&away=${away}&spread=${spread}`;

      const res = await apiFetch(url);
      const data = await res.json();
      setResults((r) => ({ ...r, [type]: data.predictions?.[0] }));
    } catch {
      setResults((r) => ({ ...r, [type]: null }));
    } finally {
      setLoading((l) => ({ ...l, [type]: false }));
    }
  };

  const selectStyle = {
    flex: 1,
    padding: "12px 16px",
    background: "var(--bg)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text)",
    fontSize: 14,
    outline: "none",
    appearance: "none",
    cursor: "pointer",
  };

  const numInput = {
    width: 100,
    padding: "10px 12px",
    background: "var(--bg)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text)",
    fontSize: 14,
    outline: "none",
    textAlign: "center",
  };

  const btnStyle = (active) => ({
    padding: "10px 24px",
    border: "none",
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 700,
    cursor: "pointer",
    background: active ? "var(--accent)" : "var(--surface)",
    color: active ? "var(--bg)" : "var(--text-muted)",
    opacity: active ? 1 : 0.6,
  });

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: "40px 20px" }}>
      <h1
        style={{
          fontSize: 28,
          fontWeight: 800,
          color: "var(--white)",
          margin: "0 0 8px",
          letterSpacing: -0.5,
        }}
      >
        Custom Predictions
      </h1>
      <p style={{ color: "var(--text-muted)", fontSize: 14, margin: "0 0 32px" }}>
        Select any matchup and get ML-powered probabilities
      </p>

      {/* Team Selection */}
      <div
        style={{
          background: "var(--surface)",
          borderRadius: 12,
          border: "1px solid var(--border)",
          padding: 24,
          marginBottom: 16,
        }}
      >
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <select value={away} onChange={(e) => setAway(e.target.value)} style={selectStyle}>
            <option value="">Away Team</option>
            {TEAMS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <span
            style={{
              color: "var(--text-dim)",
              fontWeight: 800,
              fontSize: 13,
              letterSpacing: 2,
            }}
          >
            @
          </span>
          <select value={home} onChange={(e) => setHome(e.target.value)} style={selectStyle}>
            <option value="">Home Team</option>
            {TEAMS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Prediction cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* Moneyline */}
        <PredictionCard
          title="Moneyline"
          subtitle="Win probability"
          onPredict={() => predict("ml")}
          canPredict={home && away}
          loading={loading.ml}
        >
          {results.ml && (
            <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
              <ProbBar label={away} prob={results.ml[0] * 100} color="var(--red)" />
              <ProbBar label={home} prob={results.ml[1] * 100} color="var(--accent)" reverse />
            </div>
          )}
        </PredictionCard>

        {/* Spread */}
        <PredictionCard
          title="Spread"
          subtitle="Cover probability"
          onPredict={() => predict("spread")}
          canPredict={home && away && spread}
          loading={loading.spread}
          input={
            <input
              type="number"
              step="0.5"
              placeholder="±1.5"
              value={spread}
              onChange={(e) => setSpread(e.target.value)}
              style={numInput}
            />
          }
        >
          {results.spread && (
            <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
              <ProbBar
                label={`${away} ${-spread}`}
                prob={results.spread[0] * 100}
                color="var(--red)"
              />
              <ProbBar
                label={`${home} ${spread}`}
                prob={results.spread[1] * 100}
                color="var(--accent)"
                reverse
              />
            </div>
          )}
        </PredictionCard>

        {/* Over/Under */}
        <PredictionCard
          title="Over/Under"
          subtitle="Total goals probability"
          onPredict={() => predict("ou")}
          canPredict={home && away && ou}
          loading={loading.ou}
          input={
            <input
              type="number"
              step="0.5"
              placeholder="5.5"
              value={ou}
              onChange={(e) => setOu(e.target.value)}
              style={numInput}
            />
          }
        >
          {results.ou && (
            <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
              <ProbBar label="Under" prob={results.ou[0] * 100} color="var(--red)" />
              <ProbBar label="Over" prob={results.ou[1] * 100} color="var(--green)" reverse />
            </div>
          )}
        </PredictionCard>
      </div>
    </div>
  );
}

function PredictionCard({ title, subtitle, onPredict, canPredict, loading, input, children }) {
  return (
    <div
      style={{
        background: "var(--surface)",
        borderRadius: 12,
        border: "1px solid var(--border)",
        padding: 20,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div>
            <div style={{ color: "var(--white)", fontWeight: 700, fontSize: 15 }}>{title}</div>
            <div style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 2 }}>
              {subtitle}
            </div>
          </div>
          {input}
        </div>
        <button
          onClick={onPredict}
          disabled={!canPredict || loading}
          style={{
            padding: "10px 24px",
            border: "none",
            borderRadius: 8,
            fontSize: 13,
            fontWeight: 700,
            cursor: "pointer",
            background: canPredict ? "var(--accent)" : "var(--surface)",
            color: canPredict ? "var(--bg)" : "var(--text-muted)",
            opacity: canPredict ? 1 : 0.6,
          }}
        >
          {loading ? "..." : "Predict"}
        </button>
      </div>
      {children}
    </div>
  );
}
