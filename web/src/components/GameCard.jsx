import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../api/client";
import { ESPN_TO_BACKEND } from "../api/constants";
import { useAuth } from "../hooks/useAuth";
import StatusBadge from "./StatusBadge";
import ProbBar from "./ProbBar";

function formatTime(dateStr) {
  return new Date(dateStr).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function GameCard({ game, isSelected, onSelect }) {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const homeBackend = ESPN_TO_BACKEND[game.home.abbr] || game.home.abbr;
  const awayBackend = ESPN_TO_BACKEND[game.away.abbr] || game.away.abbr;

  const fetchPrediction = async () => {
    /* if (!user?.is_subscribed) {
      navigate("/pricing");
      return;
    } */
    setLoading(true);
    try {
      const res = await apiFetch(
        `/api/nhl/ml/predict?home=${homeBackend}&away=${awayBackend}`
      );
      const data = await res.json();
      console.log(data);
      if (data.predictions?.[0]) {
        const [awayWin, homeWin] = data.predictions[0];
        setPrediction({ homeWin: homeWin * 100, awayWin: awayWin * 100 });
      }
    } catch (e) {
      console.error("Prediction failed:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: "var(--surface)",
        borderRadius: 12,
        border: `1px solid ${isSelected ? "rgba(34,211,238,0.25)" : "var(--border)"}`,
        overflow: "hidden",
        transition: "all 0.2s",
        boxShadow: isSelected ? "0 0 30px var(--accent-glow)" : "none",
      }}
    >
      {/* Main row */}
      <div
        onClick={onSelect}
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "center",
          padding: "20px 24px",
          cursor: "pointer",
        }}
      >
        {/* Away */}
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <img
            src={game.away.logo}
            alt=""
            style={{ width: 44, height: 44, objectFit: "contain" }}
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
          <div>
            <div style={{ color: "var(--white)", fontWeight: 700, fontSize: 16 }}>
              {game.away.name}
            </div>
            <div style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 2 }}>
              {game.away.record}
            </div>
          </div>
          {game.status !== "pre" && (
            <div
              style={{
                fontSize: 28,
                fontWeight: 800,
                color: "var(--white)",
                marginLeft: "auto",
              }}
            >
              {game.away.score}
            </div>
          )}
        </div>

        {/* Center */}
        <div style={{ textAlign: "center", padding: "0 32px" }}>
          <StatusBadge
            status={game.status}
            detail={
              game.status === "pre"
                ? formatTime(game.date)
                : game.status === "in"
                ? `P${game.period} ${game.clock}`
                : null
            }
          />
          {game.status === "pre" && (
            <div style={{ color: "var(--text-dim)", fontSize: 11, marginTop: 6 }}>
              {new Date(game.date).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              })}
            </div>
          )}
        </div>

        {/* Home */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            justifyContent: "flex-end",
          }}
        >
          {game.status !== "pre" && (
            <div
              style={{
                fontSize: 28,
                fontWeight: 800,
                color: "var(--white)",
                marginRight: "auto",
              }}
            >
              {game.home.score}
            </div>
          )}
          <div style={{ textAlign: "right" }}>
            <div style={{ color: "var(--white)", fontWeight: 700, fontSize: 16 }}>
              {game.home.name}
            </div>
            <div style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 2 }}>
              {game.home.record}
            </div>
          </div>
          <img
            src={game.home.logo}
            alt=""
            style={{ width: 44, height: 44, objectFit: "contain" }}
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
        </div>
      </div>

      {/* Expanded prediction panel */}
      {isSelected && (
        <div
          className="fade-in"
          style={{
            borderTop: "1px solid var(--border)",
            padding: "16px 24px",
            background: "rgba(0,0,0,0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {prediction ? (
            <div
              style={{
                display: "flex",
                gap: 24,
                alignItems: "center",
                width: "100%",
              }}
            >
              <ProbBar
                label={game.away.name}
                prob={prediction.awayWin}
                color="var(--red)"
              />
              <div
                style={{
                  color: "var(--text-dim)",
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: 1,
                }}
              >
                ML
              </div>
              <ProbBar
                label={game.home.name}
                prob={prediction.homeWin}
                color="var(--accent)"
                reverse
              />
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                width: "100%",
              }}
            >
              <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                {user?.is_subscribed
                  ? "Get AI win probabilities for this matchup"
                  : "Subscribe to unlock predictions"}
              </div>
              <button
                onClick={fetchPrediction}
                disabled={loading}
                style={{
                  marginLeft: "auto",
                  padding: "8px 20px",
                  background: "var(--accent)",
                  color: "var(--bg)",
                  border: "none",
                  borderRadius: 6,
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: "pointer",
                  opacity: loading ? 0.5 : 1,
                }}
              >
                {loading ? "Loading..." : "Predict"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
