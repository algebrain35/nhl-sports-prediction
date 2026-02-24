import { useState, useEffect } from "react";
import { ESPN_SCOREBOARD } from "../api/constants";
import { useAuth } from "../hooks/useAuth";
import GameCard from "../components/GameCard";

export default function GamesPage() {
  const { user } = useAuth();
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedGame, setSelectedGame] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(ESPN_SCOREBOARD);
        const data = await res.json();
        const events = data.events || [];

        setGames(
          events.map((ev) => {
            const comp = ev.competitions[0];
            const home = comp.competitors.find((c) => c.homeAway === "home");
            const away = comp.competitors.find((c) => c.homeAway === "away");
            const status = comp.status;

            return {
              id: ev.id,
              name: ev.shortName,
              date: comp.date,
              status: status.type.state, // "pre" | "in" | "post"
              statusDetail: status.type.shortDetail,
              period: status.period,
              clock: status.displayClock,
              home: {
                name: home.team.displayName,
                abbr: home.team.abbreviation,
                logo: home.team.logo,
                score: home.score,
                record: home.records?.[0]?.summary || "",
              },
              away: {
                name: away.team.displayName,
                abbr: away.team.abbreviation,
                logo: away.team.logo,
                score: away.score,
                record: away.records?.[0]?.summary || "",
              },
            };
          })
        );
      } catch (e) {
        console.error("ESPN fetch failed:", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            fontSize: 28,
            fontWeight: 800,
            color: "var(--white)",
            margin: 0,
            letterSpacing: -0.5,
          }}
        >
          Today's Games
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 6 }}>
          {new Date().toLocaleDateString("en-US", {
            weekday: "long",
            month: "long",
            day: "numeric",
            year: "numeric",
          })}
        </p>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 60, color: "var(--text-dim)" }}>
          <div
            style={{
              width: 32,
              height: 32,
              border: "3px solid var(--border)",
              borderTopColor: "var(--accent)",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
              margin: "0 auto 16px",
            }}
          />
          Loading games...
        </div>
      ) : games.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: 60,
            color: "var(--text-dim)",
            background: "var(--surface)",
            borderRadius: 12,
            border: "1px solid var(--border)",
          }}
        >
          <div style={{ fontSize: 40, marginBottom: 12 }}>🏒</div>
          No games scheduled today
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {games.map((game) => (
            <GameCard
              key={game.id}
              game={game}
              isSelected={selectedGame === game.id}
              onSelect={() =>
                setSelectedGame(selectedGame === game.id ? null : game.id)
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
