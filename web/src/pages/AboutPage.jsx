const STEPS = [
  {
    n: "01",
    title: "Data Collection",
    desc: "Game-by-game stats scraped from MoneyPuck spanning 2008–2025",
  },
  {
    n: "02",
    title: "Feature Engineering",
    desc: "Rolling EMAs, differentials, momentum signals, and Elo ratings",
  },
  {
    n: "03",
    title: "XGBoost Model",
    desc: "Walk-forward validated with time-series cross-validation",
  },
  {
    n: "04",
    title: "Calibrated Probabilities",
    desc: "Isotonic regression for well-calibrated output probabilities",
  },
];

export default function AboutPage() {
  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: "60px 20px" }}>
      <h1
        style={{
          fontSize: 28,
          fontWeight: 800,
          color: "var(--white)",
          margin: "0 0 24px",
          letterSpacing: -0.5,
        }}
      >
        About Radon
      </h1>

      <div
        style={{
          background: "var(--surface)",
          borderRadius: 12,
          border: "1px solid var(--border)",
          padding: 28,
          marginBottom: 16,
        }}
      >
        <p style={{ color: "var(--text-muted)", fontSize: 15, lineHeight: 1.8, margin: 0 }}>
          Project Radon uses machine learning to generate data-driven NHL
          predictions. Built to replace emotional betting with mathematical
          precision, the model analyzes expected goals, shot quality, Elo
          ratings, and dozens of advanced metrics to estimate win probabilities
          for every matchup.
        </p>
      </div>

      <div
        style={{
          background: "var(--surface)",
          borderRadius: 12,
          border: "1px solid var(--border)",
          padding: 28,
        }}
      >
        <h3
          style={{
            color: "var(--white)",
            fontWeight: 700,
            fontSize: 16,
            margin: "0 0 16px",
          }}
        >
          How it works
        </h3>
        {STEPS.map((step, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              gap: 16,
              padding: "14px 0",
              borderTop: i > 0 ? "1px solid var(--border)" : "none",
            }}
          >
            <span
              style={{
                color: "var(--accent)",
                fontWeight: 800,
                fontSize: 13,
                fontFamily: "monospace",
                minWidth: 24,
              }}
            >
              {step.n}
            </span>
            <div>
              <div style={{ color: "var(--white)", fontWeight: 600, fontSize: 14 }}>
                {step.title}
              </div>
              <div style={{ color: "var(--text-dim)", fontSize: 13, marginTop: 2 }}>
                {step.desc}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
