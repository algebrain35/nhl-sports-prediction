export default function ProbBar({ label, prob, color, reverse = false }) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: 4,
        alignItems: reverse ? "flex-end" : "flex-start",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          width: "100%",
          fontSize: 12,
        }}
      >
        {reverse ? (
          <>
            <span style={{ color: "var(--text-dim)" }}>{prob.toFixed(1)}%</span>
            <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>{label}</span>
          </>
        ) : (
          <>
            <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>{label}</span>
            <span style={{ color: "var(--text-dim)" }}>{prob.toFixed(1)}%</span>
          </>
        )}
      </div>
      <div
        style={{
          width: "100%",
          height: 4,
          background: "var(--bg)",
          borderRadius: 2,
          overflow: "hidden",
          display: "flex",
          justifyContent: reverse ? "flex-end" : "flex-start",
        }}
      >
        <div
          style={{
            width: `${prob}%`,
            height: "100%",
            borderRadius: 2,
            background: color,
            transition: "width 0.8s ease",
          }}
        />
      </div>
    </div>
  );
}
