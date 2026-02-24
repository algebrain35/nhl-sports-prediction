export default function StatusBadge({ status, detail }) {
  let bg, color, text;

  if (status === "in") {
    bg = "var(--green-dim)";
    color = "var(--green)";
    text = detail || "LIVE";
  } else if (status === "post") {
    bg = "var(--red-dim)";
    color = "var(--red)";
    text = "FINAL";
  } else {
    bg = "var(--accent-dim)";
    color = "var(--accent)";
    text = detail || "SCHEDULED";
  }

  return (
    <span
      style={{
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: 1.2,
        textTransform: "uppercase",
        padding: "3px 8px",
        borderRadius: 4,
        background: bg,
        color,
      }}
    >
      {text}
    </span>
  );
}
