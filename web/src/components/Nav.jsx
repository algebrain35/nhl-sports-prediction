import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import AuthDropdown from "./AuthDropdown";

const pages = [
  { path: "/", label: "Games" },
  { path: "/predict", label: "Predictions" },
  { path: "/pricing", label: "Pricing" },
  { path: "/about", label: "About" },
];

export default function Nav() {
  const { user } = useAuth();
  const [showAuth, setShowAuth] = useState(false);
  const location = useLocation();

  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 32px",
        height: 64,
        background: "rgba(10,14,23,0.85)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      {/* Logo */}
      <Link
        to="/"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          textDecoration: "none",
        }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: "linear-gradient(135deg, var(--accent), #818cf8)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 16,
            fontWeight: 900,
            color: "var(--bg)",
          }}
        >
          R
        </div>
        <span
          style={{
            fontSize: 18,
            fontWeight: 700,
            color: "var(--white)",
            letterSpacing: -0.5,
          }}
        >
          Radon
        </span>
      </Link>

      {/* Nav links */}
      <div style={{ display: "flex", gap: 4 }}>
        {pages.map((p) => {
          const active = location.pathname === p.path;
          return (
            <Link
              key={p.path}
              to={p.path}
              style={{
                background: active ? "var(--accent-dim)" : "transparent",
                color: active ? "var(--accent)" : "var(--text-muted)",
                border: "none",
                borderRadius: 6,
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 600,
                textDecoration: "none",
                transition: "all 0.2s",
              }}
            >
              {p.label}
            </Link>
          );
        })}
      </div>

      {/* Account button */}
      <div style={{ position: "relative" }}>
        <button
          onClick={() => setShowAuth(!showAuth)}
          style={{
            background: user ? "var(--accent-dim)" : "var(--surface)",
            color: user ? "var(--accent)" : "var(--text-muted)",
            border: user ? "none" : "1px solid var(--border)",
            borderRadius: 8,
            padding: "8px 16px",
            fontSize: 13,
            fontWeight: 600,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
          {user ? user.first_name : "Sign In"}
        </button>

        {showAuth && <AuthDropdown onClose={() => setShowAuth(false)} />}
      </div>
    </nav>
  );
}
