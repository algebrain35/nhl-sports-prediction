import { useState, useEffect, useRef } from "react";
import { useAuth } from "../hooks/useAuth";

export default function AuthDropdown({ onClose }) {
  const { user, login, register, logout } = useAuth();
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({});
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const ref = useRef();

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const inputStyle = {
    width: "100%",
    padding: "10px 12px",
    background: "var(--bg)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text)",
    fontSize: 13,
    outline: "none",
    boxSizing: "border-box",
  };

  const btnPrimary = {
    width: "100%",
    padding: "10px 0",
    background: "var(--accent)",
    color: "var(--bg)",
    border: "none",
    borderRadius: 6,
    fontSize: 13,
    fontWeight: 700,
    cursor: "pointer",
  };

  const handleLogin = async () => {
    setError("");
    setSuccess("");
    try {
      await login(form.email, form.password);
      setSuccess("Logged in!");
      setTimeout(onClose, 500);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleRegister = async () => {
    setError("");
    setSuccess("");
    try {
      await register(form.first_name, form.last_name, form.email, form.password);
      setSuccess("Registered! Please log in.");
      setMode("login");
    } catch (e) {
      setError(e.message);
    }
  };

  const handleLogout = async () => {
    await logout();
    onClose();
  };

  return (
    <div
      ref={ref}
      className="fade-in"
      style={{
        position: "absolute",
        right: 0,
        top: "calc(100% + 8px)",
        width: 280,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 12,
        padding: 20,
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        zIndex: 200,
      }}
    >
      {user ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ color: "var(--white)", fontWeight: 600, fontSize: 14 }}>
            {user.first_name} {user.last_name}
          </div>
          <div style={{ color: "var(--text-muted)", fontSize: 12 }}>{user.email}</div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "8px 12px",
              borderRadius: 6,
              background: user.is_subscribed ? "var(--green-dim)" : "var(--amber-dim)",
            }}
          >
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: user.is_subscribed ? "var(--green)" : "var(--amber)",
              }}
            />
            <span
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: user.is_subscribed ? "var(--green)" : "var(--amber)",
              }}
            >
              {user.is_subscribed ? "Premium" : "Free"}
            </span>
          </div>
          <button
            onClick={handleLogout}
            style={{
              ...btnPrimary,
              background: "transparent",
              color: "var(--red)",
              border: "1px solid var(--border)",
            }}
          >
            Sign Out
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ color: "var(--white)", fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
            {mode === "login" ? "Welcome back" : "Create account"}
          </div>

          {error && <div style={{ color: "var(--red)", fontSize: 12 }}>{error}</div>}
          {success && <div style={{ color: "var(--green)", fontSize: 12 }}>{success}</div>}

          {mode === "register" && (
            <>
              <input style={inputStyle} placeholder="First Name" onChange={set("first_name")} />
              <input style={inputStyle} placeholder="Last Name" onChange={set("last_name")} />
            </>
          )}
          <input style={inputStyle} placeholder="Email" type="email" onChange={set("email")} />
          <input
            style={inputStyle}
            placeholder="Password"
            type="password"
            onChange={set("password")}
            onKeyDown={(e) =>
              e.key === "Enter" && (mode === "login" ? handleLogin() : handleRegister())
            }
          />

          <button
            style={btnPrimary}
            onClick={mode === "login" ? handleLogin : handleRegister}
          >
            {mode === "login" ? "Sign In" : "Create Account"}
          </button>

          <button
            onClick={() => setMode(mode === "login" ? "register" : "login")}
            style={{
              background: "none",
              border: "none",
              color: "var(--accent)",
              fontSize: 12,
              cursor: "pointer",
              padding: 4,
            }}
          >
            {mode === "login"
              ? "Need an account? Register"
              : "Already have an account? Sign in"}
          </button>
        </div>
      )}
    </div>
  );
}
