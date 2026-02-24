import { useState, useEffect, useCallback, createContext, useContext } from "react";
import { apiFetch, apiPost } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem("currentUser"));
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await apiFetch("/api/user");
      if (res.ok) {
        const u = await res.json();
        setUser(u);
        sessionStorage.setItem("currentUser", JSON.stringify(u));
      } else {
        setUser(null);
        sessionStorage.removeItem("currentUser");
      }
    } catch {
      setUser(null);
      sessionStorage.removeItem("currentUser");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const login = async (email, password) => {
    const data = await apiPost("/api/login", { email, password });
    setUser(data.user);
    sessionStorage.setItem("currentUser", JSON.stringify(data.user));
    return data;
  };

  const register = async (first_name, last_name, email, password) => {
    return apiPost("/api/register", { first_name, last_name, email, password });
  };

  const logout = async () => {
    try {
      await apiFetch("/api/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } catch {
      // logout even if request fails
    }
    setUser(null);
    sessionStorage.removeItem("currentUser");
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, refresh }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
