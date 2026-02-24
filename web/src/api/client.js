/**
 * Shared API client.
 *
 * In development, Vite proxies /api/* to Flask (http://localhost:5000).
 * In production, requests go to the same origin (or configure API_BASE).
 *
 * All requests include credentials so Flask session cookies work.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function apiFetch(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    credentials: "include",
    ...opts,
  });
  return res;
}

export async function apiJson(path, opts = {}) {
  const res = await apiFetch(path, opts);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || data.message || `Request failed: ${res.status}`);
  }
  return data;
}

export function apiPost(path, body) {
  return apiJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
