import { useState } from "react";
import { apiFetch } from "../api/client";
import { STRIPE_PRICE_ID } from "../api/constants";
import { useAuth } from "../hooks/useAuth";

const FEATURES = [
  "AI-powered moneyline predictions",
  "Spread & over/under analysis",
  "Real-time game integration",
  "Advanced analytics dashboard",
];

export default function PricingPage() {
  const { user, refresh } = useAuth();
  const [loading, setLoading] = useState(false);

  const handleSubscribe = async () => {
    if (!user) {
      alert("Please sign in first.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiFetch("/api/stripe/subscription", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: user.email, price_id: STRIPE_PRICE_ID }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        setLoading(false);
        return;
      }

      // eslint-disable-next-line no-undef
      const stripe = Stripe(data.publishableKey);
      const elements = stripe.elements({ clientSecret: data.clientSecret });
      const paymentEl = elements.create("payment");

      const container = document.getElementById("stripe-mount");
      if (container) {
        container.style.display = "block";
        paymentEl.mount("#stripe-mount");

        window.__stripeConfirm = async () => {
          const { error: subErr } = await elements.submit();
          if (subErr) {
            alert(subErr.message);
            return;
          }
          const { error, paymentIntent } = await stripe.confirmPayment({
            elements,
            clientSecret: data.clientSecret,
            confirmParams: { return_url: window.location.href },
            redirect: "if_required",
          });
          if (error) {
            alert(error.message);
          } else if (paymentIntent?.status === "succeeded") {
            alert("Subscribed successfully!");
            refresh();
          }
        };
        document.getElementById("stripe-confirm-btn").style.display = "block";
      }
    } catch (e) {
      alert("Subscription error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!confirm("Cancel your subscription?")) return;
    try {
      const res = await apiFetch("/api/stripe/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: user.email }),
      });
      const data = await res.json();
      if (data.success) {
        alert("Subscription cancelled.");
        refresh();
      } else {
        alert("Failed to cancel subscription.");
      }
    } catch {
      alert("Error cancelling subscription.");
    }
  };

  return (
    <div style={{ maxWidth: 480, margin: "0 auto", padding: "60px 20px", textAlign: "center" }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 700,
          letterSpacing: 2,
          color: "var(--accent)",
          marginBottom: 12,
          textTransform: "uppercase",
        }}
      >
        Premium
      </div>
      <h1
        style={{
          fontSize: 32,
          fontWeight: 800,
          color: "var(--white)",
          margin: "0 0 8px",
          letterSpacing: -0.5,
        }}
      >
        Unlock Predictions
      </h1>
      <p
        style={{
          color: "var(--text-muted)",
          fontSize: 15,
          margin: "0 0 32px",
          lineHeight: 1.6,
        }}
      >
        Get AI-powered win probabilities for every NHL matchup
      </p>

      <div
        style={{
          background: "var(--surface)",
          borderRadius: 16,
          border: "1px solid var(--border)",
          padding: 32,
          textAlign: "left",
        }}
      >
        {/* Price */}
        <div style={{ display: "flex", alignItems: "baseline", gap: 4, marginBottom: 24 }}>
          <span style={{ fontSize: 48, fontWeight: 800, color: "var(--white)" }}>$9.99</span>
          <span style={{ color: "var(--text-dim)", fontSize: 14 }}>/month</span>
        </div>

        {/* Features */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 28 }}>
          {FEATURES.map((f, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--accent)"
                strokeWidth="2.5"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span style={{ color: "var(--text-muted)", fontSize: 14 }}>{f}</span>
            </div>
          ))}
        </div>

        {/* Action */}
        {user?.is_subscribed ? (
          <div>
            <div
              style={{
                padding: "12px 0",
                textAlign: "center",
                borderRadius: 8,
                background: "var(--green-dim)",
                color: "var(--green)",
                fontSize: 14,
                fontWeight: 700,
                marginBottom: 12,
              }}
            >
              Active Subscription
            </div>
            <button
              onClick={handleCancel}
              style={{
                width: "100%",
                padding: "12px 0",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: 8,
                color: "var(--red)",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Cancel Subscription
            </button>
          </div>
        ) : (
          <div>
            <button
              onClick={handleSubscribe}
              disabled={loading}
              style={{
                width: "100%",
                padding: "14px 0",
                border: "none",
                borderRadius: 8,
                background: "linear-gradient(135deg, var(--accent), #818cf8)",
                color: "var(--bg)",
                fontSize: 15,
                fontWeight: 800,
                cursor: "pointer",
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading ? "Loading..." : "Subscribe Now"}
            </button>
            <div id="stripe-mount" style={{ display: "none", marginTop: 16 }} />
            <button
              id="stripe-confirm-btn"
              onClick={() => window.__stripeConfirm?.()}
              style={{
                display: "none",
                width: "100%",
                padding: "12px 0",
                marginTop: 12,
                background: "var(--accent)",
                color: "var(--bg)",
                border: "none",
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              Confirm Payment
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
