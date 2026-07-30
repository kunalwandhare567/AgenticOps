/**
 * HumanGatePanel.jsx
 * ===================
 * AI Decision Validation Panel — the Human Gate frontend.
 *
 * What it does:
 *   - Polls GET /api/human-gate/pending every 2 seconds
 *   - Shows a pulsing badge when reviews are WAITING
 *   - Opens a full modal with all AI recommendation context
 *   - Countdown timer ticks down to 0 (auto-approve)
 *   - Approve / Reject buttons POST to /api/human-gate/decision/{id}
 *   - Shows confirmation state and auto-closes after decision
 *
 * Props: none (self-contained, manages its own state)
 */

import React, { useState, useEffect, useCallback, useRef } from "react";

const BACKEND_URL = "http://localhost:8080";
const POLL_INTERVAL_MS = 2000;    // poll every 2s for new reviews
const TIMER_TICK_MS    = 100;     // countdown timer resolution

// Severity colour tokens
const SEV_COLORS = {
  P1: { bg: "#ff2d55", text: "#fff", label: "Critical" },
  P2: { bg: "#ff6b35", text: "#fff", label: "High"     },
  P3: { bg: "#ffd60a", text: "#000", label: "Moderate" },
  P4: { bg: "#30d158", text: "#000", label: "Low"      },
};

const getSevColor = (sev) => SEV_COLORS[sev] || { bg: "#8e8e93", text: "#fff", label: sev };

// Format seconds remaining
const fmtSecs = (s) => {
  if (s <= 0) return "0.0s";
  return `${s.toFixed(1)}s`;
};

// Confidence bar colour
const confColor = (c) => {
  if (c >= 0.9) return "#ff2d55";
  if (c >= 0.75) return "#ff6b35";
  if (c >= 0.5)  return "#ffd60a";
  return "#30d158";
};

// ─── Main component ──────────────────────────────────────────────────────────

export default function HumanGatePanel() {
  // ── State ─────────────────────────────────────────────────────────────────
  const [pendingReviews, setPendingReviews] = useState([]);
  const [activeReview,   setActiveReview]   = useState(null);  // full review object
  const [panelOpen,      setPanelOpen]      = useState(false);
  const [decisionState,  setDecisionState]  = useState(null);  // {decision, message}
  const [timeRemaining,  setTimeRemaining]  = useState(0);     // seconds
  const [operatorName,   setOperatorName]   = useState("Admin");
  const [rejectReason,   setRejectReason]   = useState("");
  const [showRejectInput, setShowRejectInput] = useState(false);
  const [isSubmitting,   setIsSubmitting]   = useState(false);

  const timerRef   = useRef(null);
  const pollRef    = useRef(null);

  // ── Polling for pending reviews ───────────────────────────────────────────
  const fetchPending = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/human-gate/pending`);
      if (res.ok) {
        const data = await res.json();
        setPendingReviews(data);

        // Auto-open panel if a new review arrived and no panel is open
        if (data.length > 0 && !panelOpen && !activeReview) {
          handleOpenReview(data[0].review_id);
        }
      }
    } catch (_) {
      // Backend not running — silently ignore
    }
  }, [panelOpen, activeReview]);

  useEffect(() => {
    fetchPending();
    pollRef.current = setInterval(fetchPending, POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [fetchPending]);

  // ── Countdown timer ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeReview || decisionState) return;

    const computeRemaining = () => {
      const exp = new Date(activeReview.expires_at);
      const now = new Date();
      return Math.max(0, (exp - now) / 1000);
    };

    setTimeRemaining(computeRemaining());

    timerRef.current = setInterval(() => {
      const rem = computeRemaining();
      setTimeRemaining(rem);
      if (rem <= 0) {
        clearInterval(timerRef.current);
        // Show auto-approved state in UI (backend already handled it)
        setDecisionState({ decision: "AUTO_APPROVED", message: "Auto-approved — timeout reached." });
        setTimeout(() => closePanel(), 2500);
      }
    }, TIMER_TICK_MS);

    return () => clearInterval(timerRef.current);
  }, [activeReview, decisionState]);

  // ── Open a specific review ────────────────────────────────────────────────
  const handleOpenReview = useCallback(async (reviewId) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/human-gate/review/${reviewId}`);
      if (res.ok) {
        const data = await res.json();
        setActiveReview(data);
        setDecisionState(null);
        setRejectReason("");
        setShowRejectInput(false);
        setPanelOpen(true);
      }
    } catch (_) {}
  }, []);

  // ── Submit decision ───────────────────────────────────────────────────────
  const submitDecision = useCallback(async (decision) => {
    if (!activeReview || isSubmitting) return;
    if (decision === "REJECTED" && !rejectReason.trim()) {
      setShowRejectInput(true);
      return;
    }

    setIsSubmitting(true);
    clearInterval(timerRef.current);

    try {
      const res = await fetch(
        `${BACKEND_URL}/api/human-gate/decision/${activeReview.review_id}`,
        {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({
            decision: decision,
            operator: operatorName || "Admin",
            reason:   decision === "REJECTED" ? rejectReason : `Approved by ${operatorName}.`,
          }),
        }
      );

      if (res.ok) {
        const data = await res.json();
        const msg = decision === "APPROVED"
          ? `✅ Escalation approved — ${activeReview.new_severity} confirmed.`
          : `❌ Escalation rejected — keeping ${activeReview.old_severity}.`;
        setDecisionState({ decision, message: msg });
        fetchPending();
        setTimeout(() => closePanel(), 2800);
      } else {
        setDecisionState({ decision: "ERROR", message: "Submission failed. Retry." });
      }
    } catch (_) {
      setDecisionState({ decision: "ERROR", message: "Network error. Retry." });
    } finally {
      setIsSubmitting(false);
    }
  }, [activeReview, operatorName, rejectReason, isSubmitting]);

  // ── Close panel ───────────────────────────────────────────────────────────
  const closePanel = () => {
    setPanelOpen(false);
    setActiveReview(null);
    setDecisionState(null);
    setTimeRemaining(0);
    setShowRejectInput(false);
    setRejectReason("");
    clearInterval(timerRef.current);
  };

  // ── Derived values ────────────────────────────────────────────────────────
  const pendingCount  = pendingReviews.length;
  const timeoutSecs   = activeReview?.timeout_seconds || 2;
  const progressPct   = activeReview
    ? Math.max(0, Math.min(100, ((timeoutSecs - timeRemaining) / timeoutSecs) * 100))
    : 0;

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <>
      {/* ── Floating badge ─────────────────────────────────────────────── */}
      {pendingCount > 0 && !panelOpen && (
        <button
          id="human-gate-badge"
          onClick={() => handleOpenReview(pendingReviews[0].review_id)}
          style={styles.badge}
          title="Human Gate: AI escalation awaiting review"
        >
          <span style={styles.badgePulse} />
          <span style={{ fontSize: 18 }}>🛡️</span>
          <span style={styles.badgeText}>
            {pendingCount} Review{pendingCount > 1 ? "s" : ""} Pending
          </span>
        </button>
      )}

      {/* ── Modal overlay ──────────────────────────────────────────────── */}
      {panelOpen && activeReview && (
        <div id="human-gate-overlay" style={styles.overlay}>
          <div id="human-gate-modal" style={styles.modal}>

            {/* ─ Header ─ */}
            <div style={styles.modalHeader}>
              <div style={styles.headerLeft}>
                <span style={styles.shieldIcon}>🛡️</span>
                <div>
                  <div style={styles.headerTitle}>AI ESCALATION REVIEW</div>
                  <div style={styles.headerSub}>
                    Human Gate — Validate before applying severity change
                  </div>
                </div>
              </div>
              <button id="human-gate-close" style={styles.closeBtn} onClick={closePanel}>✕</button>
            </div>

            {/* ─ Incident identity row ─ */}
            <div style={styles.identityRow}>
              <span style={styles.incidentId}>{activeReview.incident_id}</span>
              <span style={styles.failureLabel}>{activeReview.failure_label}</span>
              {activeReview.is_large_jump ? (
                <span style={styles.largeJumpTag}>⚠ LARGE JUMP</span>
              ) : null}
            </div>

            {/* ─ Severity change arrow ─ */}
            <div style={styles.severityRow}>
              <div style={styles.sevBox(getSevColor(activeReview.old_severity))}>
                <div style={styles.sevLabel}>CURRENT</div>
                <div style={styles.sevValue}>{activeReview.old_severity}</div>
                <div style={styles.sevSubLabel}>{getSevColor(activeReview.old_severity).label}</div>
              </div>
              <div style={styles.arrow}>→</div>
              <div style={styles.sevBox(getSevColor(activeReview.new_severity))}>
                <div style={styles.sevLabel}>AI RECOMMENDS</div>
                <div style={styles.sevValue}>{activeReview.new_severity}</div>
                <div style={styles.sevSubLabel}>{getSevColor(activeReview.new_severity).label}</div>
              </div>
            </div>

            {/* ─ AI Evidence grid ─ */}
            <div style={styles.evidenceGrid}>
              <EvidenceCell label="Confidence" value={`${(activeReview.confidence * 100).toFixed(0)}%`}
                accent={confColor(activeReview.confidence)} />
              <EvidenceCell label="Time to Failure"
                value={activeReview.ttf_seconds > 0 ? `${activeReview.ttf_seconds.toFixed(0)}s` : "N/A"}
                accent={activeReview.ttf_seconds > 0 && activeReview.ttf_seconds < 30 ? "#ff2d55" : "#ff9f0a"} />
              <EvidenceCell label="Impact Band"   value={activeReview.impact_band}   accent="#6e6af0" />
              <EvidenceCell label="Urgency Band"  value={activeReview.urgency_band}  accent="#ff6b35" />
            </div>

            {/* ─ Root cause ─ */}
            <div style={styles.rootCauseBox}>
              <div style={styles.sectionLabel}>ROOT CAUSE / REASON</div>
              <p style={styles.rootCauseText}>{activeReview.root_cause}</p>
            </div>

            {/* ─ Escalation summary ─ */}
            <div style={styles.escalationSummary}>
              {activeReview.escalation_summary}
            </div>

            {/* ─ Countdown timer ─ */}
            {!decisionState && (
              <div style={styles.timerSection}>
                <div style={styles.timerHeader}>
                  <span style={{ color: "#ff9f0a", fontWeight: 600 }}>⏱ Auto-approve in:</span>
                  <span style={styles.timerValue(timeRemaining)}>{fmtSecs(timeRemaining)}</span>
                </div>
                <div style={styles.progressTrack}>
                  <div style={styles.progressBar(progressPct)} />
                </div>
              </div>
            )}

            {/* ─ Decision result ─ */}
            {decisionState && (
              <div style={styles.decisionResult(decisionState.decision)}>
                {decisionState.message}
              </div>
            )}

            {/* ─ Operator input ─ */}
            {!decisionState && (
              <div style={styles.operatorRow}>
                <input
                  id="human-gate-operator"
                  style={styles.operatorInput}
                  value={operatorName}
                  onChange={(e) => setOperatorName(e.target.value)}
                  placeholder="Operator name"
                />
              </div>
            )}

            {/* ─ Reject reason (shown after first REJECT click) ─ */}
            {showRejectInput && !decisionState && (
              <div style={{ marginBottom: 12 }}>
                <textarea
                  id="human-gate-reject-reason"
                  style={styles.rejectTextarea}
                  rows={2}
                  placeholder="Reason for rejection (required)…"
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                />
              </div>
            )}

            {/* ─ Action buttons ─ */}
            {!decisionState && (
              <div style={styles.actionRow}>
                <button
                  id="human-gate-approve"
                  style={styles.approveBtn}
                  disabled={isSubmitting}
                  onClick={() => submitDecision("APPROVED")}
                >
                  ✅ APPROVE ESCALATION
                </button>
                <button
                  id="human-gate-reject"
                  style={styles.rejectBtn}
                  disabled={isSubmitting}
                  onClick={() => submitDecision("REJECTED")}
                >
                  ❌ REJECT — KEEP {activeReview.old_severity}
                </button>
              </div>
            )}

            {/* ─ Footer note ─ */}
            <div style={styles.footerNote}>
              Every decision is recorded in the audit log for offline model learning.
            </div>

          </div>
        </div>
      )}
    </>
  );
}

// ─── Evidence Cell sub-component ─────────────────────────────────────────────

function EvidenceCell({ label, value, accent }) {
  return (
    <div style={styles.evidenceCell(accent)}>
      <div style={styles.evidenceCellLabel}>{label}</div>
      <div style={styles.evidenceCellValue(accent)}>{value}</div>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = {
  // ── Badge ──
  badge: {
    position:     "fixed",
    top:          "20px",
    right:        "20px",
    zIndex:       9999,
    display:      "flex",
    alignItems:   "center",
    gap:          "10px",
    background:   "linear-gradient(135deg, #1c1c2e 0%, #2a1a3a 100%)",
    border:       "2px solid #ff2d55",
    borderRadius: "50px",
    padding:      "10px 20px",
    cursor:       "pointer",
    boxShadow:    "0 0 24px rgba(255,45,85,0.5), 0 4px 16px rgba(0,0,0,0.6)",
    animation:    "hgPulse 1.8s ease-in-out infinite",
  },
  badgePulse: {
    display:      "block",
    width:        "10px",
    height:       "10px",
    borderRadius: "50%",
    background:   "#ff2d55",
    boxShadow:    "0 0 10px #ff2d55",
  },
  badgeText: {
    color:      "#fff",
    fontWeight: 700,
    fontSize:   "14px",
    fontFamily: "'Inter', sans-serif",
  },

  // ── Overlay ──
  overlay: {
    position:        "fixed",
    inset:           0,
    zIndex:          10000,
    display:         "flex",
    alignItems:      "center",
    justifyContent:  "center",
    background:      "rgba(0,0,0,0.72)",
    backdropFilter:  "blur(6px)",
  },

  // ── Modal ──
  modal: {
    background:   "linear-gradient(160deg, #0d0d1a 0%, #131330 100%)",
    border:       "1px solid rgba(255,45,85,0.35)",
    borderRadius: "20px",
    width:        "480px",
    maxWidth:     "95vw",
    maxHeight:    "90vh",
    overflowY:    "auto",
    padding:      "28px",
    boxShadow:    "0 0 60px rgba(255,45,85,0.25), 0 24px 80px rgba(0,0,0,0.8)",
    fontFamily:   "'Inter', -apple-system, sans-serif",
    color:        "#e0e0ef",
  },

  // ── Modal header ──
  modalHeader: {
    display:        "flex",
    justifyContent: "space-between",
    alignItems:     "flex-start",
    marginBottom:   "20px",
  },
  headerLeft: { display: "flex", alignItems: "flex-start", gap: "12px" },
  shieldIcon: { fontSize: "28px", lineHeight: 1 },
  headerTitle: {
    fontSize:   "13px",
    fontWeight: 800,
    color:      "#ff2d55",
    letterSpacing: "0.12em",
    textTransform: "uppercase",
  },
  headerSub: { fontSize: "11px", color: "#8888aa", marginTop: "4px" },
  closeBtn: {
    background:   "transparent",
    border:       "1px solid #333355",
    borderRadius: "8px",
    color:        "#8888aa",
    cursor:       "pointer",
    padding:      "4px 10px",
    fontSize:     "14px",
    transition:   "all 0.2s",
  },

  // ── Identity row ──
  identityRow: {
    display:      "flex",
    alignItems:   "center",
    gap:          "10px",
    marginBottom: "18px",
    flexWrap:     "wrap",
  },
  incidentId: {
    fontSize:     "18px",
    fontWeight:   700,
    color:        "#e0e0ef",
    letterSpacing: "0.02em",
  },
  failureLabel: {
    fontSize:     "13px",
    color:        "#aaaacc",
    background:   "rgba(255,255,255,0.06)",
    borderRadius: "6px",
    padding:      "3px 10px",
  },
  largeJumpTag: {
    fontSize:     "11px",
    fontWeight:   700,
    color:        "#ffd60a",
    background:   "rgba(255,214,10,0.15)",
    border:       "1px solid rgba(255,214,10,0.4)",
    borderRadius: "6px",
    padding:      "3px 10px",
  },

  // ── Severity row ──
  severityRow: {
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
    gap:            "20px",
    marginBottom:   "20px",
  },
  sevBox: (c) => ({
    background:   c.bg,
    borderRadius: "14px",
    padding:      "14px 24px",
    textAlign:    "center",
    minWidth:     "100px",
    boxShadow:    `0 0 20px ${c.bg}55`,
  }),
  sevLabel: { fontSize: "9px", letterSpacing: "0.1em", opacity: 0.8, fontWeight: 700 },
  sevValue: { fontSize: "36px", fontWeight: 900, lineHeight: 1.1 },
  sevSubLabel: { fontSize: "11px", opacity: 0.9, marginTop: "2px" },
  arrow: {
    fontSize:   "28px",
    color:      "#ff9f0a",
    fontWeight: 700,
  },

  // ── Evidence grid ──
  evidenceGrid: {
    display:     "grid",
    gridTemplateColumns: "1fr 1fr",
    gap:         "10px",
    marginBottom: "16px",
  },
  evidenceCell: (accent) => ({
    background:   "rgba(255,255,255,0.04)",
    border:       `1px solid ${accent}44`,
    borderRadius: "10px",
    padding:      "10px 14px",
  }),
  evidenceCellLabel: { fontSize: "10px", color: "#8888aa", letterSpacing: "0.08em", textTransform: "uppercase" },
  evidenceCellValue: (accent) => ({
    fontSize:   "18px",
    fontWeight: 700,
    color:      accent,
    marginTop:  "4px",
  }),

  // ── Root cause ──
  rootCauseBox: {
    background:   "rgba(255,255,255,0.04)",
    border:       "1px solid rgba(255,255,255,0.08)",
    borderRadius: "10px",
    padding:      "12px 16px",
    marginBottom: "12px",
  },
  sectionLabel: { fontSize: "9px", color: "#6666aa", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "6px" },
  rootCauseText: { fontSize: "12px", color: "#c0c0dd", margin: 0, lineHeight: 1.6 },

  // ── Escalation summary ──
  escalationSummary: {
    fontSize:     "12px",
    color:        "#ff9f0a",
    textAlign:    "center",
    marginBottom: "14px",
    fontWeight:   600,
  },

  // ── Timer ──
  timerSection: { marginBottom: "16px" },
  timerHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" },
  timerValue: (rem) => ({
    fontSize:   "22px",
    fontWeight: 800,
    color:      rem < 0.8 ? "#ff2d55" : rem < 1.5 ? "#ff9f0a" : "#30d158",
    fontFamily: "monospace",
  }),
  progressTrack: {
    background:   "rgba(255,255,255,0.1)",
    borderRadius: "4px",
    height:       "5px",
    overflow:     "hidden",
  },
  progressBar: (pct) => ({
    width:        `${pct}%`,
    height:       "100%",
    borderRadius: "4px",
    background:   pct > 80 ? "#ff2d55" : pct > 50 ? "#ff9f0a" : "#30d158",
    transition:   "width 0.1s linear",
  }),

  // ── Decision result ──
  decisionResult: (d) => ({
    background:   d === "APPROVED" ? "rgba(48,209,88,0.15)" : d === "REJECTED" ? "rgba(255,45,85,0.15)" : "rgba(255,159,10,0.15)",
    border:       `1px solid ${d === "APPROVED" ? "#30d158" : d === "REJECTED" ? "#ff2d55" : "#ff9f0a"}`,
    borderRadius: "10px",
    padding:      "14px 16px",
    textAlign:    "center",
    fontWeight:   700,
    fontSize:     "14px",
    color:        d === "APPROVED" ? "#30d158" : d === "REJECTED" ? "#ff2d55" : "#ff9f0a",
    marginBottom: "12px",
  }),

  // ── Operator row ──
  operatorRow: { marginBottom: "12px" },
  operatorInput: {
    width:        "100%",
    background:   "rgba(255,255,255,0.06)",
    border:       "1px solid rgba(255,255,255,0.12)",
    borderRadius: "8px",
    padding:      "8px 14px",
    color:        "#e0e0ef",
    fontSize:     "13px",
    outline:      "none",
    boxSizing:    "border-box",
  },

  // ── Reject textarea ──
  rejectTextarea: {
    width:        "100%",
    background:   "rgba(255,45,85,0.08)",
    border:       "1px solid rgba(255,45,85,0.3)",
    borderRadius: "8px",
    padding:      "8px 14px",
    color:        "#e0e0ef",
    fontSize:     "12px",
    outline:      "none",
    boxSizing:    "border-box",
    resize:       "none",
    fontFamily:   "inherit",
  },

  // ── Action buttons ──
  actionRow: { display: "flex", flexDirection: "column", gap: "10px", marginBottom: "14px" },
  approveBtn: {
    background:   "linear-gradient(135deg, #30d158, #25a244)",
    border:       "none",
    borderRadius: "12px",
    color:        "#000",
    fontWeight:   800,
    fontSize:     "14px",
    padding:      "14px",
    cursor:       "pointer",
    letterSpacing: "0.04em",
    transition:   "all 0.2s",
    boxShadow:    "0 4px 20px rgba(48,209,88,0.35)",
  },
  rejectBtn: {
    background:   "linear-gradient(135deg, #3a0015, #1a000a)",
    border:       "2px solid #ff2d55",
    borderRadius: "12px",
    color:        "#ff2d55",
    fontWeight:   800,
    fontSize:     "14px",
    padding:      "12px",
    cursor:       "pointer",
    letterSpacing: "0.04em",
    transition:   "all 0.2s",
  },

  // ── Footer ──
  footerNote: {
    textAlign:  "center",
    fontSize:   "10px",
    color:      "#555577",
    lineHeight: 1.5,
  },
};
