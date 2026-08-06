import React, { useState, useEffect, useCallback, useRef } from "react";
import { Shield, CheckCircle, XCircle } from "lucide-react";

const BACKEND_URL = "http://localhost:8080";
const POLL_INTERVAL_MS = 2000;
const TIMER_TICK_MS = 100;

const SEV_BADGES = {
  P1: { bg: "#ff2d55", text: "#fff" },
  P2: { bg: "#ff6b35", text: "#fff" },
  P3: { bg: "#ffd60a", text: "#000" },
  P4: { bg: "#30d158", text: "#000" },
};

const getSevStyle = (sev) => SEV_BADGES[sev] || { bg: "#8e8e93", text: "#fff" };

export default function HumanGateSidebarSection() {
  const [pendingReviews, setPendingReviews] = useState([]);
  const [activeReview, setActiveReview] = useState(null);
  const [decisionState, setDecisionState] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [history, setHistory] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [operatorName] = useState("Admin");

  const timerRef = useRef(null);

  const fetchReviewDetails = useCallback(async (reviewId) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/human-gate/review/${reviewId}`);
      if (res.ok) {
        setActiveReview(await res.json());
        setDecisionState(null);
      }
    } catch (err) {
      console.warn("Review fetch error", err);
    }
  }, []);

  // Poll pending reviews & audit history
  const fetchData = useCallback(async () => {
    try {
      // Pending reviews
      const pRes = await fetch(`${BACKEND_URL}/api/human-gate/pending`);
      if (pRes.ok) {
        const pData = await pRes.json();
        setPendingReviews(pData);
        if (pData.length > 0 && (!activeReview || activeReview.review_id !== pData[0].review_id)) {
          fetchReviewDetails(pData[0].review_id);
        }
      }

      // History
      const hRes = await fetch(`${BACKEND_URL}/api/human-gate/history?limit=5`);
      if (hRes.ok) {
        setHistory(await hRes.json());
      }
    } catch (err) {
      console.warn("Human gate fetch error", err);
    }
  }, [activeReview, fetchReviewDetails]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Countdown timer for active review
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
        setDecisionState({ decision: "AUTO_APPROVED", message: "Auto-Approved (Timeout)" });
        setTimeout(() => {
          setActiveReview(null);
          setDecisionState(null);
          fetchData();
        }, 2500);
      }
    }, TIMER_TICK_MS);

    return () => clearInterval(timerRef.current);
  }, [activeReview, decisionState, fetchData]);

  // Submit decision
  const handleDecision = async (decision) => {
    if (!activeReview || isSubmitting) return;
    setIsSubmitting(true);
    clearInterval(timerRef.current);

    try {
      const res = await fetch(
        `${BACKEND_URL}/api/human-gate/decision/${activeReview.review_id}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            decision: decision,
            operator: operatorName,
            reason: decision === "APPROVED" ? `Approved by ${operatorName}` : `Rejected by ${operatorName}`,
          }),
        }
      );

      if (res.ok) {
        const msg = decision === "APPROVED" ? "✅ Escalation Approved" : "❌ Escalation Rejected";
        setDecisionState({ decision, message: msg });
        fetchData();
        setTimeout(() => {
          setActiveReview(null);
          setDecisionState(null);
        }, 2500);
      }
    } catch (err) {
      console.error("Decision submit error", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const timeoutSecs = activeReview?.timeout_seconds || 2;
  const progressPct = activeReview
    ? Math.max(0, Math.min(100, ((timeoutSecs - timeRemaining) / timeoutSecs) * 100))
    : 0;

  return (
    <div className="human-gate-sidebar-section" style={styles.container}>
      <div style={styles.header}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Shield size={14} color="#ff2d55" />
          <span style={styles.headerTitle}>HUMAN GATE APPROVAL</span>
        </div>
        {pendingReviews.length > 0 && (
          <span style={styles.pendingBadge}>
            {pendingReviews.length} PENDING
          </span>
        )}
      </div>

      {/* ACTIVE PENDING REVIEW (INLINE) */}
      {activeReview ? (
        <div style={styles.reviewCard}>
          <div style={styles.cardHeader}>
            <span style={styles.incidentId}>{activeReview.incident_id}</span>
            <span style={styles.failureLabel}>{activeReview.failure_label}</span>
          </div>

          {/* Severity Arrow */}
          <div style={styles.sevRow}>
            <span style={styles.sevChip(getSevStyle(activeReview.old_severity))}>
              {activeReview.old_severity}
            </span>
            <span style={{ color: "#ff9f0a", fontWeight: 700 }}>→</span>
            <span style={styles.sevChip(getSevStyle(activeReview.new_severity))}>
              {activeReview.new_severity}
            </span>
          </div>

          {/* Confidence & TTF */}
          <div style={styles.metaGrid}>
            <div>
              <div style={styles.metaLabel}>Confidence</div>
              <div style={styles.metaVal}>{(activeReview.confidence * 100).toFixed(0)}%</div>
            </div>
            <div>
              <div style={styles.metaLabel}>TTF (Time To Failure)</div>
              <div style={styles.metaVal}>
                {activeReview.ttf_seconds > 0 ? `${activeReview.ttf_seconds.toFixed(0)}s` : "Imminent"}
              </div>
            </div>
          </div>

          {/* Decision state banner or timer */}
          {decisionState ? (
            <div style={styles.decisionBanner(decisionState.decision)}>
              {decisionState.message}
            </div>
          ) : (
            <>
              {/* Countdown Timer */}
              <div style={styles.timerRow}>
                <span style={{ fontSize: "10px", color: "#ff9f0a", fontWeight: 600 }}>
                  ⏱ Auto-approve: {timeRemaining.toFixed(1)}s
                </span>
              </div>
              <div style={styles.progressTrack}>
                <div style={styles.progressBar(progressPct)} />
              </div>

              {/* Action Buttons */}
              <div style={styles.btnRow}>
                <button
                  style={styles.approveBtn}
                  disabled={isSubmitting}
                  onClick={() => handleDecision("APPROVED")}
                >
                  <CheckCircle size={12} /> APPROVE
                </button>
                <button
                  style={styles.rejectBtn}
                  disabled={isSubmitting}
                  onClick={() => handleDecision("REJECTED")}
                >
                  <XCircle size={12} /> REJECT
                </button>
              </div>
            </>
          )}
        </div>
      ) : (
        /* NO PENDING REVIEWS — SHOW STANDBY + AUDIT LOG HISTORY */
        <div>
          <div style={styles.standbyBox}>
            <CheckCircle size={14} color="#30d158" />
            <span style={{ fontSize: "10px", color: "#30d158", fontWeight: 600 }}>
              All AI Escalations Validated
            </span>
          </div>

          {/* Recent Audit Decisions */}
          {history.length > 0 && (
            <div style={{ marginTop: "10px" }}>
              <div style={styles.historyLabel}>Recent Gate Decisions</div>
              <div style={styles.historyList}>
                {history.map((h, i) => {
                  const label = h.failure_label || h.failure_mode?.replace(/_/g, " ") || h.episode_id?.split("_").slice(0, 2).join("_") || h.review_id?.slice(0, 10);
                  const ttfVal = h.ttf_seconds != null && h.ttf_seconds > 0 ? `${Math.round(h.ttf_seconds)}s` : "N/A";
                  return (
                    <div key={i} style={styles.historyItem}>
                      <div style={styles.historyTopRow}>
                        <span style={styles.historyId} title={label}>
                          {label}
                        </span>
                        <span
                          style={{
                            fontSize: "8.5px",
                            fontWeight: 800,
                            padding: "2px 6px",
                            borderRadius: "4px",
                            whiteSpace: "nowrap",
                            flexShrink: 0,
                            background:
                              h.decision === "APPROVED"
                                ? "rgba(48, 209, 88, 0.18)"
                                : h.decision === "REJECTED"
                                ? "rgba(255, 45, 85, 0.18)"
                                : "rgba(255, 214, 10, 0.18)",
                            color:
                              h.decision === "APPROVED"
                                ? "#30d158"
                                : h.decision === "REJECTED"
                                ? "#ff2d55"
                                : "#ffd60a",
                            border: `1px solid ${
                              h.decision === "APPROVED"
                                ? "rgba(48, 209, 88, 0.4)"
                                : h.decision === "REJECTED"
                                ? "rgba(255, 45, 85, 0.4)"
                                : "rgba(255, 214, 10, 0.4)"
                            }`,
                          }}
                        >
                          {h.decision}
                        </span>
                      </div>
                      <div style={styles.historyBottomRow}>
                        <span>
                          {h.old_severity} → {h.final_severity || h.new_severity} ({h.operator || "system"})
                        </span>
                        <span style={styles.historyTtfChip}>
                          TTF: {ttfVal}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

}

const styles = {
  container: {
    margin: "12px 10px",
    padding: "10px",
    background: "rgba(13, 13, 26, 0.75)",
    border: "1px solid rgba(255, 45, 85, 0.25)",
    borderRadius: "10px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "8px",
  },
  headerTitle: {
    fontSize: "9.5px",
    fontWeight: 800,
    color: "#ff2d55",
    letterSpacing: "0.8px",
  },
  pendingBadge: {
    fontSize: "8.5px",
    fontWeight: 800,
    background: "rgba(255, 45, 85, 0.2)",
    border: "1px solid rgba(255, 45, 85, 0.5)",
    color: "#ff2d55",
    padding: "2px 6px",
    borderRadius: "10px",
    animation: "livePulse 1.2s infinite",
  },
  reviewCard: {
    background: "rgba(255, 255, 255, 0.03)",
    border: "1px solid rgba(255, 45, 85, 0.3)",
    borderRadius: "8px",
    padding: "10px",
  },
  cardHeader: {
    display: "flex",
    flexDirection: "column",
    gap: "2px",
    marginBottom: "6px",
  },
  incidentId: {
    fontSize: "10.5px",
    fontWeight: 700,
    color: "#e0e0ef",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    maxWidth: "100%",
    display: "block",
  },
  failureLabel: {
    fontSize: "9px",
    color: "#aaaacc",
    background: "rgba(255,255,255,0.06)",
    padding: "1px 6px",
    borderRadius: "4px",
  },
  sevRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    margin: "8px 0",
  },
  sevChip: (style) => ({
    background: style.bg,
    color: style.text,
    fontSize: "12px",
    fontWeight: 800,
    padding: "4px 10px",
    borderRadius: "6px",
  }),
  metaGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "6px",
    marginBottom: "8px",
    textAlign: "center",
  },
  metaLabel: {
    fontSize: "8px",
    color: "rgba(255,255,255,0.4)",
    textTransform: "uppercase",
  },
  metaVal: {
    fontSize: "11px",
    fontWeight: 700,
    color: "#e0e0ef",
  },
  timerRow: {
    textAlign: "center",
    marginBottom: "4px",
  },
  progressTrack: {
    background: "rgba(255,255,255,0.1)",
    borderRadius: "3px",
    height: "4px",
    overflow: "hidden",
    marginBottom: "8px",
  },
  progressBar: (pct) => ({
    width: `${pct}%`,
    height: "100%",
    background: pct > 80 ? "#ff2d55" : pct > 50 ? "#ff9f0a" : "#30d158",
    transition: "width 0.1s linear",
  }),
  btnRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "6px",
  },
  approveBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "4px",
    background: "#30d158",
    border: "none",
    borderRadius: "6px",
    color: "#000",
    fontWeight: 800,
    fontSize: "9.5px",
    padding: "7px 0",
    cursor: "pointer",
  },
  rejectBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "4px",
    background: "rgba(255, 45, 85, 0.15)",
    border: "1px solid #ff2d55",
    borderRadius: "6px",
    color: "#ff2d55",
    fontWeight: 800,
    fontSize: "9.5px",
    padding: "6px 0",
    cursor: "pointer",
  },
  decisionBanner: (decision) => ({
    padding: "6px",
    borderRadius: "6px",
    fontSize: "10px",
    fontWeight: 700,
    textAlign: "center",
    background: decision === "APPROVED" ? "rgba(48,209,88,0.15)" : "rgba(255,45,85,0.15)",
    color: decision === "APPROVED" ? "#30d158" : "#ff2d55",
    border: `1px solid ${decision === "APPROVED" ? "#30d158" : "#ff2d55"}`,
  }),
  standbyBox: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px",
    background: "rgba(48, 209, 88, 0.08)",
    border: "1px solid rgba(48, 209, 88, 0.2)",
    borderRadius: "6px",
  },
  historyLabel: {
    fontSize: "8.5px",
    fontWeight: 700,
    color: "rgba(255,255,255,0.4)",
    textTransform: "uppercase",
    marginBottom: "4px",
  },
  historyList: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  historyItem: {
    padding: "6px 8px",
    background: "rgba(255,255,255,0.03)",
    borderRadius: "6px",
    border: "1px solid rgba(255,255,255,0.06)",
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  historyTopRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "8px",
    width: "100%",
  },
  historyId: {
    fontSize: "9.5px",
    fontWeight: 700,
    color: "#e0e0ef",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    minWidth: 0,
    flex: 1,
  },
  historyBottomRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: "8.5px",
    color: "rgba(255,255,255,0.45)",
  },
  historyTtfChip: {
    fontSize: "8.5px",
    fontWeight: 700,
    color: "#ff9f0a",
    background: "rgba(255,159,10,0.12)",
    padding: "1px 4px",
    borderRadius: "3px",
  },
};

