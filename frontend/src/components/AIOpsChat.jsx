/**
 * AIOpsChat.jsx
 * ==============
 * Right-side drawer chatbot widget powered by QueryLangGraph02.
 *
 * Execution flow (mirrors the uploaded flow diagram exactly):
 *
 *   User Input
 *     → Guardrail 1: Input Check         → status: "security_blocked" (red card)
 *     → Parse Query Node (OpenRouter LLM)
 *     → Guardrail 2: Query Validation    → status: "security_blocked" (red card)
 *     → Validation Node                  → status: "validation_failed" (orange card + fix chips)
 *     → Intent Router
 *     → Persistence Layer (live_feed_db.sqlite)
 *     → Guardrail 3: Retrieval Check     → status: "security_blocked" (red card)
 *     → Retrieval & Visualization Node
 *     → Sufficiency Router               → status: "no_data_found" (blue card)
 *     → Guardrail 4: Synthesis Check     → status: "security_blocked" (red card)
 *     → Synthesis Node (OpenRouter LLM)
 *     → Response Formatter
 *     → Response Node                    → status: "success" (answer + optional chart)
 *
 * All 4 response states are rendered distinctly. No fallback/mock data.
 *
 * Props:
 *   chatbotApiUrl  — Base URL for QueryLangGraph02 API (default: http://localhost:8001)
 *   defaultOpen    — Whether drawer starts open (default: false)
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import AIOpsChartRenderer from './AIOpsChartRenderer';

// ── Constants ─────────────────────────────────────────────────────────
const DEFAULT_API_URL = 'http://localhost:8001';

const QUICK_PROMPTS = [
  { icon: '📈', label: 'CPU & Memory Metrics',    query: 'Show me the latest CPU and memory utilization metrics with a chart' },
  { icon: '⚠️', label: 'Latest Incidents',        query: 'What are the latest detected failure modes and incident classifications?' },
  { icon: '🔮', label: 'Time-to-Failure Forecast', query: 'What is the current time-to-failure forecast and which feature is most critical?' },
  { icon: '🛡️', label: 'Severity Status',          query: 'Show me the current severity level, escalation status, and severity updates' },
  { icon: '🔍', label: 'Feature Importance',       query: 'What are the top contributing features for the current failure mode?' },
  { icon: '📊', label: 'System Health Overview',  query: 'Give me a full system health overview with pipeline results and a visualization' },
];

// ── Utility: generate stable session UUID ──────────────────────────────
function getOrCreateSessionId() {
  const key = 'aiops_chat_session_id';
  let sid = sessionStorage.getItem(key);
  if (!sid) {
    sid = 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
    sessionStorage.setItem(key, sid);
  }
  return sid;
}

// ── Utility: simple markdown renderer (bold, code, lists) ─────────────
function renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="aiops-inline-code">$1</code>')
    .replace(/^#{1,3}\s(.+)$/gm, '<span class="aiops-md-heading">$1</span>')
    .replace(/^[-•]\s(.+)$/gm, '<span class="aiops-md-li">· $1</span>')
    .replace(/\n/g, '<br/>');
}

// ── Message component for security_blocked responses ──────────────────
function SecurityBlockedCard({ response }) {
  const stage = response.metadata?.stage || 'guardrail';
  const errCode = response.error_code || 'SEC_UNKNOWN';
  return (
    <div className="aiops-response-card aiops-card-security">
      <div className="aiops-card-header">
        <span className="aiops-card-icon">🔴</span>
        <span className="aiops-card-title">Security Policy Blocked</span>
        <span className="aiops-card-badge aiops-badge-red">{errCode}</span>
      </div>
      <p className="aiops-card-body">{response.answer || 'This query was blocked by security guardrails.'}</p>
      <div className="aiops-card-meta">
        <span>Stage: <strong>{stage}</strong></span>
      </div>
    </div>
  );
}

// ── Message component for validation_failed responses ─────────────────
function ValidationFailedCard({ response, onRetry }) {
  const issues   = response.details || [];
  const fixes    = response.suggested_corrections || [];
  return (
    <div className="aiops-response-card aiops-card-validation">
      <div className="aiops-card-header">
        <span className="aiops-card-icon">⚠️</span>
        <span className="aiops-card-title">Query Validation Failed</span>
        <span className="aiops-card-badge aiops-badge-amber">{response.error_code || 'ERR_VALIDATION'}</span>
      </div>
      <p className="aiops-card-body">{response.answer || 'Your query could not be validated.'}</p>
      {issues.length > 0 && (
        <ul className="aiops-issue-list">
          {issues.map((issue, i) => (
            <li key={i} className="aiops-issue-item">
              <span className="aiops-issue-bullet">›</span> {issue}
            </li>
          ))}
        </ul>
      )}
      {fixes.length > 0 && (
        <div className="aiops-corrections">
          <span className="aiops-corrections-label">💡 Suggested corrections:</span>
          <div className="aiops-correction-chips">
            {fixes.map((fix, i) => (
              <button key={i} className="aiops-correction-chip" onClick={() => onRetry(fix)}>
                {fix}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Message component for no_data_found responses ─────────────────────
function NoDataCard({ response }) {
  return (
    <div className="aiops-response-card aiops-card-nodata">
      <div className="aiops-card-header">
        <span className="aiops-card-icon">🔵</span>
        <span className="aiops-card-title">No Data Available</span>
        <span className="aiops-card-badge aiops-badge-blue">WARN_NO_DATA</span>
      </div>
      <p className="aiops-card-body">
        {response.answer || 'No records were found matching your query in the live pipeline database.'}
      </p>
      <div className="aiops-card-meta">
        <span>The inference pipeline may still be warming up, or no data exists for this query category.</span>
      </div>
    </div>
  );
}

// ── Message component for success responses ───────────────────────────
function SuccessCard({ response, onMetaToggle, metaOpen }) {
  const answer   = response.answer || 'No answer generated.';
  const vis      = response.visualization;
  const meta     = response.metadata || {};
  const hasChart = vis && vis.series && vis.series.some(s => s.data && s.data.length > 0);

  return (
    <div className="aiops-response-card aiops-card-success">
      {/* Markdown answer text */}
      <div
        className="aiops-answer-text"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(answer) }}
      />

      {/* Inline chart from VisualizationService */}
      {hasChart && (
        <div style={{ marginTop: 12 }}>
          <AIOpsChartRenderer visualization={vis} />
        </div>
      )}

      {/* Metadata drawer toggle */}
      <button className="aiops-meta-toggle" onClick={onMetaToggle}>
        {metaOpen ? '▲' : '▶'} Query Details
      </button>
      {metaOpen && (
        <div className="aiops-meta-drawer">
          <div className="aiops-meta-grid">
            {meta.execution_time_ms !== undefined && (
              <div className="aiops-meta-item">
                <span className="aiops-meta-label">Exec Time</span>
                <span className="aiops-meta-val">{meta.execution_time_ms} ms</span>
              </div>
            )}
            {meta.total_records_fetched !== undefined && (
              <div className="aiops-meta-item">
                <span className="aiops-meta-label">Records</span>
                <span className="aiops-meta-val">{meta.total_records_fetched}</span>
              </div>
            )}
            {meta.tables_queried && (
              <div className="aiops-meta-item">
                <span className="aiops-meta-label">Tables</span>
                <span className="aiops-meta-val">{
                  Array.isArray(meta.tables_queried)
                    ? meta.tables_queried.join(', ')
                    : meta.tables_queried
                }</span>
              </div>
            )}
            {meta.query_intent?.categories && (
              <div className="aiops-meta-item">
                <span className="aiops-meta-label">Categories</span>
                <span className="aiops-meta-val">{meta.query_intent.categories.join(', ')}</span>
              </div>
            )}
            {meta.routing_info?.primary_category && (
              <div className="aiops-meta-item">
                <span className="aiops-meta-label">Primary Route</span>
                <span className="aiops-meta-val">{meta.routing_info.primary_category}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Single message row (user or assistant) ────────────────────────────
function MessageRow({ msg, onRetry }) {
  const [metaOpen, setMetaOpen] = useState(false);

  if (msg.role === 'user') {
    return (
      <div className="aiops-msg-row aiops-msg-row-user">
        <div className="aiops-bubble-user">
          {msg.text}
        </div>
        <span className="aiops-msg-time">{msg.time}</span>
      </div>
    );
  }

  // Assistant message — render based on response status from flow diagram
  const status = msg.response?.status;

  return (
    <div className="aiops-msg-row aiops-msg-row-assistant">
      <div className="aiops-avatar">AI</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        {status === 'security_blocked' && <SecurityBlockedCard response={msg.response} />}
        {status === 'validation_failed' && <ValidationFailedCard response={msg.response} onRetry={onRetry} />}
        {status === 'no_data_found'    && <NoDataCard response={msg.response} />}
        {status === 'success'          && (
          <SuccessCard
            response={msg.response}
            metaOpen={metaOpen}
            onMetaToggle={() => setMetaOpen(v => !v)}
          />
        )}
        {/* Unknown/error status fallback (network or server error) */}
        {!['security_blocked','validation_failed','no_data_found','success'].includes(status) && (
          <div className="aiops-response-card aiops-card-validation">
            <div className="aiops-card-header">
              <span className="aiops-card-icon">⚠️</span>
              <span className="aiops-card-title">Unexpected Response</span>
            </div>
            <p className="aiops-card-body">{msg.errorText || JSON.stringify(msg.response)}</p>
          </div>
        )}
        <span className="aiops-msg-time">{msg.time}</span>
      </div>
    </div>
  );
}

// ── Typing indicator ───────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="aiops-msg-row aiops-msg-row-assistant">
      <div className="aiops-avatar">AI</div>
      <div className="aiops-typing">
        <span className="aiops-typing-label">Querying LangGraph pipeline</span>
        <div className="aiops-dots">
          <span className="aiops-dot" />
          <span className="aiops-dot" />
          <span className="aiops-dot" />
        </div>
      </div>
    </div>
  );
}

// ── Main AIOpsChat Component ──────────────────────────────────────────
export default function AIOpsChat({
  chatbotApiUrl = DEFAULT_API_URL,
  defaultOpen   = false,
}) {
  const [open,        setOpen]        = useState(defaultOpen);
  const [messages,    setMessages]    = useState([]);
  const [inputText,   setInputText]   = useState('');
  const [isLoading,   setIsLoading]   = useState(false);
  const [connected,   setConnected]   = useState(null); // null = checking, true/false
  const [showChips,   setShowChips]   = useState(true);

  const sessionId    = useMemo(() => getOrCreateSessionId(), []);
  const messagesRef  = useRef(null);
  const inputRef     = useRef(null);
  const healthTimer  = useRef(null);

  // ── Health check ────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${chatbotApiUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(4000),
      });
      setConnected(res.ok);
    } catch {
      setConnected(false);
    }
  }, [chatbotApiUrl]);

  useEffect(() => {
    checkHealth();
    healthTimer.current = setInterval(checkHealth, 15000);
    return () => clearInterval(healthTimer.current);
  }, [checkHealth]);

  // ── Auto-scroll to bottom on new messages ───────────────────────────
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // ── Focus input when drawer opens ───────────────────────────────────
  useEffect(() => {
    if (open && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 350);
    }
  }, [open]);

  // ── Send query through QueryLangGraph pipeline ──────────────────────
  const sendQuery = useCallback(async (query) => {
    const text = (query || inputText).trim();
    if (!text || isLoading) return;

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Add user message
    setMessages(prev => [...prev, { role: 'user', text, time: now, id: Date.now() }]);
    setInputText('');
    setShowChips(false);
    setIsLoading(true);

    try {
      const res = await fetch(`${chatbotApiUrl}/query`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ query: text, session_id: sessionId }),
        signal:  AbortSignal.timeout(90000),   // 90s — LLM can be slow
      });

      const replyTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      if (!res.ok) {
        // HTTP error (500, etc.)
        const errData = await res.json().catch(() => ({}));
        setMessages(prev => [...prev, {
          role:      'assistant',
          time:      replyTime,
          id:        Date.now(),
          errorText: errData.detail || `Server error ${res.status}`,
          response:  { status: 'http_error' },
        }]);
        return;
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        role:     'assistant',
        time:     replyTime,
        id:       Date.now(),
        response: data,
      }]);
    } catch (err) {
      const replyTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setMessages(prev => [...prev, {
        role:      'assistant',
        time:      replyTime,
        id:        Date.now(),
        errorText: `Connection failed: ${err.message}. Is the chatbot API running at ${chatbotApiUrl}?`,
        response:  { status: 'network_error' },
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [chatbotApiUrl, inputText, isLoading, sessionId]);

  // ── Keyboard: Enter sends, Shift+Enter = newline ─────────────────────
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  // ── Clear conversation ───────────────────────────────────────────────
  const clearChat = () => {
    setMessages([]);
    setShowChips(true);
  };

  // ── Connection indicator ─────────────────────────────────────────────
  const connStatus = connected === null ? 'checking' : connected ? 'connected' : 'offline';

  return (
    <>
      {/* ── Floating Action Button ─────────────────────────────────── */}
      <button
        id="aiops-fab-btn"
        className={`aiops-fab ${open ? 'aiops-fab-open' : ''}`}
        onClick={() => setOpen(v => !v)}
        title={open ? 'Close AI Assistant' : 'Open AIOps AI Assistant'}
        aria-label="AIOps AI Chat Assistant"
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"
              transform="scale(0) translate(0,0)" />
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 9h-2V5h2v6zm0 4h-2v-2h2v2z" />
          </svg>
        )}
      </button>

      {/* ── Backdrop overlay (visible when drawer is open) ────────── */}
      {open && (
        <div
          className="aiops-backdrop"
          onClick={() => setOpen(false)}
          aria-label="Close chat drawer"
        />
      )}

      {/* ── Right-side Drawer ─────────────────────────────────────── */}
      <aside
        id="aiops-chat-drawer"
        className={`aiops-drawer ${open ? 'aiops-drawer-open' : ''}`}
        role="dialog"
        aria-label="AIOps AI Chat Assistant"
        aria-modal="false"
      >
        {/* Drawer header */}
        <div className="aiops-drawer-header">
          <div className="aiops-drawer-brand">
            <div className="aiops-drawer-brand-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 9h-2V5h2v6zm0 4h-2v-2h2v2z" />
              </svg>
            </div>
            <div className="aiops-drawer-brand-text">
              <span className="aiops-drawer-title">Sentinel AI</span>
              <span className="aiops-drawer-sub">QueryLangGraph02 · Live Feed</span>
            </div>
          </div>

          <div className="aiops-drawer-actions">
            {/* Connection status */}
            <div className={`aiops-conn-pill aiops-conn-${connStatus}`} title={`API: ${chatbotApiUrl}`}>
              <span className="aiops-conn-dot" />
              <span className="aiops-conn-label">
                {connStatus === 'checking' ? 'Checking…' : connStatus === 'connected' ? 'Live' : 'Offline'}
              </span>
            </div>

            {/* Clear button */}
            {messages.length > 0 && (
              <button className="aiops-icon-btn" onClick={clearChat} title="Clear conversation">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" />
                  <path d="M10 11v6M14 11v6M9 6V4h6v2" />
                </svg>
              </button>
            )}

            {/* Close button */}
            <button className="aiops-icon-btn" onClick={() => setOpen(false)} title="Close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Offline warning banner */}
        {connStatus === 'offline' && (
          <div className="aiops-offline-banner">
            ⚠️ LangGraph API unavailable at {chatbotApiUrl} — start the chatbot server first.
          </div>
        )}

        {/* Messages area */}
        <div className="aiops-messages" ref={messagesRef}>
          {/* Welcome message (shown when no messages) */}
          {messages.length === 0 && (
            <div className="aiops-welcome">
              <div className="aiops-welcome-icon">🤖</div>
              <h3 className="aiops-welcome-title">AIOps AI Assistant</h3>
              <p className="aiops-welcome-sub">
                Ask natural-language questions about your live infrastructure metrics,
                failure modes, forecasts, and severity data.
              </p>
              <p className="aiops-welcome-powered">
                Powered by <strong>OpenRouter</strong> · <strong>Gemini 2.5 Flash</strong> · <strong>LangGraph</strong>
              </p>
            </div>
          )}

          {/* Message history */}
          {messages.map(msg => (
            <MessageRow
              key={msg.id}
              msg={msg}
              onRetry={(q) => sendQuery(q)}
            />
          ))}

          {/* Typing indicator */}
          {isLoading && <TypingIndicator />}
        </div>

        {/* Quick prompt chips (visible on fresh session) */}
        {showChips && messages.length === 0 && !isLoading && (
          <div className="aiops-chips-area">
            {QUICK_PROMPTS.map((p, i) => (
              <button
                key={i}
                className="aiops-quick-chip"
                onClick={() => sendQuery(p.query)}
                disabled={isLoading}
              >
                <span className="aiops-chip-icon">{p.icon}</span>
                <span className="aiops-chip-label">{p.label}</span>
              </button>
            ))}
          </div>
        )}

        {/* Input bar */}
        <div className="aiops-input-bar">
          <textarea
            ref={inputRef}
            className="aiops-input"
            placeholder="Ask about metrics, incidents, forecasts…"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
            aria-label="Query input"
          />
          <button
            className="aiops-send-btn"
            onClick={() => sendQuery()}
            disabled={isLoading || !inputText.trim()}
            title="Send query"
            aria-label="Send query"
          >
            {isLoading ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="aiops-spin">
                <path d="M12 4V2A10 10 0 0 0 2 12h2a8 8 0 0 1 8-8z"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            )}
          </button>
        </div>

        {/* Footer */}
        <div className="aiops-drawer-footer">
          <span>Session: <code className="aiops-session-id">{sessionId.slice(0, 16)}…</code></span>
          <span>Shift+Enter for newline</span>
        </div>
      </aside>
    </>
  );
}
