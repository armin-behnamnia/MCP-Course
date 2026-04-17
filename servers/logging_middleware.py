# middleware.py
"""
Observability middleware for the LRAA MCP server — FastMCP v3 correct.

Uses:
  - get_context()         from fastmcp.server.dependencies  (v3 way to get Context)
  - ctx.set_state() / ctx.get_state()  for per-session counters  (new in v3)
  - ctx.info() / ctx.error()           to send logs to the MCP client  (v3)
  - Python logging                     for server-side structured audit log
"""

import json
import logging
import time
import threading
from collections import defaultdict

import requests
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_context   # ← v3 correct import

# Server-side audit logger (writes to file / stdout on the server)
audit_log = logging.getLogger("lraa.audit")

# ── Cost table (USD / 1K tokens) ──────────────────────────────────────────────
COST_PER_1K = {
    "qwen3:0.6b": 0.0,                    # local Ollama – free
    "claude-sonnet-4-20250514": 0.003,
}
AVG_CHARS_PER_TOKEN = 4
LLM_TOOLS = {"summarize_filtered_sections"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Global frequency / failure counters  (across all sessions)
# ─────────────────────────────────────────────────────────────────────────────
class ToolStats:
    def __init__(self):
        self._lock  = threading.Lock()
        self.calls  = defaultdict(int)
        self.errors = defaultdict(int)

    def record(self, tool: str, *, failed: bool):
        with self._lock:
            self.calls[tool]  += 1
            if failed:
                self.errors[tool] += 1

    def failure_rate(self, tool: str) -> float:
        with self._lock:
            c = self.calls[tool]
            return round(self.errors[tool] / c, 4) if c else 0.0

_stats = ToolStats()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Thread-local bucket for external HTTP call tracking
#    (monkey-patches requests.Session.send — the single chokepoint for all
#     requests.get / requests.post calls including your Crossref call)
# ─────────────────────────────────────────────────────────────────────────────
_local = threading.local()

_original_send = requests.Session.send

def _patched_send(self, request, **kwargs):
    t0 = time.perf_counter()
    try:
        resp = _original_send(self, request, **kwargs)
        _local.ext_calls = getattr(_local, "ext_calls", [])
        _local.ext_calls.append({
            "url":        request.url,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "status":     resp.status_code,
        })
        return resp
    except Exception as exc:
        _local.ext_calls = getattr(_local, "ext_calls", [])
        _local.ext_calls.append({
            "url":        request.url,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "status":     f"ERROR:{type(exc).__name__}",
        })
        raise

requests.Session.send = _patched_send   # applied once at import


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cost helper
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_cost(args: dict, result_text: str, model: str) -> float:
    rate = COST_PER_1K.get(model, 0.0)
    if not rate:
        return 0.0
    tokens = (sum(len(str(v)) for v in args.values()) + len(result_text)) / AVG_CHARS_PER_TOKEN
    return round(tokens / 1000 * rate, 8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. The middleware
# ─────────────────────────────────────────────────────────────────────────────
class ObservabilityMiddleware(Middleware):
    """
    Intercepts every tool call and emits a structured JSON audit line.
    Also sends a summary log message back to the MCP client via ctx.info/error.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name
        raw_args  = dict(context.message.arguments or {})

        # Redact credentials before they touch any log
        args = {k: ("***" if k == "token" else v) for k, v in raw_args.items()}

        # Reset the external-call bucket for this request
        _local.ext_calls = []

        t0             = time.perf_counter()
        result         = None
        success        = True
        failure_reason = None

        try:
            result = await call_next(context)
            return result

        except Exception as exc:
            success        = False
            failure_reason = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            latency_ms    = round((time.perf_counter() - t0) * 1000, 1)
            result_text   = str(result) if result is not None else ""
            response_size = len(result_text.encode("utf-8"))
            ext_calls     = list(getattr(_local, "ext_calls", []))

            model    = "qwen3:0.6b" if tool_name in LLM_TOOLS else ""
            cost_usd = _estimate_cost(args, result_text, model)

            _stats.record(tool_name, failed=not success)

            entry = {
                "tool":            tool_name,
                "args":            args,
                "latency_ms":      latency_ms,
                "response_size_b": response_size,
                "success":         success,
                "failure_reason":  failure_reason,
                "ext_calls":       ext_calls,
                "cost_usd":        cost_usd,
                "call_count":      _stats.calls[tool_name],
                "failure_rate":    _stats.failure_rate(tool_name),
            }

            # ── server-side structured log (file / stdout) ────────────────────
            if success:
                audit_log.info(json.dumps(entry))
            else:
                audit_log.error(json.dumps(entry))

            # ── v3: send a summary log back to the MCP client ─────────────────
            # get_context() retrieves the active Context for this request.
            # It only works inside an active request — exactly where we are.
            try:
                ctx = get_context()
                if success:
                    await ctx.info(
                        f"[audit] {tool_name} OK | {latency_ms}ms | {response_size}b"
                    )
                else:
                    await ctx.error(
                        f"[audit] {tool_name} FAILED | {latency_ms}ms | {failure_reason}"
                    )
            except RuntimeError:
                # get_context() raises RuntimeError if no active context.
                # This can happen during startup probes — safe to skip.
                pass