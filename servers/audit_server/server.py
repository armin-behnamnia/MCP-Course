# audit_server/server.py
"""
LRAA Audit MCP Server — FastMCP v3

Three record types (written by the agent, not auto-captured):
  1. tool_trace       — what tool ran, why, what context existed
  2. context_snapshot — a point-in-time capture of the agent's working memory
  3. decision         — a named choice with rationale and rejected alternatives

Three query resources (read by the agent or by humans):
  audit://session/{session_id}  — all records for a session
  audit://tool/{tool_name}      — all traces for a specific tool
  audit://recent                — last N records across all sessions

One query tool (for the agent to search its own history):
  query_audit — filter by type, session, tool, or failure status
"""

import json
from typing import Optional, Literal
from fastmcp import FastMCP, Context
from .store import AuditStore

audit = FastMCP(
    name="LRAAuditServer",
    instructions=(
        "Audit server for the Local Research and Action Audit system. "
        "Use record_tool_trace to log tool calls with context. "
        "Use record_context_snapshot to save working memory state. "
        "Use record_decision to log choices with rationale. "
        "Use query_audit to search past records."
    ),
)

_store = AuditStore()


# ═════════════════════════════════════════════════════════════════════════════
# WRITE TOOLS — called by the agent during its work
# ═════════════════════════════════════════════════════════════════════════════

@audit.tool(
    name="record_tool_trace",
    description=(
        "Record that a tool was called, why it was called, and what context "
        "existed at the time. Call this AFTER each significant tool use so the "
        "audit trail captures intent, not just mechanics."
    ),
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
async def record_tool_trace(
    session_id:   str,
    tool_called:  str,
    rationale:    str,
    inputs:       dict,
    result_summary: str,
    success:      bool,
    parent_trace_id: Optional[str] = None,
    failure_reason:  Optional[str] = None,
    ctx: Context = None,
) -> dict:
    """
    Parameters
    ----------
    session_id      : Unique ID for the current agent session.
    tool_called     : Name of the tool that was invoked.
    rationale       : WHY this tool was called — the agent's intent.
    inputs          : The actual arguments passed to the tool (redact secrets).
    result_summary  : A short description of what the result contained.
    success         : Whether the tool completed without error.
    parent_trace_id : ID of a prior trace that caused this call (for causal chains).
    failure_reason  : If success=False, the error message.
    """
    record = {
        "type":            "tool_trace",
        "session_id":      session_id,
        "tool_called":     tool_called,
        "rationale":       rationale,
        "inputs":          inputs,
        "result_summary":  result_summary,
        "success":         success,
        "parent_trace_id": parent_trace_id,
        "failure_reason":  failure_reason,
    }
    record_id = _store.append(record)

    if ctx:
        await ctx.info(f"[audit] trace recorded: {tool_called} → {record_id}")

    return {"record_id": record_id, "status": "recorded"}


@audit.tool(
    name="record_context_snapshot",
    description=(
        "Save a snapshot of the agent's current working memory: what query is "
        "being answered, which documents were retrieved, what has been read so far. "
        "Call this at the start of a research task and after major retrieval steps."
    ),
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
async def record_context_snapshot(
    session_id:          str,
    current_query:       str,
    documents_retrieved: list[str],
    sections_read:       list[str],
    working_hypothesis:  Optional[str] = None,
    notes:               Optional[str] = None,
    ctx: Context = None,
) -> dict:
    """
    Parameters
    ----------
    session_id           : Unique ID for the current agent session.
    current_query        : The research question being answered right now.
    documents_retrieved  : List of document IDs / filenames retrieved so far.
    sections_read        : List of 'doc_id::section_name' strings already read.
    working_hypothesis   : The agent's current best answer before it is finalised.
    notes                : Free-text observations, contradictions noticed, gaps.
    """
    record = {
        "type":                 "context_snapshot",
        "session_id":           session_id,
        "current_query":        current_query,
        "documents_retrieved":  documents_retrieved,
        "sections_read":        sections_read,
        "working_hypothesis":   working_hypothesis,
        "notes":                notes,
    }
    record_id = _store.append(record)

    if ctx:
        await ctx.info(f"[audit] snapshot recorded for query: '{current_query[:60]}…'")

    return {"record_id": record_id, "status": "recorded"}


@audit.tool(
    name="record_decision",
    description=(
        "Record a named decision: what was chosen, why, and what alternatives "
        "were rejected. Use this for any choice that affects the final answer — "
        "which paper to trust, which method to use, which section to cite."
    ),
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
async def record_decision(
    session_id:           str,
    decision_name:        str,
    chosen:               str,
    rationale:            str,
    alternatives_rejected: list[str],
    confidence:           Literal["high", "medium", "low"] = "medium",
    evidence:             Optional[list[str]] = None,
    ctx: Context = None,
) -> dict:
    """
    Parameters
    ----------
    session_id            : Unique ID for the current agent session.
    decision_name         : Short label, e.g. 'which_paper_to_cite'.
    chosen                : What was selected.
    rationale             : Why this option was chosen over the others.
    alternatives_rejected : List of options that were considered but not used.
    confidence            : Agent's confidence in this decision.
    evidence              : Document IDs or section refs supporting the decision.
    """
    record = {
        "type":                 "decision",
        "session_id":           session_id,
        "decision_name":        decision_name,
        "chosen":               chosen,
        "rationale":            rationale,
        "alternatives_rejected": alternatives_rejected,
        "confidence":           confidence,
        "evidence":             evidence or [],
    }
    record_id = _store.append(record)

    if ctx:
        await ctx.info(
            f"[audit] decision '{decision_name}' recorded "
            f"(chose: {chosen[:60]}, confidence: {confidence})"
        )

    return {"record_id": record_id, "status": "recorded"}


# ═════════════════════════════════════════════════════════════════════════════
# QUERY TOOL — the agent can search its own history
# ═════════════════════════════════════════════════════════════════════════════

@audit.tool(
    name="query_audit",
    description=(
        "Search past audit records. Use this to recall what was already tried, "
        "avoid repeating failed tool calls, or review decisions made earlier."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def query_audit(
    session_id:  Optional[str] = None,
    tool_name:   Optional[str] = None,
    record_type: Optional[Literal["tool_trace", "context_snapshot", "decision"]] = None,
    failures_only: bool = False,
    limit:       int  = 20,
) -> list[dict]:
    """
    All filters are optional and AND-combined.
    Returns up to `limit` records, most recent first.
    """
    records = _store.all()

    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]
    if tool_name:
        records = [r for r in records if r.get("tool_called") == tool_name]
    if record_type:
        records = [r for r in records if r.get("type") == record_type]
    if failures_only:
        records = [r for r in records if not r.get("success", True)]

    # most recent first, then cap
    records = list(reversed(records))[:limit]
    return records


# ═════════════════════════════════════════════════════════════════════════════
# RESOURCES — read-only views the agent or a human can pull
# ═════════════════════════════════════════════════════════════════════════════

@audit.resource(
    "audit://recent",
    name="RecentAuditRecords",
    description="The 20 most recent audit records across all sessions.",
    mime_type="application/json",
    tags={"audit", "monitoring"},
)
def resource_recent() -> str:
    return json.dumps(_store.recent(20), indent=2)


@audit.resource(
    "audit://session/{session_id}",
    name="SessionAuditRecords",
    description="All audit records for a specific agent session.",
    mime_type="application/json",
    tags={"audit", "session"},
)
def resource_by_session(session_id: str) -> str:
    return json.dumps(_store.by_session(session_id), indent=2)


@audit.resource(
    "audit://tool/{tool_name}",
    name="ToolAuditRecords",
    description="All tool_trace records for a specific tool name.",
    mime_type="application/json",
    tags={"audit", "tool"},
)
def resource_by_tool(tool_name: str) -> str:
    return json.dumps(_store.by_tool(tool_name), indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    audit.run(transport="http", port=8788)   # main LRAA server is on 8787