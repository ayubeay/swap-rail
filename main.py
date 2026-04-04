"""
execution-coordinator v0.7.0
OROS — Agent OS control-plane service.

Four kernel primitives:
  observe  → EVM Enrich → LITMUS → ORA → Intent Verify
  adjudicate → IAM → Governor → VERITY → Shield Router
  settle → VYRE artifact creation
  learn → outcome evaluation → reputation update → policy feedback

The full loop: observe → adjudicate → settle → learn

Design principles:
  - JSON internally, VYRE-signed at the boundary
  - Coordination only — no module business logic embedded
  - ORA + Intent Verifier are kernel logic, not microservices
  - Schema-versioned from day one
  - Artifact-eligible by default, artifact-enforced at output
"""

import os
import uuid
import hashlib
import json
import time
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evm_adapter import enrich_with_evm_context, evm_adapter
from intent_verifier import intent_verifier, IntentVerification
from learn import (learn_engine, reputation_ledger, OutcomeReport,
                   OutcomeFeedback, ExecutionOutcome)
import store as state
from hexagram_engine import hexagram_engine
from mars_engineer import mars_engineer, ThermalAssessment
from telemetry import telemetry_engine, TelemetryEvent
from physics_kernel import assess_governance_physics
from anima_potentia import anima_potentia
from galileo_adapter import galileo_engine

logger = logging.getLogger("oros")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.4.1"
ARTIFACT_VERSION = "0.1.0"
SERVICE_NAME = "execution-coordinator"

# Downstream module URLs (Railway service URLs, set via env)
IAM_URL = os.getenv("IAM_URL", "")
VERITY_URL = os.getenv("VERITY_URL", "")
SURVIVOR_URL = os.getenv("SURVIVOR_URL", "")
SHIELD_URL = os.getenv("SHIELD_URL", "")
LITMUS_URL = os.getenv("LITMUS_URL", "")

# API keys for downstream services
SURVIVOR_API_KEY = os.getenv("SURVIVOR_API_KEY", "")

# VYRE signing key (Ed25519 seed, hex-encoded, set via env)
VYRE_SIGNING_KEY = os.getenv("VYRE_SIGNING_KEY", "")

# ---------------------------------------------------------------------------
# Event Lifecycle
# ---------------------------------------------------------------------------

class EventState(str, Enum):
    RECEIVED = "received"
    AUTHORIZED = "authorized"
    DENIED = "denied"
    OBSERVED = "observed"
    INTERPRETED = "interpreted"
    VERIFIED = "verified"
    SCORED = "scored"
    ARTIFACTED = "artifacted"
    ROUTED = "routed"
    FAILED = "failed"


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_GUARDRAILS = "ALLOW_WITH_GUARDRAILS"
    SIMULATE_ONLY = "SIMULATE_ONLY"
    THROTTLE = "THROTTLE"
    DEFER = "DEFER"
    READ_ONLY = "READ_ONLY"
    DENY = "DENY"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ActionPayload(BaseModel):
    """Flexible action data — varies by action_type."""
    model_config = {"extra": "allow"}


class EnvironmentContext(BaseModel):
    chain: Optional[str] = None
    network: Optional[str] = None
    runtime: Optional[str] = None
    model_config = {"extra": "allow"}


class PolicyContext(BaseModel):
    policy_id: Optional[str] = None
    decision: Optional[PolicyDecision] = None
    reason: Optional[str] = None
    reputation_band: Optional[str] = None  # restricted | watch | standard | trusted
    spend_limit: Optional[float] = None  # max spend allowed for this event


class ExecutionEventInput(BaseModel):
    """Inbound event — what callers send."""
    agent_id: str
    action_type: str
    action_payload: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """One hop in the execution pipeline."""
    module: str
    state: EventState
    timestamp: str
    latency_ms: Optional[float] = None
    output: dict[str, Any] = Field(default_factory=dict)


class ORAContext(BaseModel):
    """ORA interpretation output — feeds VERITY scoring."""
    anomaly_score: float = 0.0
    behavior_class: str = "normal"  # normal | edge_case | suspicious | policy_risk
    risk_signal: str = "low"  # low | medium | high | critical
    policy_flag: bool = False
    signals: list[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    archetype: Optional[str] = None  # governance archetype from hexagram framework
    governance_posture: Optional[str] = None  # operational posture derived from archetype


class GovernanceSummary(BaseModel):
    """Top-level governance judgment — the readable verdict."""
    decision: Optional[str] = None          # ALLOW | THROTTLE | DENY
    reputation_score: Optional[float] = None
    reputation_band: Optional[str] = None   # standard_new | standard_plus | watch_borderline | etc.
    spend_limit: Optional[float] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None        # LOW | MEDIUM | HIGH | EXTREME
    archetype: Optional[str] = None         # modesty | breakthrough | standstill | etc.
    governance_posture: Optional[str] = None
    hexagram_state: Optional[str] = None    # alignment | risk | conflict | dissolution | etc.
    hexagram_index: Optional[int] = None    # 0-63 binary state code
    hexagram_strategy: Optional[str] = None # recommended action from state
    intent_match: Optional[str] = None      # match | mismatch | unverifiable | suspicious
    intent_adjustment: Optional[str] = None # what policy did about the intent state
    artifact_hash: Optional[str] = None
    artifact_verified: bool = False
    signer: Optional[str] = None


class ExecutionEvent(BaseModel):
    """Full internal event with trace state."""
    event_id: str
    schema_version: str = SCHEMA_VERSION
    agent_id: str
    session_id: str
    action_type: str
    action_payload: dict[str, Any]
    environment: dict[str, Any]
    governance: GovernanceSummary = Field(default_factory=GovernanceSummary)
    policy_context: PolicyContext = Field(default_factory=PolicyContext)
    ora_context: ORAContext = Field(default_factory=ORAContext)
    intent_verification: Optional[dict[str, Any]] = None
    trace_id: str
    parent_event_id: Optional[str] = None
    state: EventState = EventState.RECEIVED
    integrity_score: Optional[float] = None
    risk_score: Optional[float] = None
    artifact_hash: Optional[str] = None
    trace: list[TraceStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class VYREEnvelope(BaseModel):
    """Signed artifact wrapping an execution event."""
    artifact_type: str = "execution_event"
    artifact_version: str = ARTIFACT_VERSION
    event_id: str
    event_hash: str
    trace_id: str
    agent_id: str
    action_type: str
    policy_decision: Optional[PolicyDecision] = None
    ora_behavior_class: Optional[str] = None
    ora_anomaly_score: Optional[float] = None
    ora_risk_signal: Optional[str] = None
    intent_match: Optional[str] = None
    intent_risk_score: Optional[float] = None
    integrity_score: Optional[float] = None
    risk_score: Optional[float] = None
    signatures: list[dict[str, str]] = Field(default_factory=list)
    created_at: str = ""


# ---------------------------------------------------------------------------
# State persistence (Redis-backed, in-memory fallback)
# See store.py for implementation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VYRE signing utilities
# ---------------------------------------------------------------------------

def hash_event(event: ExecutionEvent) -> str:
    """Deterministic SHA-256 of the canonical event JSON."""
    canonical = json.dumps({
        "event_id": event.event_id,
        "agent_id": event.agent_id,
        "action_type": event.action_type,
        "action_payload": event.action_payload,
        "policy_context": event.policy_context.model_dump(),
        "ora_context": event.ora_context.model_dump(),
        "intent_verification": event.intent_verification,
        "integrity_score": event.integrity_score,
        "risk_score": event.risk_score,
        "state": event.state.value,
        "trace_id": event.trace_id,
    }, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def sign_artifact(event_hash: str) -> dict[str, str]:
    """
    Sign the event hash with Ed25519 via VYRE key.
    Returns {"signer": ..., "signature": ...}
    Falls back to unsigned stub if no key configured.
    """
    if not VYRE_SIGNING_KEY:
        return {"signer": SERVICE_NAME, "signature": "unsigned:no_key_configured"}
    try:
        from nacl.signing import SigningKey
        seed = bytes.fromhex(VYRE_SIGNING_KEY)
        sk = SigningKey(seed)
        signed = sk.sign(event_hash.encode())
        return {
            "signer": SERVICE_NAME,
            "signature": signed.signature.hex(),
            "public_key": sk.verify_key.encode().hex(),
        }
    except Exception as e:
        return {"signer": SERVICE_NAME, "signature": f"error:{e}"}


def create_artifact(event: ExecutionEvent) -> VYREEnvelope:
    """Package a completed event into a signed VYRE envelope."""
    event_hash = hash_event(event)
    sig = sign_artifact(event_hash)
    now = datetime.now(timezone.utc).isoformat()

    # Extract intent verification fields if present
    iv = event.intent_verification or {}

    return VYREEnvelope(
        event_id=event.event_id,
        event_hash=event_hash,
        trace_id=event.trace_id,
        agent_id=event.agent_id,
        action_type=event.action_type,
        policy_decision=event.policy_context.decision,
        ora_behavior_class=event.ora_context.behavior_class,
        ora_anomaly_score=event.ora_context.anomaly_score,
        ora_risk_signal=event.ora_context.risk_signal,
        intent_match=iv.get("intent_match"),
        intent_risk_score=iv.get("intent_risk_score"),
        integrity_score=event.integrity_score,
        risk_score=event.risk_score,
        signatures=[sig],
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Pipeline modules — wired to real service APIs
# ---------------------------------------------------------------------------

import httpx


async def call_module(url: str, path: str, payload: dict,
                       headers: dict = None, method: str = "POST") -> dict:
    """Generic downstream module call. Returns response JSON or error dict."""
    if not url:
        return {"status": "skipped", "reason": "no_url_configured"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if method == "GET":
                resp = await client.get(f"{url}{path}", headers=headers or {})
            else:
                resp = await client.post(f"{url}{path}", json=payload,
                                          headers=headers or {})
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        return {"status": "error", "reason": "timeout"}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "reason": f"http_{e.response.status_code}",
                "detail": e.response.text[:200]}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def add_trace(event: ExecutionEvent, module: str, state: EventState,
              output: dict, start_time: float):
    """Append a trace step to the event."""
    event.trace.append(TraceStep(
        module=module,
        state=state,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=round((time.time() - start_time) * 1000, 2),
        output=output,
    ))
    event.state = state
    event.updated_at = datetime.now(timezone.utc).isoformat()


async def pipeline_iam_governor(event: ExecutionEvent) -> bool:
    """Step 1: IAM authorization + Governor policy check.

    IAM at identityaware.pages.dev provides agent integrity scoring.
    We map to its available endpoints:
      - GET /agents/index — agent integrity data
      - POST /integrity/score — full integrity evaluation (if available)

    If IAM is configured, we check agent identity and integrity.
    If not configured, default allow for development.
    """
    t = time.time()

    if not IAM_URL:
        # No IAM configured — default allow for development
        event.policy_context.decision = PolicyDecision.ALLOW
        event.policy_context.reason = "iam_not_configured:default_allow"
        add_trace(event, "iam/governor", EventState.AUTHORIZED, {
            "status": "skipped", "reason": "no_url_configured"
        }, t)
        return True

    # Try POST /authorize first (canonical OROS-IAM contract)
    result = await call_module(IAM_URL, "/authorize", {
        "agent_id": event.agent_id,
        "action_type": event.action_type,
        "action_payload": event.action_payload,
        "environment": event.environment,
    })

    # If /authorize returns 404 or error, fall back to /integrity/check
    # This allows IAM to expose a lightweight free endpoint
    if result.get("status") == "error" and "404" in result.get("reason", ""):
        # Fallback: try free integrity check endpoint
        result = await call_module(IAM_URL, "/integrity/check", {
            "agent_id": event.agent_id,
            "action_type": event.action_type,
        })

    if result.get("status") in ("skipped", "error"):
        # IAM unavailable — degrade but flag it
        event.policy_context.decision = PolicyDecision.ALLOW
        event.policy_context.reason = f"iam_degraded:{result.get('reason', 'unavailable')}"
        event.metadata["iam_degraded"] = True
        add_trace(event, "iam/governor", EventState.AUTHORIZED, {
            "status": "degraded",
            "reason": result.get("reason", "unknown"),
            "fallback": "default_allow",
        }, t)
        return True

    decision = result.get("decision", "ALLOW")
    event.policy_context.policy_id = result.get("policy_id")
    event.policy_context.decision = PolicyDecision(decision)
    event.policy_context.reason = result.get("reason", "")

    # Capture IAM integrity score if returned
    iam_integrity = result.get("integrity_score") or result.get("coherence")
    if iam_integrity is not None:
        event.metadata["iam_integrity"] = iam_integrity

    # Capture IAM identity state
    iam_identity = result.get("identity_state") or result.get("identity")
    if iam_identity:
        event.metadata["iam_identity"] = iam_identity

    if decision == "DENY":
        add_trace(event, "iam/governor", EventState.DENIED, result, t)
        return False

    add_trace(event, "iam/governor", EventState.AUTHORIZED, result, t)
    return True


# ---------------------------------------------------------------------------
# Reputation-informed governance
# ---------------------------------------------------------------------------
# Past behavior changes future permissions.
# Reputation bands determine base policy constraints.
# Within-band modifiers reward earned trust and penalize erosion.
# ---------------------------------------------------------------------------

REPUTATION_BANDS = {
    "restricted": {"min": 0.0, "max": 39.99},
    "watch":      {"min": 40.0, "max": 49.99},
    "standard":   {"min": 50.0, "max": 69.99},
    "trusted":    {"min": 70.0, "max": 100.0},
}

BAND_POLICIES = {
    "restricted": {
        "spend_limit": 50.0,
        "block_new_counterparties": True,
        "require_human_review": True,
        "sensitive_action_mode": PolicyDecision.DENY,
        "reason_code": "REPUTATION_RESTRICTED",
    },
    "watch": {
        "spend_limit": 500.0,
        "block_new_counterparties": False,
        "require_human_review": False,
        "sensitive_action_mode": PolicyDecision.THROTTLE,
        "reason_code": "REPUTATION_WATCH",
    },
    "standard": {
        "spend_limit": 5000.0,
        "block_new_counterparties": False,
        "require_human_review": False,
        "sensitive_action_mode": None,
        "reason_code": "REPUTATION_STANDARD",
    },
    "trusted": {
        "spend_limit": 50000.0,
        "block_new_counterparties": False,
        "require_human_review": False,
        "sensitive_action_mode": None,
        "reason_code": "REPUTATION_TRUSTED",
    },
}


def get_reputation_band(score: float) -> str:
    """Map a reputation score to a governance band."""
    if score >= 70.0:
        return "trusted"
    elif score >= 50.0:
        return "standard"
    elif score >= 40.0:
        return "watch"
    return "restricted"


def apply_within_band_modifier(score: float, band: str,
                                 base_limit: float,
                                 total_events: int,
                                 positive_count: int,
                                 negative_count: int) -> tuple[float, str]:
    """
    Apply continuous trust scaling within a band.

    Returns (adjusted_spend_limit, modifier_label).

    Within-band modifiers reward earned trust:
      - standard_base: new agent, no history → base limit
      - standard_plus: positive history → elevated limit
      - standard_proven: strong positive track record → higher limit
      - watch_recovering: improving from negative → slightly better limit
      - trusted_elite: extensive positive history → maximum limit

    The modifier is a multiplier on the base spend limit,
    scaled by the agent's position within their band.
    """
    modifier_label = f"{band}_base"
    adjusted_limit = base_limit

    if band == "standard":
        # 50.0 = base, 50.0-54.99 = plus, 55.0-69.99 = proven
        if total_events == 0:
            modifier_label = "standard_new"
            adjusted_limit = base_limit  # no history, base limit
        elif score >= 55.0:
            modifier_label = "standard_proven"
            adjusted_limit = base_limit * 3.0  # 15,000
        elif score >= 50.3 and positive_count > 0 and negative_count == 0:
            modifier_label = "standard_plus"
            adjusted_limit = base_limit * 1.5  # 7,500
        elif score >= 50.3 and positive_count > negative_count:
            modifier_label = "standard_improving"
            adjusted_limit = base_limit * 1.25  # 6,250
        else:
            modifier_label = "standard_base"

    elif band == "watch":
        # Agent is in penalty territory — reward recovery signals
        if positive_count > 0 and positive_count >= negative_count:
            modifier_label = "watch_recovering"
            adjusted_limit = base_limit * 1.5  # 750 instead of 500
        elif score >= 48.0:
            modifier_label = "watch_borderline"
            adjusted_limit = base_limit * 1.2  # 600
        else:
            modifier_label = "watch_penalized"

    elif band == "trusted":
        if score >= 85.0 and total_events >= 50:
            modifier_label = "trusted_elite"
            adjusted_limit = base_limit * 2.0  # 100,000
        elif score >= 75.0 and positive_count > 10:
            modifier_label = "trusted_established"
            adjusted_limit = base_limit * 1.5  # 75,000
        else:
            modifier_label = "trusted_base"

    elif band == "restricted":
        modifier_label = "restricted_locked"

    return round(adjusted_limit, 2), modifier_label


async def pipeline_reputation_governor(event: ExecutionEvent):
    """
    Step 1.5: Reputation-informed governance.

    Reads agent's cumulative reputation from Redis and applies
    policy constraints based on their band + within-band modifier:
      - restricted (0-39): heavy limits, may deny sensitive actions
      - watch (40-49): reduced limits, throttle sensitive actions
      - standard (50-69): normal operations, scaled by trust
      - trusted (70+): elevated limits, faster processing

    Within-band modifiers:
      - standard_new: no history → base limit
      - standard_plus: positive track → 1.5x limit
      - standard_proven: strong track → 3x limit
      - watch_recovering: improving → slightly better
      - trusted_elite: extensive history → 2x limit

    This is where past behavior changes future permissions.
    """
    t = time.time()

    # Fetch reputation from Redis ledger
    rep_data = reputation_ledger.get_reputation(event.agent_id)
    score = rep_data.get("score", 50.0)
    total_events = rep_data.get("total_events", 0)
    positive_count = rep_data.get("positive_count", 0)
    negative_count = rep_data.get("negative_count", 0)

    band = get_reputation_band(score)
    band_policy = BAND_POLICIES[band]

    # Apply within-band modifier for continuous trust scaling
    adjusted_limit, modifier_label = apply_within_band_modifier(
        score, band, band_policy["spend_limit"],
        total_events, positive_count, negative_count
    )

    # Apply band + modifier to event
    event.policy_context.reputation_band = modifier_label
    event.policy_context.spend_limit = adjusted_limit

    # Check spend against adjusted limit
    amount = event.action_payload.get("amount")
    spend_blocked = False
    if amount is not None:
        try:
            amount = float(amount)
            if amount > adjusted_limit:
                event.policy_context.decision = PolicyDecision.DENY
                event.policy_context.reason = (
                    f"spend_exceeds_reputation_limit:"
                    f"amount={amount}:limit={adjusted_limit}:"
                    f"modifier={modifier_label}:score={score}"
                )
                spend_blocked = True
        except (ValueError, TypeError):
            pass

    # Apply sensitive action override
    sensitive_actions = {
        "payment_attempt", "transfer", "withdrawal",
        "contract_deploy", "delegation", "permission_change",
    }
    sensitive_blocked = False
    if (not spend_blocked
            and event.action_type in sensitive_actions
            and band_policy["sensitive_action_mode"] is not None):
        event.policy_context.decision = band_policy["sensitive_action_mode"]
        event.policy_context.reason = (
            f"reputation_policy:{modifier_label}:{band_policy['sensitive_action_mode'].value}:"
            f"score={score}"
        )
        sensitive_blocked = (
            band_policy["sensitive_action_mode"] == PolicyDecision.DENY
        )

    add_trace(event, "reputation_governor", EventState.AUTHORIZED, {
        "reputation_score": score,
        "band": band,
        "modifier": modifier_label,
        "base_limit": band_policy["spend_limit"],
        "adjusted_limit": adjusted_limit,
        "total_events": total_events,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "reason_code": band_policy["reason_code"],
        "spend_blocked": spend_blocked,
        "sensitive_blocked": sensitive_blocked,
    }, t)

    # Return False if action should be blocked
    return not (spend_blocked or sensitive_blocked)


async def pipeline_litmus_observe(event: ExecutionEvent):
    """Step 2: LITMUS observability capture."""
    t = time.time()
    result = await call_module(LITMUS_URL, "/observe", {
        "event_id": event.event_id,
        "trace_id": event.trace_id,
        "agent_id": event.agent_id,
        "action_type": event.action_type,
        "action_payload": event.action_payload,
        "state": event.state.value,
    })
    add_trace(event, "litmus", EventState.OBSERVED, result, t)


# ---------------------------------------------------------------------------
# ORA Engine — Observational Risk Agent (internal module)
# ---------------------------------------------------------------------------
# ORA is the interpretation layer between observation and scoring.
# It reads telemetry, detects anomalies, classifies behavior, and
# enriches context so VERITY can score integrity accurately.
#
# ORA is kernel logic — it runs inside the coordinator, not as a
# separate service. It becomes a microservice only when multiple
# services need independent interpretation or when ML inference
# requires dedicated compute.
#
# Governance Archetype Framework (I Ching-derived):
# ORA uses a library of governance archetypes to classify situational
# posture. These are not divination — they are a pattern language for
# operational governance, mapping hexagram traits to risk modes.
# ---------------------------------------------------------------------------

# Agent behavior history (Redis-backed via store module)
# AGENT_HISTORY moved to store.py

# Thresholds (configurable via env later)
HIGH_VALUE_THRESHOLD = float(os.getenv("ORA_HIGH_VALUE_THRESHOLD", "1000"))
VELOCITY_WINDOW_SECONDS = int(os.getenv("ORA_VELOCITY_WINDOW", "300"))
VELOCITY_MAX_EVENTS = int(os.getenv("ORA_VELOCITY_MAX", "10"))


# ---------------------------------------------------------------------------
# Governance Archetypes — derived from I Ching hexagram traits
# ---------------------------------------------------------------------------
# Each archetype encodes a situational governance posture:
#   - conditions: when this archetype applies (signal patterns)
#   - posture: operational stance the system should adopt
#   - recommendation_override: optional override for ORA's recommendation
#
# Not mysticism. Pattern language for situational intelligence.
# ---------------------------------------------------------------------------

GOVERNANCE_ARCHETYPES = {

    # Hexagram 43 — Breakthrough / Resoluteness (Guai)
    # Hidden risk surfaces. Act decisively to expose and resolve.
    "breakthrough": {
        "hexagram": 43,
        "name": "Breakthrough",
        "trait": "decisive exposure of hidden risk",
        "posture": "high_anomaly_transparency",
        "conditions": {
            # Triggers when: hidden patterns surface (unregistered agent + high value)
            # or ERC-8004 mismatch detected
            "min_anomaly": 0.3,
            "required_signals_any": [
                "erc8004:unregistered_agent",
                "evm:unvalidated_value_transfer",
                "erc8004:low_reputation",
            ],
        },
        "recommendation_override": "escalate_and_surface_risk",
    },

    # Hexagram 61 — Inner Truth (Zhong Fu)
    # Alignment between declared intent and observed behavior.
    # The core verification archetype.
    "inner_truth": {
        "hexagram": 61,
        "name": "Inner Truth",
        "trait": "alignment between intent and action",
        "posture": "verify_intent_alignment",
        "conditions": {
            # Triggers when: event is verifiable and clean
            "max_anomaly": 0.15,
            "behavior_class": "normal",
            "required_signals_none": [
                "erc8004:unregistered_agent",
                "erc8004:low_reputation",
            ],
        },
        "recommendation_override": None,  # trust confirmed, no override needed
    },

    # Hexagram 29 — The Abysmal / Repeated Danger (Kan)
    # Danger encountered repeatedly. Move carefully, don't freeze.
    "abysmal": {
        "hexagram": 29,
        "name": "The Abysmal",
        "trait": "repeated risk requiring careful navigation",
        "posture": "heightened_caution",
        "conditions": {
            # Triggers when: velocity spike + high risk action
            "min_anomaly": 0.25,
            "required_signals_any": [
                "velocity_spike",
                "velocity_elevated",
            ],
        },
        "recommendation_override": "throttle_and_require_validation",
    },

    # Hexagram 15 — Modesty (Qian)
    # Restraint. Don't overreach. Conservative when uncertain.
    "modesty": {
        "hexagram": 15,
        "name": "Modesty",
        "trait": "restraint and conservative judgment",
        "posture": "conservative_fallback",
        "conditions": {
            # Triggers when: new agent + moderate risk (uncertainty)
            "min_anomaly": 0.1,
            "max_anomaly": 0.35,
            "required_signals_any": [
                "new_agent:first_event",
                "missing_chain_context",
            ],
        },
        "recommendation_override": "monitor_with_restraint",
    },

    # Hexagram 12 — Standstill / Stagnation (Pi)
    # Blocked state. Execution should not proceed.
    "standstill": {
        "hexagram": 12,
        "name": "Standstill",
        "trait": "blocked execution, system obstruction",
        "posture": "halt_execution",
        "conditions": {
            # Triggers when: critical anomaly, policy risk
            "min_anomaly": 0.6,
            "behavior_class": "suspicious",
        },
        "recommendation_override": "recommend_deny_or_manual_review",
    },

    # Hexagram 33 — Retreat (Dun)
    # Strategic withdrawal. Reduce exposure, tighten permissions.
    "retreat": {
        "hexagram": 33,
        "name": "Retreat",
        "trait": "strategic de-escalation and permission reduction",
        "posture": "reduce_exposure",
        "conditions": {
            # Triggers when: policy_risk + high value
            "min_anomaly": 0.35,
            "required_signals_any": [
                "high_value",
                "elevated_value",
            ],
        },
        "recommendation_override": "recommend_throttle",
    },

    # Hexagram 25 — Innocence / The Unexpected (Wu Wang)
    # Clean action without hidden agenda. Normal operations.
    "innocence": {
        "hexagram": 25,
        "name": "Innocence",
        "trait": "clean action, no hidden patterns",
        "posture": "normal_operations",
        "conditions": {
            # Triggers when: low anomaly, no flags
            "max_anomaly": 0.08,
            "behavior_class": "normal",
        },
        "recommendation_override": None,
    },

    # Hexagram 48 — The Well (Jing)
    # Deep resource, reliable foundation. System operating from depth.
    "the_well": {
        "hexagram": 48,
        "name": "The Well",
        "trait": "reliable infrastructure, deep operational trust",
        "posture": "trusted_operations",
        "conditions": {
            # Triggers when: ERC-8004 registered + good reputation + low anomaly
            "max_anomaly": 0.12,
            "required_signals_any": [
                "erc8004:registered",
                "erc8004:good_reputation",
            ],
        },
        "recommendation_override": None,
    },

    # Hexagram 63 — After Completion (Ji Ji)
    # Process completed but fragile. Vigilance required.
    "after_completion": {
        "hexagram": 63,
        "name": "After Completion",
        "trait": "completed process requiring continued vigilance",
        "posture": "post_settlement_vigilance",
        "conditions": {
            # Triggers when: settlement/escrow events with low-moderate risk
            "max_anomaly": 0.3,
            "required_signals_any": [
                "evm:escrow_flow",
            ],
        },
        "recommendation_override": "monitor",
    },
}


def resolve_archetype(anomaly_score: float, behavior_class: str,
                       signals: list[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve the governance archetype for the current situation.

    Evaluates conditions for each archetype against the current signal state.
    Returns (archetype_name, governance_posture) or (None, None).

    Priority order matters — first match wins.
    Ordered from most specific/critical to most general.
    """
    def has_signal_match(signal_patterns: list[str]) -> bool:
        """Check if any signal starts with any of the patterns."""
        for pattern in signal_patterns:
            for signal in signals:
                if signal.startswith(pattern):
                    return True
        return False

    def has_no_signal_match(signal_patterns: list[str]) -> bool:
        """Check that no signal starts with any of the patterns."""
        for pattern in signal_patterns:
            for signal in signals:
                if signal.startswith(pattern):
                    return False
        return True

    # Evaluate archetypes in priority order
    priority_order = [
        "standstill",        # Critical — check first
        "abysmal",           # Repeated danger
        "retreat",           # High value + risk
        "breakthrough",      # Hidden risk surfacing
        "after_completion",  # Post-settlement
        "modesty",           # Uncertainty / new agent
        "the_well",          # Trusted agent
        "inner_truth",       # Intent alignment
        "innocence",         # Clean / normal (fallback)
    ]

    for archetype_key in priority_order:
        archetype = GOVERNANCE_ARCHETYPES[archetype_key]
        conditions = archetype["conditions"]
        matched = True

        # Check anomaly bounds
        if "min_anomaly" in conditions and anomaly_score < conditions["min_anomaly"]:
            matched = False
        if "max_anomaly" in conditions and anomaly_score > conditions["max_anomaly"]:
            matched = False

        # Check behavior_class
        if "behavior_class" in conditions and behavior_class != conditions["behavior_class"]:
            matched = False

        # Check required signals (any match)
        if matched and "required_signals_any" in conditions:
            if not has_signal_match(conditions["required_signals_any"]):
                matched = False

        # Check required signals (none should match)
        if matched and "required_signals_none" in conditions:
            if not has_no_signal_match(conditions["required_signals_none"]):
                matched = False

        if matched:
            return archetype_key, archetype["posture"]

    return None, None


class ORAEngine:
    """
    Observational Risk Agent.
    Interprets execution events and produces context for VERITY.

    Responsibilities:
      1. Signal interpretation — read telemetry from LITMUS
      2. Anomaly detection — flag unusual patterns
      3. Behavioral classification — categorize agent behavior
      4. Context enrichment — produce ORAContext for VERITY
    """

    @staticmethod
    def interpret(event: ExecutionEvent) -> ORAContext:
        """
        Main interpretation function.
        Reads the event, analyzes signals, returns enriched context.
        """
        signals: list[str] = []
        anomaly_score = 0.0
        behavior_class = "normal"
        risk_signal = "low"
        policy_flag = False
        recommendation = None

        # --- Signal 1: Action type risk weighting ---
        high_risk_actions = {
            "payment_attempt", "transfer", "withdrawal", "contract_deploy",
            "permission_change", "key_rotation", "delegation",
        }
        medium_risk_actions = {
            "swap", "stake", "unstake", "approve", "mint",
        }

        if event.action_type in high_risk_actions:
            anomaly_score += 0.15
            signals.append(f"high_risk_action:{event.action_type}")
        elif event.action_type in medium_risk_actions:
            anomaly_score += 0.05
            signals.append(f"medium_risk_action:{event.action_type}")

        # --- Signal 2: High-value transaction detection ---
        amount = event.action_payload.get("amount")
        if amount is not None:
            try:
                amount = float(amount)
                if amount > HIGH_VALUE_THRESHOLD:
                    anomaly_score += 0.20
                    signals.append(f"high_value:{amount}")
                elif amount > HIGH_VALUE_THRESHOLD * 0.5:
                    anomaly_score += 0.08
                    signals.append(f"elevated_value:{amount}")
            except (ValueError, TypeError):
                pass

        # --- Signal 3: Velocity check (event frequency via store) ---
        agent_id = event.agent_id
        event_count = state.get_agent_event_count(agent_id)

        if event_count >= VELOCITY_MAX_EVENTS:
            anomaly_score += 0.25
            signals.append(f"velocity_spike:{event_count}_in_{VELOCITY_WINDOW_SECONDS}s")
            policy_flag = True
        elif event_count >= VELOCITY_MAX_EVENTS * 0.7:
            anomaly_score += 0.10
            signals.append(f"velocity_elevated:{event_count}_in_{VELOCITY_WINDOW_SECONDS}s")

        # Record this event in history (persisted in Redis or memory)
        state.record_agent_event(agent_id, event.event_id, event.action_type)

        # --- Signal 4: New agent detection ---
        if event_count == 0:
            anomaly_score += 0.05
            signals.append("new_agent:first_event")

        # --- Signal 5: Environment anomalies ---
        chain = event.environment.get("chain", "")
        network = event.environment.get("network", "")
        if network in ("devnet", "testnet") and event.action_type in high_risk_actions:
            signals.append(f"testnet_high_risk:{network}:{event.action_type}")
        if not chain:
            anomaly_score += 0.05
            signals.append("missing_chain_context")

        # --- Signal 6: Replay detection ---
        if event.parent_event_id:
            anomaly_score += 0.03
            signals.append(f"replay_of:{event.parent_event_id}")

        # --- Signal 7: EVM/ERC-8004 context (if enriched) ---
        evm_ctx = event.metadata.get("evm_context", {})
        erc8004 = event.metadata.get("erc8004_profile", {})

        if evm_ctx:
            evm_chain = evm_ctx.get("chain", "unknown")
            # Normalize: might be ChainType enum value or string
            if hasattr(evm_chain, 'value'):
                evm_chain = evm_chain.value
            evm_chain = str(evm_chain).replace("ChainType.", "").lower()
            if evm_chain in ("base", "ethereum"):
                signals.append(f"evm_chain:{evm_chain}")

                # Check if agent is ERC-8004 registered
                identity = erc8004.get("identity", {})
                if identity.get("registered"):
                    # Registered agent — reduce anomaly (trusted)
                    anomaly_score = max(0, anomaly_score - 0.10)
                    signals.append("erc8004:registered")
                elif evm_ctx.get("agent_address"):
                    # Has address but not registered — flag it
                    anomaly_score += 0.08
                    signals.append("erc8004:unregistered_agent")

                # Check reputation
                reputation = erc8004.get("reputation", {})
                feedback_count = reputation.get("feedback_count", 0)
                positive_ratio = reputation.get("positive_ratio")
                if feedback_count > 0 and positive_ratio is not None:
                    if positive_ratio < 0.5:
                        anomaly_score += 0.15
                        signals.append(f"erc8004:low_reputation:{positive_ratio}")
                    elif positive_ratio > 0.8:
                        anomaly_score = max(0, anomaly_score - 0.05)
                        signals.append(f"erc8004:good_reputation:{positive_ratio}")

                # High-value EVM transaction without validation
                validation = erc8004.get("validation", {})
                if evm_ctx.get("value_wei") and not validation.get("validated"):
                    anomaly_score += 0.05
                    signals.append("evm:unvalidated_value_transfer")

                # Commerce event classification
                if evm_ctx.get("escrow_id"):
                    signals.append(f"evm:escrow_flow:{evm_ctx['escrow_id']}")
                if evm_ctx.get("tx_hash"):
                    signals.append(f"evm:tx_anchored:{evm_ctx['tx_hash'][:16]}")

        # --- Classify behavior ---
        anomaly_score = min(anomaly_score, 1.0)

        if anomaly_score >= 0.6:
            behavior_class = "suspicious"
            risk_signal = "critical"
            recommendation = "recommend_deny_or_manual_review"
            policy_flag = True
        elif anomaly_score >= 0.35:
            behavior_class = "policy_risk"
            risk_signal = "high"
            recommendation = "recommend_throttle"
            policy_flag = True
        elif anomaly_score >= 0.15:
            behavior_class = "edge_case"
            risk_signal = "medium"
            recommendation = "monitor"
        else:
            behavior_class = "normal"
            risk_signal = "low"
            recommendation = None

        # --- Resolve governance archetype ---
        archetype, governance_posture = resolve_archetype(
            anomaly_score, behavior_class, signals
        )

        # Archetype can override recommendation (governance posture takes priority)
        if archetype:
            archetype_def = GOVERNANCE_ARCHETYPES.get(archetype, {})
            override = archetype_def.get("recommendation_override")
            if override:
                recommendation = override
            signals.append(f"archetype:{archetype}")

        return ORAContext(
            anomaly_score=round(anomaly_score, 4),
            behavior_class=behavior_class,
            risk_signal=risk_signal,
            policy_flag=policy_flag,
            signals=signals,
            recommendation=recommendation,
            archetype=archetype,
            governance_posture=governance_posture,
        )


# Singleton engine instance
ora_engine = ORAEngine()


async def pipeline_ora_interpret(event: ExecutionEvent):
    """Step 3: ORA interpretation — analyze signals, enrich context.

    After ORA collects signals and classifies behavior, the Mars Engineer
    runs thermodynamic analysis to modulate anomaly_score based on
    signal coherence, contradictions, and energy efficiency.
    """
    t = time.time()
    context = ora_engine.interpret(event)

    # --- Mars Engineer: thermodynamic modulation ---
    # Run after ORA signals are collected, before archetype resolution
    event_count = state.get_agent_event_count(event.agent_id)
    rep_band = event.policy_context.reputation_band

    thermal = mars_engineer.evaluate(
        signals=context.signals,
        anomaly_score=context.anomaly_score,
        behavior_class=context.behavior_class,
        action_type=event.action_type,
        reputation_band=rep_band,
        event_count=event_count,
    )

    # Apply thermal adjustment to anomaly score
    original_anomaly = context.anomaly_score
    adjusted_anomaly = max(0.0, min(1.0,
        context.anomaly_score + thermal.thermal_adjustment))
    context.anomaly_score = round(adjusted_anomaly, 4)

    # If thermal analysis shifts behavior class, update it
    if adjusted_anomaly >= 0.6 and context.behavior_class != "suspicious":
        context.behavior_class = "suspicious"
        context.risk_signal = "critical"
        context.policy_flag = True
    elif adjusted_anomaly < 0.15 and original_anomaly >= 0.15:
        # Thermal cooling relaxed the score below edge_case threshold
        context.behavior_class = "normal"
        context.risk_signal = "low"

    # Mars recommendation can override ORA recommendation
    if thermal.recommendation:
        if thermal.recommendation.startswith("mars:emergency_halt"):
            context.recommendation = "recommend_deny_or_manual_review"
        elif thermal.recommendation.startswith("mars:restrict"):
            context.recommendation = context.recommendation or "recommend_throttle"
        elif thermal.recommendation.startswith("mars:relax") and \
             context.recommendation == "monitor":
            # Only relax if ORA wasn't already flagging something serious
            context.recommendation = None

    # Add thermal signals to ORA signal list
    context.signals.append(f"thermal:{thermal.thermal_state.value}")
    if thermal.contradiction_count > 0:
        context.signals.append(
            f"thermal:contradictions:{thermal.contradiction_count}")
    if not thermal.is_efficient:
        context.signals.append("thermal:inefficient_governance")

    # Store thermal assessment in event metadata
    event.metadata["thermal"] = {
        "thermal_state": thermal.thermal_state.value,
        "energy_budget": thermal.energy_budget.value,
        "entropy_score": thermal.entropy_score,
        "thermal_adjustment": thermal.thermal_adjustment,
        "signal_coherence": thermal.signal_coherence,
        "contradiction_count": thermal.contradiction_count,
        "wasted_energy": thermal.wasted_energy,
        "original_anomaly": original_anomaly,
        "adjusted_anomaly": adjusted_anomaly,
        "reason_codes": thermal.reason_codes,
        "recommendation": thermal.recommendation,
    }

    event.ora_context = context

    add_trace(event, "ora", EventState.INTERPRETED, {
        **context.model_dump(),
        "mars_engineer": {
            "thermal_state": thermal.thermal_state.value,
            "entropy": thermal.entropy_score,
            "adjustment": thermal.thermal_adjustment,
            "coherence": thermal.signal_coherence,
            "contradictions": thermal.contradiction_count,
        },
    }, t)


async def pipeline_verity_score(event: ExecutionEvent):
    """Step 5: VERITY integrity scoring.

    VERITY at identityaware.pages.dev:
      GET /agents/index — returns scored agent list (free)
      GET /health — health check
      POST /integrity/score — full scoring with x402 (paid)
      POST /integrity/check — lightweight free endpoint (OROS-internal)

    Strategy:
      1. Try free /integrity/check first (purpose-built for OROS)
      2. Fall back to /agents/index lookup
      3. If both fail, degrade to no-score mode (flag, don't block)
    """
    t = time.time()

    if not VERITY_URL:
        add_trace(event, "verity", EventState.SCORED,
                  {"status": "skipped", "reason": "no_url_configured"}, t)
        return

    # Strategy 1: Try the free internal endpoint
    check_result = await call_module(VERITY_URL, "/integrity/check", {
        "agent_id": event.agent_id,
        "action_type": event.action_type,
    }, method="POST")

    if check_result.get("status") not in ("error", "skipped"):
        # Got a direct integrity check response
        event.integrity_score = check_result.get("integrity_score") or \
                                check_result.get("ais") or \
                                check_result.get("coherence")
        event.metadata["verity_method"] = "integrity_check"

        # Capture additional VERITY context
        if check_result.get("identity_state"):
            event.metadata["verity_identity"] = check_result["identity_state"]
        if check_result.get("decision"):
            event.metadata["verity_decision"] = check_result["decision"]

        add_trace(event, "verity", EventState.SCORED, {
            "method": "integrity_check",
            "ais": event.integrity_score,
            "decision": check_result.get("decision"),
            "coherence": check_result.get("coherence"),
        }, t)
        return

    # Strategy 2: Fall back to /agents/index lookup
    result = await call_module(VERITY_URL, "/agents/index", {},
                                method="GET")

    if result.get("status") == "error":
        # VERITY unavailable — degrade: flag but don't block
        event.integrity_score = None
        event.metadata["verity_degraded"] = True
        add_trace(event, "verity", EventState.SCORED, {
            "status": "degraded",
            "reason": result.get("reason", "unknown"),
            "fallback": "no_integrity_score_available",
        }, t)
        return

    # Find agent in VERITY's index
    agents = result if isinstance(result, list) else result.get("agents", [])
    agent_id = event.agent_id.lower()
    matched_agent = None
    for agent in agents:
        wallet = (agent.get("wallet") or agent.get("agent_id") or "").lower()
        if wallet == agent_id or agent_id in wallet:
            matched_agent = agent
            break

    if matched_agent:
        event.integrity_score = matched_agent.get("ais") or matched_agent.get("score")
        event.metadata["verity_method"] = "index_lookup"
        add_trace(event, "verity", EventState.SCORED, {
            "method": "index_lookup",
            "agent_found": True,
            "ais": event.integrity_score,
            "tier": matched_agent.get("tier"),
            "win_rate": matched_agent.get("winRate"),
        }, t)
    else:
        # Agent not in VERITY — unknown reputation
        event.integrity_score = None
        add_trace(event, "verity", EventState.SCORED, {
            "method": "index_lookup",
            "agent_found": False,
            "reason": "agent_not_in_verity_index",
        }, t)


async def pipeline_shield_route(event: ExecutionEvent):
    """Step 7: Shield Router — SURVIVOR Oracle risk scoring + RPE policy.

    SURVIVOR Oracle API:
      POST /attest — score a mint/target, returns signed attestation
        Headers: x-api-key
        Body: { mint, router_program_id }
      POST /rpe/evaluate — ALLOW/CHALLENGE/DENY policy decision
        Headers: x-api-key
        Body: { mint, amount, action }

    If SURVIVOR is unavailable:
      - Sensitive actions (payments, transfers) → fail closed (DENY)
      - Non-sensitive actions → degrade to ALLOW with warning
    """
    t = time.time()

    if not SHIELD_URL:
        add_trace(event, "shield_router", EventState.ROUTED,
                  {"status": "skipped", "reason": "no_url_configured"}, t)
        return

    headers = {}
    if SURVIVOR_API_KEY:
        headers["x-api-key"] = SURVIVOR_API_KEY

    # Extract target for attestation
    target = (event.action_payload.get("mint")
              or event.action_payload.get("target")
              or event.action_payload.get("contract_address")
              or "")

    # Step A: Get risk attestation from SURVIVOR Oracle
    attest_result = {}
    if target:
        attest_result = await call_module(SHIELD_URL, "/attest", {
            "mint": target,
            "router_program_id": os.getenv("SHIELD_ROUTER_PROGRAM_ID", ""),
        }, headers=headers)

        if attest_result.get("status") == "error":
            # SURVIVOR unavailable — check if action is sensitive
            sensitive_actions = {"payment_attempt", "transfer", "withdrawal",
                                 "contract_deploy", "delegation"}
            if event.action_type in sensitive_actions:
                # Fail closed for sensitive actions
                event.policy_context.decision = PolicyDecision.DENY
                event.policy_context.reason = "survivor_unavailable:fail_closed"
                add_trace(event, "shield_router", EventState.ROUTED, {
                    "status": "fail_closed",
                    "reason": attest_result.get("reason"),
                    "action": "sensitive_action_blocked",
                }, t)
                return
            else:
                # Degrade for non-sensitive
                add_trace(event, "shield_router", EventState.ROUTED, {
                    "status": "degraded",
                    "reason": attest_result.get("reason"),
                    "fallback": "allow_with_warning",
                }, t)
                return

    # Extract risk score from attestation response
    # SURVIVOR /attest returns: { attestation: {...}, signature: ..., signer: ..., meta: { score, risk_level, policy: { decision } } }
    meta = attest_result.get("meta", {})
    attestation = attest_result.get("attestation", {})
    risk_score = meta.get("score") or attestation.get("score")
    risk_level = meta.get("risk_level")
    if risk_score is not None:
        event.risk_score = risk_score

    # Check if attestation already contains a policy decision
    policy = meta.get("policy", {})
    rpe_decision = policy.get("decision", "").upper()

    # If attestation includes inline policy, use it directly
    if rpe_decision in ("ALLOW", "DENY", "CHALLENGE"):
        pass  # use the inline decision
    else:
        # Step B: Fallback — call RPE separately with full attestation
        rpe_result = await call_module(SHIELD_URL, "/rpe/evaluate", {
            "attestation": attestation,
            "signature": attest_result.get("signature"),
            "signer": attest_result.get("signer"),
            "meta": meta,
            "amount_usd": event.action_payload.get("amount"),
        }, headers=headers)

        rpe_decision = rpe_result.get("decision", "").upper()
        policy = rpe_result

    # Map decision to policy
    if rpe_decision == "DENY":
        event.policy_context.decision = PolicyDecision.DENY
        event.policy_context.reason = f"rpe:deny:risk_level={risk_level}"
    elif rpe_decision == "CHALLENGE":
        event.policy_context.decision = PolicyDecision.THROTTLE
        event.policy_context.reason = f"rpe:challenge:risk_level={risk_level}"
    elif rpe_decision == "ALLOW":
        # Don't override if already denied by earlier stage
        if event.policy_context.decision != PolicyDecision.DENY:
            event.policy_context.decision = PolicyDecision.ALLOW
            event.policy_context.reason = f"rpe:allow:score={risk_score}:risk_level={risk_level}"

    add_trace(event, "shield_router", EventState.ROUTED, {
        "attestation": {
            "score": risk_score,
            "risk_level": risk_level,
            "tier": attestation.get("tier"),
        },
        "policy": {
            "decision": rpe_decision,
            "reason_codes": policy.get("reason_codes", []),
            "limits": policy.get("limits", {}),
        },
    }, t)


# ---------------------------------------------------------------------------
# Full pipeline execution
# ---------------------------------------------------------------------------

async def run_pipeline(event: ExecutionEvent) -> ExecutionEvent:
    """
    Execute the full trace pipeline:
      EVM Enrich → IAM → Reputation Governor → LITMUS → ORA → Intent Verify → VERITY → VYRE → Shield Router
    """
    try:
        # Step 0: EVM context enrichment (chain-aware metadata)
        t = time.time()
        try:
            enrichment = await enrich_with_evm_context(event)
            event.metadata.update(enrichment)
            evm_ctx = enrichment.get("evm_context", {})
            chain_info = f"{evm_ctx.get('chain', 'unknown')}:{evm_ctx.get('network', 'unknown')}"
            add_trace(event, "evm_adapter", EventState.RECEIVED,
                      {"chain": chain_info, "has_erc8004": "erc8004_profile" in enrichment}, t)
        except Exception as e:
            add_trace(event, "evm_adapter", EventState.RECEIVED,
                      {"status": "enrichment_failed", "reason": str(e)}, t)

        # Step 1: IAM + Governor
        authorized = await pipeline_iam_governor(event)
        if not authorized:
            # Still create artifact for denied events (audit trail)
            artifact = create_artifact(event)
            event.artifact_hash = artifact.event_hash
            state.store_artifact(event.event_id, artifact.model_dump())
            add_trace(event, "vyre", EventState.ARTIFACTED,
                      {"artifact_hash": artifact.event_hash}, time.time())
            finalize_governance_summary(event)
            try:
                telemetry_engine.emit(event.model_dump())
            except Exception:
                pass
            return event

        # Step 1.5: Reputation-informed governance
        rep_allowed = await pipeline_reputation_governor(event)
        if not rep_allowed:
            # Reputation blocked — still create artifact
            artifact = create_artifact(event)
            event.artifact_hash = artifact.event_hash
            state.store_artifact(event.event_id, artifact.model_dump())
            add_trace(event, "vyre", EventState.ARTIFACTED,
                      {"artifact_hash": artifact.event_hash}, time.time())
            finalize_governance_summary(event)
            try:
                telemetry_engine.emit(event.model_dump())
            except Exception:
                pass
            return event

        # Step 2: LITMUS observe
        await pipeline_litmus_observe(event)

        # Step 3: ORA interpret (internal — no network call)
        await pipeline_ora_interpret(event)

        # Step 3.5: Hexagram state classification
        t = time.time()
        try:
            hex_state = hexagram_engine.classify_from_event(event.model_dump())
            event.metadata["hexagram"] = {
                "state": hex_state.archetype.value,
                "index": hex_state.state_index,
                "bits": list(hex_state.bits),
                "strategy": hex_state.strategy,
                "triggered": hex_state.triggered_conditions,
            }
            add_trace(event, "hexagram_engine", EventState.INTERPRETED, {
                "state": hex_state.archetype.value,
                "index": hex_state.state_index,
                "bits": list(hex_state.bits),
                "triggered": hex_state.triggered_conditions,
            }, t)
        except Exception as e:
            add_trace(event, "hexagram_engine", EventState.INTERPRETED, {
                "status": "classification_failed",
                "reason": str(e),
            }, t)

        # Step 3.6: State memory + transition tracking
        t = time.time()
        try:
            hex_data = event.metadata.get("hexagram", {})
            thermal_data = event.metadata.get("thermal", {})
            anima_potentia.state_memory.record(
                archetype=hex_data.get("state", "unknown"),
                state_index=hex_data.get("index", 0),
                thermal_state=thermal_data.get("thermal_state"),
                anomaly_score=event.ora_context.anomaly_score,
            )
            memory_summary = anima_potentia.state_memory.get_summary()
            event.metadata["state_memory"] = memory_summary
            add_trace(event, "state_memory", EventState.INTERPRETED, memory_summary, t)
        except Exception as e:
            add_trace(event, "state_memory", EventState.INTERPRETED, {
                "status": "memory_failed", "reason": str(e),
            }, t)

        # Step 3.7: Physics kernel — governance feasibility check
        t = time.time()
        try:
            amount = event.action_payload.get("amount", 0)
            try:
                amount_f = float(amount)
            except (ValueError, TypeError):
                amount_f = 0.0

            rep_data = reputation_ledger.get_reputation(event.agent_id)
            rep_score = rep_data.get("score", 50.0)
            pipeline_latency = sum(
                s.latency_ms for s in event.trace if s.latency_ms
            )

            physics = assess_governance_physics(
                spend_limit=event.policy_context.spend_limit or 5000.0,
                amount=amount_f,
                anomaly_score=event.ora_context.anomaly_score,
                reputation_score=rep_score,
                dependency_count=max(1, len(event.trace)),
                pipeline_latency_ms=pipeline_latency,
                verification_cost=pipeline_latency * 0.01,
            )
            event.metadata["physics"] = {
                "force_balance": physics.force_balance,
                "efficiency": physics.efficiency,
                "efficiency_class": physics.efficiency_class,
                "can_complete_mission": physics.can_complete_mission,
                "fuel_margin": physics.fuel_margin,
                "physics_score": physics.physics_score,
                "recommendation": physics.recommendation,
                "reason_codes": physics.reason_codes,
            }
            add_trace(event, "physics_kernel", EventState.INTERPRETED, {
                "score": physics.physics_score,
                "force": physics.force_balance,
                "efficiency": physics.efficiency_class,
                "feasible": physics.can_complete_mission,
                "recommendation": physics.recommendation,
            }, t)

            # Physics-informed decision composition
            # If physics says infeasible, governance cannot return plain ALLOW
            if not physics.can_complete_mission:
                if physics.physics_score < 0.25:
                    # Very low feasibility — defer execution
                    if event.policy_context.decision not in (
                        PolicyDecision.DENY,
                    ):
                        event.policy_context.decision = PolicyDecision.DEFER
                        event.policy_context.reason = (
                            f"physics_defer:score={physics.physics_score}:"
                            f"force={physics.force_balance}:"
                            f"mission_infeasible"
                        )
                elif event.policy_context.decision == PolicyDecision.ALLOW:
                    # Infeasible but not critically so — constrained allow
                    event.policy_context.decision = PolicyDecision.ALLOW_WITH_GUARDRAILS
                    event.policy_context.reason = (
                        f"physics_guardrails:score={physics.physics_score}:"
                        f"force={physics.force_balance}:"
                        f"efficiency={physics.efficiency_class}"
                    )
                    event.metadata["execution_constraints"] = {
                        "mode": "restricted",
                        "retry_budget": 0,
                        "simulation_preferred": True,
                        "human_review_recommended": True,
                    }
            elif physics.recommendation == "physics:caution:marginal_feasibility":
                # Feasible but marginal — flag it
                if event.policy_context.decision == PolicyDecision.ALLOW:
                    event.metadata.setdefault("execution_constraints", {})
                    event.metadata["execution_constraints"]["marginal_physics"] = True
        except Exception as e:
            add_trace(event, "physics_kernel", EventState.INTERPRETED, {
                "status": "physics_failed", "reason": str(e),
            }, t)

        # Step 3.8: Anima Potentia — polarity balance check
        t = time.time()
        try:
            hex_data = event.metadata.get("hexagram", {})
            thermal_data = event.metadata.get("thermal", {})
            verification_count = sum(
                1 for s in event.trace
                if s.module in ("intent_verifier", "verity", "shield_router",
                                "iam/governor")
            )
            policy_restrictions = sum(
                1 for s in event.trace
                if s.output.get("spend_blocked") or s.output.get("sensitive_blocked")
            )
            signals_list = event.ora_context.signals
            new_agent = any("new_agent" in s for s in signals_list)
            new_chain = any("missing_chain" in s for s in signals_list)
            spend_limit = event.policy_context.spend_limit or 5000.0
            spend_ratio = min(amount_f / max(spend_limit, 0.01), 1.0) if amount_f else 0.0

            balance = anima_potentia.assess_balance(
                anomaly_score=event.ora_context.anomaly_score,
                verification_count=verification_count,
                policy_restrictions=policy_restrictions,
                recovery_ready=False,  # GhostLedger integration future
                pipeline_latency_ms=pipeline_latency,
                new_agent=new_agent,
                new_chain=new_chain,
                spend_ratio=spend_ratio,
                reputation_score=rep_score,
                thermal_state=thermal_data.get("thermal_state"),
                hexagram_archetype=hex_data.get("state"),
            )
            event.metadata["polarity"] = {
                "overall_balance": balance.overall_balance,
                "dominant_polarity": balance.dominant_polarity,
                "system_state": balance.system_state,
                "counterweights": balance.counterweights,
                "reason_codes": balance.reason_codes,
            }
            add_trace(event, "anima_potentia", EventState.INTERPRETED, {
                "balance": balance.overall_balance,
                "state": balance.system_state,
                "dominant": balance.dominant_polarity,
                "counterweights_count": len(balance.counterweights),
            }, t)

            # Polarity enforcement — convert counterweights to constraints
            if balance.system_state == "chaotic" and balance.counterweights:
                constraints = event.metadata.setdefault("execution_constraints", {})
                constraints["polarity_state"] = "chaotic"

                for cw in balance.counterweights:
                    if "increase_control" in cw:
                        # Reduce spend limit by 30%
                        current = event.policy_context.spend_limit or 5000.0
                        event.policy_context.spend_limit = round(current * 0.7, 2)
                        constraints["spend_limit_reduced_by_polarity"] = True
                    if "increase_safety" in cw:
                        constraints["max_parallel_actions"] = 1
                        constraints["cooldown_seconds"] = 30
                    if "increase_verification" in cw:
                        constraints["require_artifact_verification"] = True
                        constraints["allow_novel_targets"] = False
                    if "increase_survival" in cw:
                        constraints["daily_risk_budget_multiplier"] = 0.5

                # If chaotic + ALLOW, upgrade to guardrails
                if event.policy_context.decision == PolicyDecision.ALLOW:
                    event.policy_context.decision = PolicyDecision.ALLOW_WITH_GUARDRAILS
                    event.policy_context.reason = (
                        f"polarity_chaotic:counterweights={len(balance.counterweights)}:"
                        f"dominant={balance.dominant_polarity}"
                    )

            elif balance.system_state == "rigid":
                # Over-controlled — note it but don't block
                constraints = event.metadata.setdefault("execution_constraints", {})
                constraints["polarity_state"] = "rigid"
                constraints["note"] = "system_over_governed:consider_relaxing"

            elif balance.system_state == "frozen":
                # Neither side active — escalate
                if event.policy_context.decision in (
                    PolicyDecision.ALLOW, PolicyDecision.ALLOW_WITH_GUARDRAILS
                ):
                    event.policy_context.decision = PolicyDecision.SIMULATE_ONLY
                    event.policy_context.reason = "polarity_frozen:no_active_governance"
        except Exception as e:
            add_trace(event, "anima_potentia", EventState.INTERPRETED, {
                "status": "balance_check_failed", "reason": str(e),
            }, t)

        # Step 4: Intent Verification (transaction simulation)
        t = time.time()
        try:
            iv_result = await intent_verifier.verify(event)
            event.intent_verification = iv_result.model_dump()

            add_trace(event, "intent_verifier", EventState.VERIFIED, {
                "intent_match": iv_result.intent_match.value,
                "intent_risk_score": iv_result.intent_risk_score,
                "simulation_method": iv_result.simulation_method,
                "warnings_count": len(iv_result.warnings),
            }, t)
        except Exception as e:
            add_trace(event, "intent_verifier", EventState.VERIFIED, {
                "status": "verification_failed",
                "reason": str(e),
            }, t)

        # Step 4.5: Intent-aware policy adjustment
        t = time.time()
        iv_data = event.intent_verification or {}
        intent_state = iv_data.get("intent_match", "unverifiable")
        intent_risk = iv_data.get("intent_risk_score", 0.0)
        current_limit = event.policy_context.spend_limit or 5000.0
        intent_adjustment = None

        if intent_state == "mismatch":
            # Mismatch: deny or heavy restriction
            if intent_risk >= 0.5:
                event.policy_context.decision = PolicyDecision.DENY
                event.policy_context.reason = (
                    f"intent_mismatch:risk={intent_risk}:execution_blocked"
                )
                intent_adjustment = "blocked"
            else:
                # Lower risk mismatch: throttle + cut limit 70%
                event.policy_context.decision = PolicyDecision.THROTTLE
                event.policy_context.spend_limit = round(current_limit * 0.3, 2)
                event.policy_context.reason = (
                    f"intent_mismatch:risk={intent_risk}:limit_reduced_70pct"
                )
                intent_adjustment = "limit_reduced_70pct"

        elif intent_state == "unverifiable":
            # Unverifiable: reduce limits based on action type
            sensitive_actions = {
                "payment_attempt", "transfer", "withdrawal",
                "contract_deploy", "delegation",
            }
            if event.action_type in sensitive_actions:
                # Sensitive + unverifiable: cut limit 40%
                event.policy_context.spend_limit = round(current_limit * 0.6, 2)
                intent_adjustment = "sensitive_unverifiable:limit_reduced_40pct"
            else:
                # Non-sensitive + unverifiable: mild reduction 15%
                event.policy_context.spend_limit = round(current_limit * 0.85, 2)
                intent_adjustment = "unverifiable:limit_reduced_15pct"

        elif intent_state == "sim_failed":
            # Simulation failed: treat like unverifiable but note it
            event.policy_context.spend_limit = round(current_limit * 0.7, 2)
            intent_adjustment = "sim_failed:limit_reduced_30pct"

        elif intent_state == "match":
            # Verified match: reward with limit boost
            event.policy_context.spend_limit = round(current_limit * 1.2, 2)
            intent_adjustment = "verified:limit_boosted_20pct"

        elif intent_state == "suspicious":
            # Suspicious patterns: heavy cut
            event.policy_context.spend_limit = round(current_limit * 0.4, 2)
            event.policy_context.decision = PolicyDecision.THROTTLE
            event.policy_context.reason = (
                f"intent_suspicious:risk={intent_risk}:limit_reduced_60pct"
            )
            intent_adjustment = "suspicious:limit_reduced_60pct"

        add_trace(event, "intent_policy", EventState.AUTHORIZED, {
            "intent_state": intent_state,
            "intent_risk": intent_risk,
            "adjustment": intent_adjustment,
            "previous_limit": current_limit,
            "adjusted_limit": event.policy_context.spend_limit,
        }, t)

        # Step 5: VERITY score (receives ORA context + intent verification)
        await pipeline_verity_score(event)

        # Step 5: VYRE artifact (boundary signing)
        t = time.time()
        artifact = create_artifact(event)
        event.artifact_hash = artifact.event_hash
        state.store_artifact(event.event_id, artifact.model_dump())
        add_trace(event, "vyre", EventState.ARTIFACTED,
                  {"artifact_hash": artifact.event_hash}, t)

        # Step 6: Shield Router (receives ORA context + scores)
        await pipeline_shield_route(event)

        # Finalize governance summary
        finalize_governance_summary(event)

        # Emit telemetry — the control tower signal
        try:
            telemetry_engine.emit(event.model_dump())
        except Exception as te:
            logger.warning("Telemetry emission failed: %s", te)

        # Galileo evaluation — external judgment on governance quality
        try:
            galileo_eval = galileo_engine.evaluate(event.model_dump())
            event.metadata["galileo"] = {
                "score": galileo_eval.galileo_score,
                "quality": galileo_eval.trajectory_quality,
                "proportionality": galileo_eval.proportionality,
                "grounded": galileo_eval.decision_grounded,
                "overhead_pct": galileo_eval.governance_overhead_pct,
                "regressions": galileo_eval.regression_flags,
                "recommendations": galileo_eval.recommendations,
            }
        except Exception as ge:
            logger.warning("Galileo evaluation failed: %s", ge)

        return event

    except Exception as e:
        add_trace(event, "pipeline", EventState.FAILED,
                  {"error": str(e)}, time.time())
        finalize_governance_summary(event)

        # Emit telemetry even for failed events
        try:
            telemetry_engine.emit(event.model_dump())
        except Exception:
            pass

        # Galileo eval for failed events too
        try:
            galileo_engine.evaluate(event.model_dump())
        except Exception:
            pass

        return event


def finalize_governance_summary(event: ExecutionEvent):
    """
    Populate the top-level governance summary from pipeline results.
    Called at the end of every pipeline execution path.

    This is the readable verdict — operators, auditors, and downstream
    services read this instead of reconstructing from traces.
    """
    iv = event.intent_verification or {}

    # Check if artifact signature is real
    artifact_verified = False
    signer = None
    if event.artifact_hash and VYRE_SIGNING_KEY:
        artifact_verified = True
        signer = SERVICE_NAME

    # Extract hexagram state from metadata
    hex_data = event.metadata.get("hexagram", {})

    event.governance = GovernanceSummary(
        decision=event.policy_context.decision.value if event.policy_context.decision else None,
        reputation_score=None,  # will be filled from trace
        reputation_band=event.policy_context.reputation_band,
        spend_limit=event.policy_context.spend_limit,
        risk_score=event.risk_score,
        risk_level=None,  # will be filled from shield trace
        archetype=event.ora_context.archetype,
        governance_posture=event.ora_context.governance_posture,
        hexagram_state=hex_data.get("state"),
        hexagram_index=hex_data.get("index"),
        hexagram_strategy=hex_data.get("strategy"),
        intent_match=iv.get("intent_match"),
        artifact_hash=event.artifact_hash,
        artifact_verified=artifact_verified,
        signer=signer,
    )

    # Enrich from trace data
    for step in event.trace:
        if step.module == "reputation_governor":
            event.governance.reputation_score = step.output.get("reputation_score")
        elif step.module == "shield_router":
            att = step.output.get("attestation", {})
            if att.get("risk_level"):
                event.governance.risk_level = att["risk_level"]
        elif step.module == "intent_policy":
            event.governance.intent_adjustment = step.output.get("adjustment")
            # Update spend_limit to reflect intent adjustment
            adj_limit = step.output.get("adjusted_limit")
            if adj_limit is not None:
                event.governance.spend_limit = adj_limit


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="execution-coordinator",
    version=SCHEMA_VERSION,
    description="Agent OS control-plane — execution trace pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    store_status = state.get_store_status()
    return {
        "service": SERVICE_NAME,
        "version": SCHEMA_VERSION,
        "status": "operational",
        "store": store_status,
        "modules": {
            "iam": IAM_URL[:40] + "..." if IAM_URL else "not_configured",
            "verity": VERITY_URL[:40] + "..." if VERITY_URL else "not_configured",
            "shield_router": SHIELD_URL[:40] + "..." if SHIELD_URL else "not_configured",
            "shield_api_key": "configured" if SURVIVOR_API_KEY else "not_configured",
            "litmus": LITMUS_URL[:40] + "..." if LITMUS_URL else "not_configured",
            "vyre_signing": "configured" if VYRE_SIGNING_KEY else "not_configured",
        },
        "failure_policy": {
            "sensitive_actions_without_shield": "fail_closed",
            "non_sensitive_without_shield": "degrade_allow",
            "verity_unavailable": "degrade_no_score",
        },
        "kernel_modules": {
            "mars_engineer": "active",
            "physics_kernel": "active",
            "anima_potentia": "active",
            "galileo": "active",
            "telemetry": "active",
            "state_memory": "active",
        },
    }


@app.get("/galileo/{event_id}")
async def get_galileo_evaluation(event_id: str):
    """Retrieve Galileo evaluation for a specific event."""
    result = galileo_engine.get_evaluation(event_id)
    if not result:
        raise HTTPException(status_code=404, detail="Galileo evaluation not found")
    return result


@app.get("/telemetry/dashboard")
async def telemetry_dashboard():
    """
    Control tower dashboard — aggregated governance telemetry.

    Shows:
      - total events
      - allow / throttle / deny counts
      - intent state distribution
      - router posture distribution
      - thermal state distribution
      - chain distribution
      - archetype distribution
      - notional volume by decision
      - escalation count
      - top reason codes
    """
    return telemetry_engine.get_dashboard()


@app.get("/telemetry/event/{event_id}")
async def get_telemetry_event(event_id: str):
    """Retrieve telemetry record for a specific event."""
    r = state._get_redis()
    key = f"telemetry:{event_id}"

    if r:
        value = r.get(key)
    else:
        value = getattr(state, '_mem_telemetry', {}).get(event_id)

    if not value:
        raise HTTPException(status_code=404, detail="Telemetry not found")

    return json.loads(value) if isinstance(value, str) else value


@app.post("/events", response_model=ExecutionEvent)
async def create_event(input: ExecutionEventInput):
    """
    Accept an execution event, run the full pipeline.
    Returns the completed event with trace + artifact hash.
    """
    now = datetime.now(timezone.utc).isoformat()
    event = ExecutionEvent(
        event_id=str(uuid.uuid4()),
        agent_id=input.agent_id,
        session_id=input.session_id or str(uuid.uuid4()),
        action_type=input.action_type,
        action_payload=input.action_payload,
        environment=input.environment,
        parent_event_id=input.parent_event_id,
        trace_id=str(uuid.uuid4()),
        metadata=input.metadata,
        created_at=now,
        updated_at=now,
    )

    # Run full pipeline
    event = await run_pipeline(event)

    # Persist
    state.store_event(event.event_id, event.model_dump())
    return event


@app.get("/events/{event_id}", response_model=ExecutionEvent)
async def get_event(event_id: str):
    """Retrieve a completed execution event by ID."""
    data = state.get_event(event_id)
    if not data:
        raise HTTPException(status_code=404, detail="Event not found")
    return ExecutionEvent(**data)


@app.post("/events/{event_id}/replay", response_model=ExecutionEvent)
async def replay_event(event_id: str):
    """
    Replay an existing event through the pipeline.
    Creates a new event linked to the original via parent_event_id.
    """
    original_data = state.get_event(event_id)
    if not original_data:
        raise HTTPException(status_code=404, detail="Event not found")

    original = ExecutionEvent(**original_data)
    now = datetime.now(timezone.utc).isoformat()
    replayed = ExecutionEvent(
        event_id=str(uuid.uuid4()),
        agent_id=original.agent_id,
        session_id=original.session_id,
        action_type=original.action_type,
        action_payload=original.action_payload,
        environment=original.environment,
        parent_event_id=original.event_id,
        trace_id=str(uuid.uuid4()),
        metadata={**original.metadata, "replay_of": original.event_id},
        created_at=now,
        updated_at=now,
    )

    replayed = await run_pipeline(replayed)
    state.store_event(replayed.event_id, replayed.model_dump())
    return replayed


@app.get("/events/{event_id}/artifact", response_model=VYREEnvelope)
async def get_artifact(event_id: str):
    """Retrieve the VYRE artifact envelope for an event."""
    data = state.get_artifact(event_id)
    if not data:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return VYREEnvelope(**data)


@app.get("/events", response_model=list[ExecutionEvent])
async def list_events(
    agent_id: Optional[str] = None,
    action_type: Optional[str] = None,
    event_state: Optional[EventState] = None,
    limit: int = 50,
):
    """List events with optional filters."""
    events_data = state.list_events(
        agent_id=agent_id,
        action_type=action_type,
        state=event_state.value if event_state else None,
        limit=limit,
    )
    return [ExecutionEvent(**e) for e in events_data]


@app.get("/evm/status")
async def evm_status():
    """Check EVM adapter configuration and registry status."""
    from evm_adapter import (BASE_RPC_URL, BASE_CHAIN_ID,
                              ERC8004_IDENTITY_REGISTRY,
                              ERC8004_REPUTATION_REGISTRY,
                              ERC8004_VALIDATION_REGISTRY)
    return {
        "base": {
            "rpc_url": BASE_RPC_URL[:30] + "..." if BASE_RPC_URL else "not_configured",
            "chain_id": BASE_CHAIN_ID,
        },
        "erc8004": {
            "identity_registry": ERC8004_IDENTITY_REGISTRY or "not_configured",
            "reputation_registry": ERC8004_REPUTATION_REGISTRY or "not_configured",
            "validation_registry": ERC8004_VALIDATION_REGISTRY or "not_configured",
        },
        "cached_profiles": len(evm_adapter._profile_cache),
    }


# ---------------------------------------------------------------------------
# learn() — Fourth kernel primitive
# ---------------------------------------------------------------------------

@app.post("/learn")
async def submit_outcome(report: OutcomeReport):
    """
    Submit an execution outcome to learn().
    Evaluates the outcome, updates reputation, and returns feedback.

    This completes the OROS loop:
      observe → adjudicate → settle → learn
    """
    # Retrieve the original event
    event_data = state.get_event(report.event_id)
    if not event_data:
        raise HTTPException(status_code=404, detail="Event not found")

    # Run learn() evaluation
    feedback = learn_engine.evaluate(event_data, report)

    # Apply reputation delta
    updated_reputation = reputation_ledger.apply_delta(feedback.reputation_delta)

    # Persist feedback for audit trail
    reputation_ledger.store_feedback(feedback)

    return {
        "feedback": feedback.model_dump(),
        "updated_reputation": updated_reputation,
    }


@app.get("/reputation/{actor_id}")
async def get_reputation(actor_id: str):
    """Get current reputation score for an agent."""
    return reputation_ledger.get_reputation(actor_id)


@app.get("/reputation")
async def list_reputations():
    """List all tracked agent reputations."""
    r = state._get_redis()
    reputations = []

    if r:
        try:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match="reputation:*", count=100)
                for key in keys:
                    data = r.get(key)
                    if data:
                        reputations.append(json.loads(data))
                if cursor == 0:
                    break
        except Exception:
            pass
    else:
        if hasattr(state, '_mem_reputation'):
            reputations = list(state._mem_reputation.values())

    reputations.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"agents": reputations, "count": len(reputations)}


# ---------------------------------------------------------------------------
# Artifact verification — proof consumption layer
# ---------------------------------------------------------------------------
# Turns internal confidence into external trust.
# Anyone with the public key can verify that a governance decision
# came from OROS and wasn't tampered with.
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    """Input for artifact verification."""
    event_hash: str
    signature: str
    public_key: Optional[str] = None  # optional: use server's key if not provided


@app.get("/vyre/key")
async def vyre_public_key():
    """
    Returns the VYRE public signing key.
    Use this key to independently verify any OROS artifact signature.
    """
    if not VYRE_SIGNING_KEY:
        raise HTTPException(status_code=503, detail="VYRE signing not configured")
    try:
        from nacl.signing import SigningKey
        seed = bytes.fromhex(VYRE_SIGNING_KEY)
        sk = SigningKey(seed)
        pub = sk.verify_key.encode().hex()
        return {
            "public_key": pub,
            "algorithm": "Ed25519",
            "signer": SERVICE_NAME,
            "usage": "Verify OROS artifact signatures with this key",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key derivation failed: {e}")


@app.post("/vyre/verify")
async def verify_artifact(req: VerifyRequest):
    """
    Verify an artifact signature.

    Accepts an event_hash and signature, returns verification result.
    Uses the server's public key by default, or a provided one.
    """
    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError

        # Get public key
        if req.public_key:
            pub_bytes = bytes.fromhex(req.public_key)
        elif VYRE_SIGNING_KEY:
            from nacl.signing import SigningKey
            seed = bytes.fromhex(VYRE_SIGNING_KEY)
            sk = SigningKey(seed)
            pub_bytes = sk.verify_key.encode()
        else:
            return {
                "verified": False,
                "reason": "no_public_key_available",
            }

        vk = VerifyKey(pub_bytes)
        sig_bytes = bytes.fromhex(req.signature)
        msg_bytes = req.event_hash.encode()

        # Verify
        vk.verify(msg_bytes, sig_bytes)

        return {
            "verified": True,
            "event_hash": req.event_hash,
            "public_key": pub_bytes.hex(),
            "algorithm": "Ed25519",
            "signer": SERVICE_NAME,
        }

    except BadSignatureError:
        return {
            "verified": False,
            "reason": "signature_invalid",
            "event_hash": req.event_hash,
        }
    except Exception as e:
        return {
            "verified": False,
            "reason": f"verification_error:{e}",
        }


@app.get("/vyre/artifacts/{event_id}")
async def get_verified_artifact(event_id: str):
    """
    Retrieve an artifact with inline verification status.
    Returns the full artifact + whether the signature is currently valid.
    """
    artifact_data = state.get_artifact(event_id)
    if not artifact_data:
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = VYREEnvelope(**artifact_data)

    # Auto-verify if we have the signing key
    verified = False
    if artifact.signatures and VYRE_SIGNING_KEY:
        try:
            from nacl.signing import SigningKey, VerifyKey
            from nacl.exceptions import BadSignatureError

            seed = bytes.fromhex(VYRE_SIGNING_KEY)
            sk = SigningKey(seed)
            vk = sk.verify_key

            sig_hex = artifact.signatures[0].get("signature", "")
            if sig_hex and not sig_hex.startswith("unsigned"):
                sig_bytes = bytes.fromhex(sig_hex)
                msg_bytes = artifact.event_hash.encode()
                vk.verify(msg_bytes, sig_bytes)
                verified = True
        except (BadSignatureError, Exception):
            verified = False

    return {
        "artifact": artifact.model_dump(),
        "verification": {
            "verified": verified,
            "algorithm": "Ed25519",
            "signer": SERVICE_NAME,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
