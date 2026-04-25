"""
HELIX — Governed Swap Execution Layer v0.1.0
=====================================
A policy-aware execution rail for swaps.

Intent → Governance (OROS + PRAETOR) → Route → Execute → Receipt

Endpoints:
    POST /swap/quote       — Get route preview without governance
    POST /swap/evaluate    — Governance check: should this swap happen?
    POST /swap/execute     — Full flow: evaluate → route → execute → receipt
    GET  /swap/receipt/:id — Retrieve execution receipt
    GET  /health           — Service status

Environment:
    PORT                   — Server port (default: 8080)
    OROS_URL               — OROS endpoint (default: execution-coordinator on Railway)
    SURVIVOR_GATE_URL      — Gate service for receipt creation
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("swap-rail")

# ── Config ───────────────────────────────────────────────────────────────────

OROS_URL = os.getenv("OROS_URL", "https://execution-coordinator-production.up.railway.app")
SURVIVOR_GATE_URL = os.getenv("SURVIVOR_GATE_URL", "https://survivor-oracle-production-1501.up.railway.app")
SERVICE_VERSION = "0.1.0"

# ── Asset Resolution ────────────────────────────────────────────────────────
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
MINT_MAP = {
    "SOL": SOL_MINT, "WSOL": SOL_MINT,
    "USDC": USDC_MINT, "USDT": USDT_MINT,
}

def resolve_mint(chain: str, asset: str) -> str:
    if chain.lower() == "solana":
        return MINT_MAP.get(asset.upper(), asset)
    return asset


# ============================================================================
# Models
# ============================================================================

class SwapDecision(str, Enum):
    ALLOW = "ALLOW"
    THROTTLE = "THROTTLE"
    READ_ONLY = "READ_ONLY"
    DENY = "DENY"


class SwapStatus(str, Enum):
    PREPARED = "PREPARED"
    AUTHORIZED = "AUTHORIZED"
    DRY_RUN = "DRY_RUN"
    DENIED = "DENIED"
    READ_ONLY = "READ_ONLY"
    SIGNED = "SIGNED"
    BROADCAST = "BROADCAST"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"


class Chain(str, Enum):
    SOLANA = "solana"
    BASE = "base"
    ETHEREUM = "ethereum"


class Venue(str, Enum):
    JUPITER = "jupiter"
    RAYDIUM = "raydium"
    UNISWAP = "uniswap"
    UNKNOWN = "unknown"


class SwapConstraints(BaseModel):
    max_notional_usd: Optional[float] = None
    require_receipt: bool = True
    allowed_venues: List[str] = Field(default_factory=list)
    destination_whitelist_only: bool = False


class SwapIntent(BaseModel):
    actor_id: str
    chain: Chain
    from_asset: str
    to_asset: str
    amount: float = Field(..., gt=0)
    slippage_bps: int = Field(default=50, ge=1, le=10_000)
    destination: str
    constraints: SwapConstraints = Field(default_factory=SwapConstraints)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuoteRequest(BaseModel):
    intent: SwapIntent


class ExecuteRequest(BaseModel):
    intent: SwapIntent
    approved_swap_id: Optional[str] = None
    dry_run: bool = False


class RoutePreview(BaseModel):
    venue: Venue
    expected_out: Optional[str] = None
    price_impact_bps: Optional[float] = None
    estimated_fee_usd: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class QuoteResponse(BaseModel):
    quote_id: str
    chain: Chain
    route_preview: RoutePreview
    quote_valid_for_s: int = 30


class EvaluateResponse(BaseModel):
    swap_id: str
    decision: SwapDecision
    size_multiplier: float = 1.0
    posture: str
    posture_reason: str
    reason_codes: List[str] = Field(default_factory=list)
    route_preview: Optional[RoutePreview] = None
    governance_trace_id: Optional[str] = None
    receipt_id: Optional[str] = None
    raw_governance: Dict[str, Any] = Field(default_factory=dict)


class ExecuteResponse(BaseModel):
    swap_id: str
    decision: SwapDecision
    executed: bool
    venue: Venue = Venue.UNKNOWN
    tx_hash: Optional[str] = None
    receipt_id: Optional[str] = None
    verify_url: Optional[str] = None
    status: str = "pending"
    posture: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    raw_execution: Dict[str, Any] = Field(default_factory=dict)


class ReceiptResponse(BaseModel):
    swap_id: str
    decision: SwapDecision
    posture: Optional[str] = None
    venue: Venue = Venue.UNKNOWN
    tx_hash: Optional[str] = None
    receipt_id: Optional[str] = None
    verify_url: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    governance_trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Adapter layer
# ============================================================================

class SwapAdapter:
    async def quote(self, intent: SwapIntent) -> RoutePreview:
        raise NotImplementedError

    async def execute(self, intent: SwapIntent, route: RoutePreview, dry_run: bool = False) -> Dict[str, Any]:
        raise NotImplementedError


class JupiterAdapter(SwapAdapter):
    """Jupiter v6 adapter — real quotes, guarded execution."""

    QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"  # free public tier, no key needed
    DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1/solana"

    # Known Solana token mints
    MINT_MAP = {
        "SOL": "So11111111111111111111111111111111111111112",
        "WSOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    }

    # Token decimals for amount conversion
    DECIMAL_MAP = {
        "SOL": 9,
        "WSOL": 9,
        "USDC": 6,
        "USDT": 6,
    }

    def _resolve_mint(self, symbol: str) -> str:
        upper = symbol.upper()
        if upper in self.MINT_MAP:
            return self.MINT_MAP[upper]
        # If it looks like a mint address already, use it directly
        if len(symbol) > 20:
            return symbol
        return symbol

    def _get_decimals(self, symbol: str) -> int:
        return self.DECIMAL_MAP.get(symbol.upper(), 9)

    async def quote(self, intent: SwapIntent) -> RoutePreview:
        input_mint = self._resolve_mint(intent.from_asset)
        output_mint = self._resolve_mint(intent.to_asset)
        decimals = self._get_decimals(intent.from_asset)
        amount_raw = int(intent.amount * (10 ** decimals))

        # Try Jupiter lite-api (free public tier, no API key required)
        try:
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount_raw),
                "slippageBps": str(intent.slippage_bps),
            }
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.QUOTE_URL, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        out_amount_raw = int(data.get("outAmount", 0))
                        out_decimals = self._get_decimals(intent.to_asset)
                        out_amount = out_amount_raw / (10 ** out_decimals)
                        # priceImpactPct from Jupiter is already a fraction (e.g. "0.0023" = 0.23%)
                        # bps = fraction * 10000
                        price_impact_pct = float(data.get("priceImpactPct", 0))
                        price_impact_bps = round(price_impact_pct * 10000, 2)

                        return RoutePreview(
                            venue=Venue.JUPITER,
                            expected_out=f"{out_amount:.6f} {intent.to_asset}",
                            price_impact_bps=price_impact_bps,
                            estimated_fee_usd=None,
                            raw={
                                "adapter": "jupiter_lite",
                                "mock": False,
                                "indicative_only": False,
                                "input_mint": input_mint,
                                "output_mint": output_mint,
                                "in_amount": str(amount_raw),
                                "out_amount": str(out_amount_raw),
                                "price_impact_pct": data.get("priceImpactPct"),
                                "route_plan_count": len(data.get("routePlan", [])),
                                "route_plan": data.get("routePlan", []),
                                "swap_usd_value": data.get("swapUsdValue"),
                                "context_slot": data.get("contextSlot"),
                                "quote_response": data,
                            },
                        )
                    else:
                        body = await resp.text()
                        logger.warning(f"[jupiter_lite] quote HTTP {resp.status}: {body[:200]}")
        except Exception as e:
            logger.warning(f"[jupiter_lite] quote failed: {e}")

        # Fallback: DexScreener indicative pricing
        try:
            to_mint = self._resolve_mint(intent.to_asset)
            url = f"{self.DEXSCREENER_URL}/{to_mint}"
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        pairs = await resp.json()
                        if pairs and len(pairs) > 0:
                            pair = pairs[0]
                            price_usd = float(pair.get("priceUsd", 0))
                            liquidity_usd = pair.get("liquidity", {}).get("usd", 0)
                            if price_usd > 0:
                                expected_out = intent.amount / price_usd
                                return RoutePreview(
                                    venue=Venue.UNKNOWN,
                                    expected_out=f"{expected_out:.6f} {intent.to_asset} (indicative)",
                                    price_impact_bps=None,
                                    estimated_fee_usd=None,
                                    raw={
                                        "adapter": "dexscreener",
                                        "mock": False,
                                        "indicative_only": True,
                                        "reason": "jupiter_unreachable_or_no_route",
                                        "price_usd": price_usd,
                                        "liquidity_usd": liquidity_usd,
                                        "pair": pair.get("pairAddress"),
                                    },
                                )
        except Exception as e:
            logger.warning(f"[dexscreener] fallback failed: {e}")

        # Final fallback
        return RoutePreview(
            venue=Venue.JUPITER,
            expected_out=f"unavailable",
            price_impact_bps=None,
            estimated_fee_usd=None,
            raw={"adapter": "none", "mock": True, "reason": "all_quote_sources_failed"},
        )

    async def execute(self, intent: SwapIntent, route: RoutePreview,
                      dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            return {
                "executed": False,
                "status": "dry_run",
                "venue": route.venue.value,
                "tx_hash": None,
            }
        # Phase 3: real execution goes here
        # For now, return mock tx to prevent accidental spend
        return {
            "executed": False,
            "status": "execution_not_wired",
            "venue": route.venue.value,
            "tx_hash": None,
            "note": "Jupiter swap build/broadcast not yet implemented",
        }


# ============================================================================
# Governance bridge — REAL OROS + PRAETOR
# ============================================================================

def build_oros_event(intent: SwapIntent) -> Dict[str, Any]:
    """Build an OROS-compatible event from a swap intent."""
    return {
        "agent_id": intent.actor_id,
        "action_type": "swap",
        "action_payload": {
            "from_asset": intent.from_asset,
            "to_asset": intent.to_asset,
            "amount": intent.amount,
            "mint": intent.to_asset,
            "destination": intent.destination,
            "chain": intent.chain.value,
        },
        "environment": {
            "chain": intent.chain.value,
            "network": "mainnet",
        },
        "metadata": {
            "phase": "pre_commit",
            "swap_rail": True,
            "slippage_bps": intent.slippage_bps,
            "market_cap_usd": intent.metadata.get("market_cap_usd", 0),
            "source": "swap_rail",
            **{k: v for k, v in intent.metadata.items() if k != "market_cap_usd"},
        },
    }


async def call_oros(intent: SwapIntent) -> Dict[str, Any]:
    """Call the live OROS /events endpoint for governance evaluation."""
    event = build_oros_event(intent)
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{OROS_URL}/events", json=event) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                else:
                    body = await resp.text()
                    logger.warning(f"[oros] HTTP {resp.status}: {body[:200]}")
                    return {"error": f"oros_http_{resp.status}", "body": body[:200]}
    except Exception as e:
        logger.error(f"[oros] call failed: {e}")
        return {"error": f"oros_unreachable:{str(e)[:80]}"}


def map_oros_to_decision(oros_response: Dict[str, Any]) -> tuple:
    """Extract governance verdict from OROS response."""
    if "error" in oros_response:
        return SwapDecision.DENY, 0.0, "ERROR", str(oros_response["error"]), ["OROS_UNREACHABLE"]

    gov = oros_response.get("governance", {})
    decision_str = gov.get("decision", "DENY")
    posture = oros_response.get("metadata", {}).get("system_posture", "UNKNOWN")
    posture_reason = oros_response.get("metadata", {}).get("posture_reason", "unknown")

    # Map OROS decision to SwapDecision
    decision_map = {
        "ALLOW": SwapDecision.ALLOW,
        "ALLOW_WITH_GUARDRAILS": SwapDecision.ALLOW,
        "CHALLENGE": SwapDecision.THROTTLE,
        "THROTTLE": SwapDecision.THROTTLE,
        "DENY": SwapDecision.DENY,
        "DEFER": SwapDecision.READ_ONLY,
    }
    decision = decision_map.get(decision_str, SwapDecision.DENY)

    # Size multiplier from posture
    size_map = {"NORMAL": 1.0, "CAUTIOUS": 0.5, "DEFENSIVE": 0.25, "HALT": 0.0}
    size_multiplier = size_map.get(posture, 1.0)

    # If HALT posture, override to DENY
    if posture == "HALT":
        decision = SwapDecision.DENY

    # Reason codes
    reason_codes = []
    if gov.get("risk_level"):
        reason_codes.append(f"RISK_{gov['risk_level']}")
    if gov.get("archetype"):
        reason_codes.append(f"ARCH_{gov['archetype']}")
    if posture != "NORMAL":
        reason_codes.append(f"PRAETOR_{posture}")
    if gov.get("intent_match") and gov["intent_match"] != "match":
        reason_codes.append(f"INTENT_{gov['intent_match']}")

    return decision, size_multiplier, posture, posture_reason, reason_codes


async def evaluate_swap(intent: SwapIntent, route_preview: RoutePreview) -> EvaluateResponse:
    """Full governance evaluation through OROS + PRAETOR."""
    swap_id = f"swap_{uuid.uuid4().hex[:12]}"

    # Basic validation first
    if intent.from_asset.upper() == intent.to_asset.upper():
        return EvaluateResponse(
            swap_id=swap_id,
            decision=SwapDecision.DENY,
            size_multiplier=0.0,
            posture="NORMAL",
            posture_reason="invalid_pair",
            reason_codes=["INVALID_PAIR"],
            route_preview=route_preview,
        )

    # Call OROS for real governance
    oros_response = await call_oros(intent)
    decision, size_multiplier, posture, posture_reason, reason_codes = map_oros_to_decision(oros_response)

    # Extract trace info
    governance_trace_id = oros_response.get("event_id") or oros_response.get("trace_id")

    # Call SURVIVOR Gate for receipt (if decision allows)
    receipt_id = None
    if decision in (SwapDecision.ALLOW, SwapDecision.THROTTLE):
        try:
            gate_response = await call_survivor_gate(intent)
            receipt_id = gate_response.get("receipt", {}).get("receipt_id")
        except Exception as e:
            logger.warning(f"[gate] receipt creation failed: {e}")

    return EvaluateResponse(
        swap_id=swap_id,
        decision=decision,
        size_multiplier=size_multiplier,
        posture=posture,
        posture_reason=posture_reason,
        reason_codes=reason_codes,
        route_preview=route_preview,
        governance_trace_id=governance_trace_id,
        receipt_id=receipt_id,
        raw_governance={
            "oros_decision": oros_response.get("governance", {}),
            "system_posture": posture,
        },
    )


async def call_survivor_gate(intent: SwapIntent) -> Dict[str, Any]:
    """Call SURVIVOR Gate to create an execution receipt."""
    gate_payload = {
        "chain": intent.chain.value,
        "from_asset": resolve_mint(intent.chain.value, intent.from_asset),
        "to_asset": resolve_mint(intent.chain.value, intent.to_asset),
        "notional_usd": intent.amount,
        "slippage_bps": intent.slippage_bps,
        "kind": "swap",
    }
    headers = {
        "Content-Type": "application/json",
        "x-agent-id": intent.actor_id,
        "x-caller-ref": f"swap_rail_{int(time.time())}",
    }
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{SURVIVOR_GATE_URL}/gate", json=gate_payload, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    body = await resp.text()
                    logger.warning(f"[gate] HTTP {resp.status}: {body[:200]}")
                    return {}
    except Exception as e:
        logger.warning(f"[gate] call failed: {e}")
        return {}


# ============================================================================
# Receipt builder
# ============================================================================

def build_receipt(evaluation: EvaluateResponse,
                  execution: Optional[ExecuteResponse] = None) -> ReceiptResponse:
    return ReceiptResponse(
        swap_id=evaluation.swap_id,
        decision=evaluation.decision,
        posture=evaluation.posture,
        venue=execution.venue if execution else Venue.UNKNOWN,
        tx_hash=execution.tx_hash if execution else None,
        receipt_id=evaluation.receipt_id or (execution.receipt_id if execution else None),
        verify_url=f"{SURVIVOR_GATE_URL}/receipts/{evaluation.receipt_id}/verify" if evaluation.receipt_id else None,
        reason_codes=evaluation.reason_codes,
        governance_trace_id=evaluation.governance_trace_id,
        metadata={
            "oros_url": OROS_URL,
            "gate_url": SURVIVOR_GATE_URL,
            "evaluated_at": time.time(),
        },
    )


# ============================================================================
# App
# ============================================================================

app = FastAPI(
    title="HELIX — Governed Swap Execution Layer",
    version=SERVICE_VERSION,
    description="Policy-aware execution rail: intent → governance → route → execute → receipt",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

adapter = JupiterAdapter()

RECEIPT_STORE: Dict[str, Dict[str, Any]] = {}
SWAP_STORE: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
async def health():
    return {
        "service": "HELIX",
        "version": SERVICE_VERSION,
        "status": "ok",
        "oros": OROS_URL,
        "gate": SURVIVOR_GATE_URL,
        "adapter": "jupiter_lite",
    }


@app.post("/swap/quote", response_model=QuoteResponse)
async def swap_quote(body: QuoteRequest):
    """Get a route preview without governance evaluation."""
    route = await adapter.quote(body.intent)
    return QuoteResponse(
        quote_id=f"quote_{uuid.uuid4().hex[:12]}",
        chain=body.intent.chain,
        route_preview=route,
        quote_valid_for_s=30,
    )


@app.post("/swap/evaluate", response_model=EvaluateResponse)
async def swap_evaluate(body: QuoteRequest):
    """Governance check: should this swap happen?"""
    route = await adapter.quote(body.intent)
    evaluation = await evaluate_swap(body.intent, route)

    logger.info(
        f"[evaluate] {body.intent.actor_id} "
        f"{body.intent.from_asset}→{body.intent.to_asset} "
        f"${body.intent.amount} → {evaluation.decision.value} "
        f"posture={evaluation.posture} codes={evaluation.reason_codes}"
    )
    return evaluation


@app.post("/swap/execute", response_model=ExecuteResponse)
async def swap_execute(body: ExecuteRequest):
    """Full flow: evaluate → route → execute → receipt."""
    route = await adapter.quote(body.intent)
    evaluation = await evaluate_swap(body.intent, route)

    if evaluation.decision == SwapDecision.DENY:
        execution = ExecuteResponse(
            swap_id=evaluation.swap_id,
            decision=evaluation.decision,
            executed=False,
            venue=route.venue,
            status="denied",
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            receipt_id=evaluation.receipt_id,
            verify_url=f"{SURVIVOR_GATE_URL}/receipts/{evaluation.receipt_id}/verify" if evaluation.receipt_id else None,
        )
    elif evaluation.decision == SwapDecision.READ_ONLY:
        execution = ExecuteResponse(
            swap_id=evaluation.swap_id,
            decision=evaluation.decision,
            executed=False,
            venue=route.venue,
            status="read_only",
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            receipt_id=evaluation.receipt_id,
        )
    else:
        # ALLOW or THROTTLE — execute (with size adjustment for THROTTLE)
        exec_result = await adapter.execute(body.intent, route, dry_run=body.dry_run)
        execution = ExecuteResponse(
            swap_id=evaluation.swap_id,
            decision=evaluation.decision,
            executed=exec_result["executed"],
            venue=route.venue,
            tx_hash=exec_result.get("tx_hash"),
            status=exec_result.get("status", "submitted"),
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            receipt_id=evaluation.receipt_id,
            verify_url=f"{SURVIVOR_GATE_URL}/receipts/{evaluation.receipt_id}/verify" if evaluation.receipt_id else None,
            raw_execution=exec_result,
        )

    # Build and store receipt
    receipt = build_receipt(evaluation, execution)
    RECEIPT_STORE[evaluation.swap_id] = receipt.model_dump()

    # Track swap lifecycle
    exec_status = "DENIED"
    if execution.status == "dry_run":
        exec_status = "DRY_RUN"
    elif execution.status == "denied":
        exec_status = "DENIED"
    elif execution.status == "read_only":
        exec_status = "READ_ONLY"
    elif execution.executed:
        exec_status = "BROADCAST"
    else:
        exec_status = "AUTHORIZED"

    SWAP_STORE[evaluation.swap_id] = {
        "swap_id": evaluation.swap_id,
        "intent": body.intent.model_dump(),
        "decision": evaluation.decision.value,
        "posture": evaluation.posture,
        "posture_reason": evaluation.posture_reason,
        "size_multiplier": evaluation.size_multiplier,
        "reason_codes": evaluation.reason_codes,
        "receipt_id": evaluation.receipt_id,
        "governance_trace_id": evaluation.governance_trace_id,
        "execution_status": exec_status,
        "tx_hash": execution.tx_hash,
        "verify_url": execution.verify_url,
        "created_at": time.time(),
        "timeline": [
            {"status": "PREPARED", "ts": time.time()},
            {"status": exec_status, "ts": time.time()},
        ],
    }

    logger.info(
        f"[execute] {body.intent.actor_id} "
        f"{body.intent.from_asset}→{body.intent.to_asset} "
        f"${body.intent.amount} → {execution.decision.value} "
        f"executed={execution.executed} tx={execution.tx_hash}"
    )
    return execution


@app.get("/swap/receipt/{swap_id}", response_model=ReceiptResponse)
async def swap_receipt(swap_id: str):
    """Retrieve execution receipt."""
    receipt = RECEIPT_STORE.get(swap_id)
    if not receipt:
        raise HTTPException(status_code=404, detail="receipt_not_found")
    return receipt




@app.get("/swap/status/{swap_id}")
async def swap_status(swap_id: str):
    """Get full swap lifecycle status."""
    record = SWAP_STORE.get(swap_id)
    if not record:
        raise HTTPException(status_code=404, detail="swap_not_found")
    return record


@app.get("/swap/history")
async def swap_history():
    """List recent swaps with lifecycle status."""
    swaps = sorted(SWAP_STORE.values(), key=lambda s: s.get("created_at", 0), reverse=True)[:50]
    return {
        "count": len(swaps),
        "swaps": [{
            "swap_id": s["swap_id"],
            "decision": s["decision"],
            "posture": s["posture"],
            "execution_status": s["execution_status"],
            "receipt_id": s.get("receipt_id"),
            "tx_hash": s.get("tx_hash"),
            "created_at": s.get("created_at"),
        } for s in swaps],
    }

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
