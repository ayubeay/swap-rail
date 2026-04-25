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
import base58
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


def validate_solana_pubkey(s: str) -> bool:
    """Return True iff s is a valid base58-encoded 32-byte Solana pubkey."""
    if not s or not isinstance(s, str):
        return False
    try:
        decoded = base58.b58decode(s)
        return len(decoded) == 32
    except Exception:
        return False


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


class BuildTxRequest(BaseModel):
    intent: SwapIntent
    user_public_key: str
    allow_different_destination: bool = False
    wrap_and_unwrap_sol: bool = True
    use_shared_accounts: bool = True
    compute_unit_price_micro_lamports: Optional[int] = None
    prioritization_fee_lamports: Optional[int] = None


class BuildTxResponse(BaseModel):
    swap_id: str
    decision: SwapDecision
    status: str
    executed: bool = False
    venue: Venue = Venue.UNKNOWN
    user_public_key: str
    destination: str
    swap_transaction: Optional[str] = None
    last_valid_block_height: Optional[int] = None
    prioritization_fee_lamports: Optional[int] = None
    compute_unit_limit: Optional[int] = None
    simulation_error: Optional[str] = None
    expected_out: Optional[str] = None
    price_impact_bps: Optional[float] = None
    route_plan_count: Optional[int] = None
    receipt_id: Optional[str] = None
    verify_url: Optional[str] = None
    posture: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    governance_trace_id: Optional[str] = None
    quote_used: Optional[Dict[str, Any]] = None


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

    SWAP_BUILD_URL = "https://lite-api.jup.ag/swap/v1/swap"

    async def build_swap_tx(
        self,
        quote_response: Dict[str, Any],
        user_public_key: str,
        wrap_and_unwrap_sol: bool = True,
        use_shared_accounts: bool = True,
        compute_unit_price_micro_lamports: Optional[int] = None,
        prioritization_fee_lamports: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call Jupiter /swap to build an unsigned VersionedTransaction."""
        payload: Dict[str, Any] = {
            "quoteResponse": quote_response,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": wrap_and_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
            "asLegacyTransaction": False,
            "useTokenLedger": False,
        }
        if compute_unit_price_micro_lamports is not None:
            payload["computeUnitPriceMicroLamports"] = compute_unit_price_micro_lamports
        if prioritization_fee_lamports is not None:
            payload["prioritizationFeeLamports"] = prioritization_fee_lamports

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.SWAP_BUILD_URL, json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    logger.warning(f"[jupiter_lite] /swap HTTP {resp.status}: {body[:300]}")
                    return {
                        "error": f"jupiter_swap_http_{resp.status}",
                        "detail": body[:500],
                    }
                import json as _json
                try:
                    return _json.loads(body)
                except Exception as e:
                    logger.warning(f"[jupiter_lite] /swap response not JSON: {e}")
                    return {"error": "jupiter_swap_invalid_json", "detail": body[:500]}


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


@app.post("/swap/build-tx", response_model=BuildTxResponse)
async def swap_build_tx(body: BuildTxRequest):
    """
    Non-custodial swap transaction builder.

    Validates pubkeys, fetches a fresh Jupiter quote, runs OROS+SURVIVOR
    governance, and (if allowed) returns an unsigned base64 VersionedTransaction
    that the caller signs and broadcasts themselves.

    No private keys are ever held by HELIX in this path.
    """
    intent = body.intent

    # 1. Validate user_public_key
    if not validate_solana_pubkey(body.user_public_key):
        raise HTTPException(
            status_code=400,
            detail="invalid user_public_key: must be base58-encoded 32-byte Solana pubkey",
        )

    # 2. Default destination to user_public_key, or validate explicit destination
    destination = intent.destination if intent.destination else body.user_public_key
    if not validate_solana_pubkey(destination):
        raise HTTPException(
            status_code=400,
            detail="invalid destination: must be base58-encoded 32-byte Solana pubkey",
        )

    # 3. Enforce destination match unless explicitly allowed to differ
    if destination != body.user_public_key and not body.allow_different_destination:
        raise HTTPException(
            status_code=400,
            detail="destination differs from user_public_key; set allow_different_destination=true to permit",
        )

    # 4. Fresh quote via existing JupiterAdapter
    intent.destination = destination  # ensure intent reflects resolved destination
    route = await adapter.quote(intent)

    # 5. Run governance evaluation
    evaluation = await evaluate_swap(intent, route)
    swap_id = evaluation.swap_id
    verify_url = (
        f"{SURVIVOR_GATE_URL}/receipts/{evaluation.receipt_id}/verify"
        if evaluation.receipt_id
        else None
    )

    # 6. Handle non-build decisions
    if evaluation.decision == SwapDecision.DENY:
        SWAP_STORE[swap_id] = {
            "swap_id": swap_id,
            "intent": intent.model_dump(),
            "user_public_key": body.user_public_key,
            "destination": destination,
            "decision": evaluation.decision.value,
            "status": "GOVERNANCE_DENIED",
            "posture": evaluation.posture,
            "reason_codes": evaluation.reason_codes,
            "receipt_id": evaluation.receipt_id,
            "governance_trace_id": evaluation.governance_trace_id,
            "created_at": time.time(),
        }
        return BuildTxResponse(
            swap_id=swap_id,
            decision=evaluation.decision,
            status="denied",
            executed=False,
            venue=route.venue,
            user_public_key=body.user_public_key,
            destination=destination,
            swap_transaction=None,
            expected_out=route.expected_out,
            price_impact_bps=route.price_impact_bps,
            route_plan_count=route.raw.get("route_plan_count") if isinstance(route.raw, dict) else None,
            receipt_id=evaluation.receipt_id,
            verify_url=verify_url,
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            governance_trace_id=evaluation.governance_trace_id,
            quote_used=None,
        )

    if evaluation.decision == SwapDecision.READ_ONLY:
        SWAP_STORE[swap_id] = {
            "swap_id": swap_id,
            "intent": intent.model_dump(),
            "user_public_key": body.user_public_key,
            "destination": destination,
            "decision": evaluation.decision.value,
            "status": "GOVERNANCE_READ_ONLY",
            "posture": evaluation.posture,
            "reason_codes": evaluation.reason_codes,
            "receipt_id": evaluation.receipt_id,
            "governance_trace_id": evaluation.governance_trace_id,
            "created_at": time.time(),
        }
        return BuildTxResponse(
            swap_id=swap_id,
            decision=evaluation.decision,
            status="read_only",
            executed=False,
            venue=route.venue,
            user_public_key=body.user_public_key,
            destination=destination,
            swap_transaction=None,
            expected_out=route.expected_out,
            price_impact_bps=route.price_impact_bps,
            route_plan_count=route.raw.get("route_plan_count") if isinstance(route.raw, dict) else None,
            receipt_id=evaluation.receipt_id,
            verify_url=verify_url,
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            governance_trace_id=evaluation.governance_trace_id,
            quote_used=None,
        )

    # 7. THROTTLE: re-quote at adjusted size
    if evaluation.decision == SwapDecision.THROTTLE and evaluation.size_multiplier < 1.0:
        original_amount = intent.amount
        intent.amount = original_amount * evaluation.size_multiplier
        logger.info(
            f"[build-tx] THROTTLE: amount {original_amount} -> {intent.amount} "
            f"(size_multiplier={evaluation.size_multiplier})"
        )
        route = await adapter.quote(intent)

    # 8. Extract the cached quote_response embedded in route.raw by JupiterAdapter
    quote_response = None
    if isinstance(route.raw, dict):
        quote_response = route.raw.get("quote_response")
    if not quote_response:
        SWAP_STORE[swap_id] = {
            "swap_id": swap_id,
            "intent": intent.model_dump(),
            "user_public_key": body.user_public_key,
            "destination": destination,
            "decision": evaluation.decision.value,
            "status": "TX_BUILD_FAILED",
            "tx_build_failure_reason": "no_quote_response_available",
            "created_at": time.time(),
        }
        return BuildTxResponse(
            swap_id=swap_id,
            decision=evaluation.decision,
            status="tx_build_failed",
            executed=False,
            venue=route.venue,
            user_public_key=body.user_public_key,
            destination=destination,
            swap_transaction=None,
            simulation_error="no_quote_response_available_from_adapter",
            expected_out=route.expected_out,
            price_impact_bps=route.price_impact_bps,
            receipt_id=evaluation.receipt_id,
            verify_url=verify_url,
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            governance_trace_id=evaluation.governance_trace_id,
            quote_used=None,
        )

    # 9. Build the unsigned tx via Jupiter /swap
    jup_response = await adapter.build_swap_tx(
        quote_response=quote_response,
        user_public_key=body.user_public_key,
        wrap_and_unwrap_sol=body.wrap_and_unwrap_sol,
        use_shared_accounts=body.use_shared_accounts,
        compute_unit_price_micro_lamports=body.compute_unit_price_micro_lamports,
        prioritization_fee_lamports=body.prioritization_fee_lamports,
    )

    # 10. Handle Jupiter errors
    if "error" in jup_response:
        SWAP_STORE[swap_id] = {
            "swap_id": swap_id,
            "intent": intent.model_dump(),
            "user_public_key": body.user_public_key,
            "destination": destination,
            "decision": evaluation.decision.value,
            "status": "TX_BUILD_FAILED",
            "tx_build_failure_reason": jup_response.get("error"),
            "tx_build_failure_detail": jup_response.get("detail", "")[:500],
            "created_at": time.time(),
        }
        return BuildTxResponse(
            swap_id=swap_id,
            decision=evaluation.decision,
            status="tx_build_failed",
            executed=False,
            venue=route.venue,
            user_public_key=body.user_public_key,
            destination=destination,
            swap_transaction=None,
            simulation_error=f"{jup_response.get('error')}: {jup_response.get('detail', '')[:200]}",
            expected_out=route.expected_out,
            price_impact_bps=route.price_impact_bps,
            receipt_id=evaluation.receipt_id,
            verify_url=verify_url,
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            governance_trace_id=evaluation.governance_trace_id,
            quote_used=quote_response,
        )

    # 11. Auto-reject on simulationError (per design: don't return broken txs)
    sim_error = jup_response.get("simulationError")
    if sim_error:
        SWAP_STORE[swap_id] = {
            "swap_id": swap_id,
            "intent": intent.model_dump(),
            "user_public_key": body.user_public_key,
            "destination": destination,
            "decision": evaluation.decision.value,
            "status": "TX_BUILD_FAILED",
            "tx_build_failure_reason": "simulation_error",
            "simulation_error": sim_error,
            "created_at": time.time(),
        }
        return BuildTxResponse(
            swap_id=swap_id,
            decision=evaluation.decision,
            status="tx_build_failed",
            executed=False,
            venue=route.venue,
            user_public_key=body.user_public_key,
            destination=destination,
            swap_transaction=None,
            simulation_error=str(sim_error),
            expected_out=route.expected_out,
            price_impact_bps=route.price_impact_bps,
            receipt_id=evaluation.receipt_id,
            verify_url=verify_url,
            posture=evaluation.posture,
            reason_codes=evaluation.reason_codes,
            governance_trace_id=evaluation.governance_trace_id,
            quote_used=quote_response,
        )

    # 12. Success: store lifecycle + return unsigned tx
    swap_transaction_b64 = jup_response.get("swapTransaction")
    last_valid_block_height = jup_response.get("lastValidBlockHeight")
    prioritization_fee = jup_response.get("prioritizationFeeLamports")
    compute_unit_limit = jup_response.get("computeUnitLimit")

    SWAP_STORE[swap_id] = {
        "swap_id": swap_id,
        "intent": intent.model_dump(),
        "user_public_key": body.user_public_key,
        "destination": destination,
        "decision": evaluation.decision.value,
        "status": "UNSIGNED_TX_BUILT",
        "posture": evaluation.posture,
        "size_multiplier": evaluation.size_multiplier,
        "reason_codes": evaluation.reason_codes,
        "receipt_id": evaluation.receipt_id,
        "governance_trace_id": evaluation.governance_trace_id,
        "last_valid_block_height": last_valid_block_height,
        "created_at": time.time(),
        "timeline": [
            {"status": "PREPARED", "ts": time.time()},
            {"status": "QUOTE_FETCHED", "ts": time.time()},
            {"status": "UNSIGNED_TX_BUILT", "ts": time.time()},
        ],
    }

    logger.info(
        f"[build-tx] {body.user_public_key[:8]}... "
        f"{intent.from_asset}->{intent.to_asset} amount={intent.amount} "
        f"decision={evaluation.decision.value} venue={route.venue.value} "
        f"sim_ok lastValidBlockHeight={last_valid_block_height}"
    )

    return BuildTxResponse(
        swap_id=swap_id,
        decision=evaluation.decision,
        status="unsigned_tx_built",
        executed=False,
        venue=route.venue,
        user_public_key=body.user_public_key,
        destination=destination,
        swap_transaction=swap_transaction_b64,
        last_valid_block_height=last_valid_block_height,
        prioritization_fee_lamports=prioritization_fee,
        compute_unit_limit=compute_unit_limit,
        simulation_error=None,
        expected_out=route.expected_out,
        price_impact_bps=route.price_impact_bps,
        route_plan_count=route.raw.get("route_plan_count") if isinstance(route.raw, dict) else None,
        receipt_id=evaluation.receipt_id,
        verify_url=verify_url,
        posture=evaluation.posture,
        reason_codes=evaluation.reason_codes,
        governance_trace_id=evaluation.governance_trace_id,
        quote_used=quote_response,
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
