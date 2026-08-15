"""Polymarket CLOB order router - FAK (= IOC) orders only.

Safety contract:

  * Will NEVER place a real order unless ALL of the following are true:
      - ENABLE_REAL_ORDERS=1 env var is set
      - POLYMARKET_PRIVATE_KEY env var is set (or "Private Key" line in
        API_Keys file)
      - PolymarketOrderRouter.live=True passed at construction

  * In DRY-RUN mode (default) it logs a fully-prepared OrderArgs to the
    decisions JSONL and returns a synthetic fill receipt.

  * If the live order errors out for any reason (HTTP, signing, balance)
    we LOG and RETURN — never raise into the trading loop.

API credentials come from the existing `API_Keys` file (or env vars).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Credentials
# ──────────────────────────────────────────────────────────────────────────────
_HEADER_ALIASES = {
    # canonical lowercase key  ->  list of header strings (lowercased) that introduce
    # a value on a SUBSEQUENT line. Match is exact-equals after lower+strip.
    "private key":     ["polygon private key", "private key", "wallet private key"],
    "funder address":  ["polymarket wallet address", "funder address", "proxy address",
                        "polymarket proxy address", "wallet address"],
    "api key":         ["api key", "polymarket api key"],
    "api secret":      ["api secret", "polymarket api secret"],
    "api passphrase":  ["api passphrase", "polymarket api passphrase"],
    "signature type":  ["signature type", "polymarket signature type"],
}


def _load_keyfile(path: Path) -> dict[str, str]:
    """Parse the API_Keys file.

    Supports two formats interchangeably:

      1. `key: value`  (or `key = value`) on a single line.
      2. A `Header` line followed (after any blank/comment lines) by a
         standalone value line.  Recognized headers are in `_HEADER_ALIASES`.
    """
    if not path.exists(): return {}
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Build inverse map: header_alias_lower -> canonical_key
    inverse: dict[str, str] = {}
    for canonical, aliases in _HEADER_ALIASES.items():
        for a in aliases:
            inverse[a.strip().lower()] = canonical

    out: dict[str, str] = {}
    pending_header: str | None = None    # canonical key waiting for its value
    for raw in raw_lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Form 1: key: value
        if ":" in s and not s.lower() in inverse:
            k, v = s.split(":", 1); k = k.strip().lower(); v = v.strip()
            if k in inverse:                     # e.g. "API Key: foo"
                out[inverse[k]] = v
                pending_header = None
                continue
            if k and v:
                out[k] = v
                pending_header = None
                continue
        if "=" in s and not s.lower() in inverse:
            k, v = s.split("=", 1); k = k.strip().lower(); v = v.strip()
            if k in inverse:
                out[inverse[k]] = v
                pending_header = None
                continue
            if k and v:
                out[k] = v
                pending_header = None
                continue
        # Form 2: header line
        if s.lower() in inverse:
            pending_header = inverse[s.lower()]
            continue
        # Form 2: value line that fulfills a pending header
        if pending_header is not None:
            out[pending_header] = s
            pending_header = None
            continue
        # Unrecognized standalone line — ignore
    return out


@dataclass
class PolymarketCreds:
    api_key: str
    api_secret: str
    api_passphrase: str
    private_key: Optional[str] = None        # 0x-prefixed Polygon wallet key
    funder: Optional[str] = None             # proxy/deposit wallet address
    signature_type: int = 0                  # 0 = EOA direct, 1 = poly proxy, 2 = poly gnosis safe
    chain_id: int = 137                      # Polygon

    @classmethod
    def from_keyfile_and_env(cls, keyfile_path: Path = Path("API_Keys")) -> "PolymarketCreds":
        kv = _load_keyfile(keyfile_path)
        env = os.environ
        funder = env.get("POLYMARKET_FUNDER_ADDR") or kv.get("funder address") or None
        # Polymarket SignatureTypeV2: EOA=0, POLY_PROXY=1, POLY_1271=3 (Magic/Safe).
        # NOTE: there is NO value "2" in the v2 SDK — sending 2 causes Polymarket
        # to return order_version_mismatch.  Default is 3 (POLY_1271) when a
        # funder is present, since that's what newer web-onboarded accounts use.
        # For old-style Polymarket proxy wallets set POLYMARKET_SIGNATURE_TYPE=1.
        # For raw EOA (no funder), use 0.
        default_sig_type = "3" if funder else "0"
        return cls(
            api_key        = env.get("POLYMARKET_API_KEY")        or kv.get("api key")        or "",
            api_secret     = env.get("POLYMARKET_API_SECRET")     or kv.get("api secret")     or "",
            api_passphrase = env.get("POLYMARKET_API_PASSPHRASE") or kv.get("api passphrase") or "",
            private_key    = env.get("POLYMARKET_PRIVATE_KEY")    or kv.get("private key")    or None,
            funder         = funder,
            signature_type = int(env.get("POLYMARKET_SIGNATURE_TYPE") or kv.get("signature type") or default_sig_type),
            chain_id       = int(env.get("POLYMARKET_CHAIN_ID", "137")),
        )

    def has_signing_key(self) -> bool:
        return bool(self.private_key)

    def has_api_creds(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FillReceipt:
    """What the router returns after submitting an order. In DRY-RUN
    mode the fields below are synthetic."""
    success:        bool
    dry_run:        bool
    order_id:       Optional[str]
    filled_size:    float
    avg_price:      float
    status:         str
    raw_response:   dict = field(default_factory=dict)
    err:            Optional[str] = None


class PolymarketOrderRouter:
    def __init__(self, *, creds: PolymarketCreds,
                 live: bool = False,
                 logger: logging.Logger | None = None) -> None:
        self.creds = creds
        self.live = bool(live and creds.has_signing_key()
                         and os.environ.get("ENABLE_REAL_ORDERS") == "1")
        self.logger = logger or logging.getLogger("order_router")
        self._client = None
        self._quote_client = None
        if self.live:
            self._init_client()
        else:
            self.logger.warning(
                "order router in DRY-RUN — set ENABLE_REAL_ORDERS=1 and provide POLYMARKET_PRIVATE_KEY "
                "(env or API_Keys file 'Private Key:' line) to enable live order placement."
            )

    def _init_client(self) -> None:
        # Imported lazily so the bot starts without py-clob-client-v2 installed if dry-run
        from py_clob_client_v2 import ApiCreds, ClobClient
        self._api_creds = ApiCreds(
            api_key=self.creds.api_key, api_secret=self.creds.api_secret,
            api_passphrase=self.creds.api_passphrase,
        )
        kwargs = dict(
            host="https://clob.polymarket.com",
            chain_id=self.creds.chain_id,
            key=self.creds.private_key,
            creds=self._api_creds,
        )
        if self.creds.funder:
            kwargs["funder"] = self.creds.funder
            kwargs["signature_type"] = self.creds.signature_type
        self._client = ClobClient(**kwargs)
        try:
            self._client.set_api_creds(self._api_creds)
        except Exception:
            pass
        self.logger.info(
            "CLOB v2 client initialized (live mode, signature_type=%s, funder=%s)",
            self.creds.signature_type,
            "yes" if self.creds.funder else "no",
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    # Polymarket rejects any marketable order whose notional is below $1.
    def _quote_book_client(self):
        if self.live and self._client is not None:
            return self._client
        if self._quote_client is None:
            from py_clob_client_v2 import ClobClient
            self._quote_client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=self.creds.chain_id,
            )
        return self._quote_client

    POLYMARKET_MIN_NOTIONAL_USD = 1.0
    # ...and (newer rule) any order whose SIZE is below 5 shares.
    POLYMARKET_MIN_SHARES = 5.0

    @staticmethod
    def _quantize_shares(limit_price: float, target_shares: float,
                         *, min_shares: float = 5.0) -> tuple:
        """Return (size_shares, notional) for a FIXED-SHARE target, satisfying:

          * size_shares has ≤4 decimal places  (taker amount rule)
          * limit_price × size_shares has ≤2 decimal places  (maker amount rule)
          * size_shares ≥ `min_shares`  (Polymarket's 5-share hard floor)

        Picks the valid size CLOSEST to `target_shares` (never below the floor),
        so fixed-size sizing stays ~constant instead of being shrunk by the
        notional round-trip in `_quantize_order` (which optimises $ spend and at
        an "ugly" price like 0.47 collapsed 5.1 shares → 4.0, under the floor).
        Pure integer arithmetic.
        """
        from math import gcd, ceil
        price_basis = round(limit_price * 100)        # cents, e.g. 47 for 0.47
        if price_basis <= 0:
            return 0.0, 0.0
        # valid sizes are multiples of L (in 1e-4-share units): the smallest step
        # for which price_basis × size lands on a clean cent.
        L = 10000 // gcd(price_basis, 10000)
        floor_u = ((int(ceil(min_shares * 10000)) + L - 1) // L) * L  # smallest valid ≥ floor
        t_u = int(round(target_shares * 10000))
        k = (t_u // L) * L                            # largest valid <= target
        if k < floor_u:
            k = floor_u
        size = k / 10000
        notional = round(price_basis * size / 100, 2)
        return size, notional

    @staticmethod
    def _quantize_order(limit_price: float, target_notional: float,
                        *, min_notional: float = 1.0,
                        max_over_target: float = 1.0) -> tuple:
        """Return (size_shares, notional) satisfying Polymarket's precision rules:

          * size_shares has ≤4 decimal places  (taker amount constraint)
          * limit_price × size_shares has ≤2 decimal places  (maker amount constraint)
          * notional ≥ `min_notional`  (Polymarket's $1 hard floor)

        Strategy:
          1. Prefer the LARGEST valid notional in [min_notional, target] (walk down).
          2. If none exists there (common for "ugly" prices like 0.47 where the
             only clean combo under $1 is $0.94), walk UP from max(min, target)
             to the SMALLEST valid notional that clears the floor, capped at
             target + max_over_target so we never wildly overspend.

        Pure integer arithmetic — no float rounding surprises.
        Returns (0.0, 0.0) if no valid pair can be found in range.
        """
        price_basis = round(limit_price * 100)        # cents, e.g. 47 for 0.47
        if price_basis <= 0:
            return 0.0, 0.0
        target_cents = int(round(target_notional * 100))
        min_cents    = int(round(min_notional * 100))

        def _valid(n_cents: int) -> tuple | None:
            # size = n_cents / price_basis ; need size to be a clean 4dp value
            numerator = n_cents * 10000
            if numerator % price_basis == 0:
                return (numerator // price_basis) / 10000, n_cents / 100
            return None

        # (1) walk DOWN from target to min — prefer largest ≤ target but ≥ floor
        if target_cents >= min_cents:
            for n_cents in range(target_cents, min_cents - 1, -1):
                got = _valid(n_cents)
                if got:
                    return got
        # (2) walk UP from max(min, target) to clear the $1 floor
        start = max(min_cents, target_cents)
        ceil_cents = start + int(round(max_over_target * 100))
        for n_cents in range(start, ceil_cents + 1):
            got = _valid(n_cents)
            if got:
                return got
        return 0.0, 0.0

    # ── public API ────────────────────────────────────────────────────────────
    def submit_buy_down_fak(self, *,
                            down_token_id: str,
                            limit_price: float,
                            size_shares: float) -> FillReceipt:
        """Buy DOWN as a MARKET order, FAK semantics. We submit a true
        Polymarket market order with `amount = limit_price * size_shares`
        USDC; the matching engine sweeps the book at whatever ask exists
        and FAK accepts partial fills. No limit price -> no
        "no orders found to match" rejection (which is what FAK with a
        too-low limit was causing). Just take what we can get.

        We keep the (limit_price, size_shares) signature for callsite
        compatibility -- they're used solely to compute the spend amount.
        """
        limit_price = round(float(limit_price), 2)
        # Spend amount in USDC. Quantize the AMOUNT to 2dp (Polymarket
        # market-buy precision rule).
        amount_usdc = round(limit_price * float(size_shares), 2)
        if amount_usdc < self.POLYMARKET_MIN_NOTIONAL_USD:
            amount_usdc = self.POLYMARKET_MIN_NOTIONAL_USD
        if amount_usdc <= 0 or limit_price <= 0 or limit_price >= 1:
            return FillReceipt(success=False, dry_run=not self.live, order_id=None,
                               filled_size=0.0, avg_price=0.0, status="reject",
                               err=f"degenerate: amount=${amount_usdc:.2f}")

        if not self.live:
            return FillReceipt(
                success=True, dry_run=True, order_id="DRY-RUN-MKT",
                filled_size=amount_usdc / max(limit_price, 1e-6),
                avg_price=limit_price,
                status="dry_run_simulated_market_fill",
                raw_response={"token_id": down_token_id, "amount": amount_usdc,
                              "side": "BUY", "order_type": "MARKET+FAK"},
            )

        try:
            from py_clob_client_v2 import (
                MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side
            )
            # MARKET order with FAK = "buy this amount of USDC worth of the
            # token at whatever the book gives us; partial fills are fine".
            # No price = no rejection-for-no-match.
            market_args = MarketOrderArgs(
                token_id=down_token_id,
                amount=amount_usdc,
                side=Side.BUY,
            )
            self.logger.info(
                "MARKET DOWN order: token=%s amount=$%.2f (FAK)",
                down_token_id, amount_usdc,
            )
            resp = self._client.create_and_post_market_order(
                order_args=market_args,
                options=PartialCreateOrderOptions(tick_size="0.01"),
                order_type=OrderType.FAK,
            )
            self.logger.info("MARKET DOWN response: %s", resp)
            status_str = str(resp.get("status") or "")
            success = bool(
                resp.get("success", False)
                or status_str in ("matched", "filled", "delayed", "MMA")
            )
            def _parse_filled(r: dict) -> float:
                for key in ("filledSize", "filled_size", "takingAmount",
                            "takerAmount", "filled"):
                    v = r.get(key)
                    if v is not None:
                        try: return float(v)
                        except (TypeError, ValueError): pass
                maker = r.get("makingAmount") or r.get("makerAmount")
                if maker is not None:
                    try: return float(maker) / max(limit_price, 1e-6)
                    except (TypeError, ValueError): pass
                return 0.0
            filled = _parse_filled(resp)
            maker_raw = resp.get("makingAmount") or resp.get("makerAmount")
            try: spent = float(maker_raw) if maker_raw is not None else 0.0
            except (TypeError, ValueError): spent = 0.0
            avg_price = spent / filled if spent > 0 and filled > 0 else limit_price
            return FillReceipt(
                success=success, dry_run=False,
                order_id=str(resp.get("orderID") or resp.get("orderId")
                            or resp.get("id") or ""),
                filled_size=filled, avg_price=avg_price,
                status=status_str or "submitted",
                raw_response=resp,
            )
        except Exception as exc:
            self.logger.exception("MARKET DOWN submission failed: %r", exc)
            return FillReceipt(
                success=False, dry_run=False, order_id=None,
                filled_size=0.0, avg_price=0.0, status="error",
                err=repr(exc),
            )

    # ── BUY UP (MARKET order, FAK semantics) ───────────────────────────────
    # Mirrors submit_buy_down_fak. Submits a Polymarket MARKET order with
    # FAK so the order sweeps whatever the book gives us at any price, and
    # accepts partial fills. No limit price -> no "no orders found" reject.
    def submit_buy_up_fak(self, *,
                          up_token_id: str,
                          limit_price: float,
                          size_shares: float) -> FillReceipt:
        """Buy UP as a MARKET order, FAK semantics. (limit_price, size_shares)
        are used only to compute the USDC spend; there is no limit price on
        the wire."""
        limit_price = round(float(limit_price), 2)
        amount_usdc = round(limit_price * float(size_shares), 2)
        if amount_usdc < self.POLYMARKET_MIN_NOTIONAL_USD:
            amount_usdc = self.POLYMARKET_MIN_NOTIONAL_USD
        if amount_usdc <= 0 or limit_price <= 0 or limit_price >= 1:
            return FillReceipt(success=False, dry_run=not self.live, order_id=None,
                               filled_size=0.0, avg_price=0.0, status="reject",
                               err=f"degenerate: amount=${amount_usdc:.2f}")

        if not self.live:
            return FillReceipt(
                success=True, dry_run=True, order_id="DRY-RUN-MKT-UP",
                filled_size=amount_usdc / max(limit_price, 1e-6),
                avg_price=limit_price,
                status="dry_run_simulated_market_fill",
                raw_response={"token_id": up_token_id, "amount": amount_usdc,
                              "side": "BUY", "outcome": "Up",
                              "order_type": "MARKET+FAK"},
            )

        try:
            from py_clob_client_v2 import (
                MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side
            )
            market_args = MarketOrderArgs(
                token_id=up_token_id,
                amount=amount_usdc,
                side=Side.BUY,
            )
            self.logger.info(
                "MARKET UP order: token=%s amount=$%.2f (FAK)",
                up_token_id, amount_usdc,
            )
            resp = self._client.create_and_post_market_order(
                order_args=market_args,
                options=PartialCreateOrderOptions(tick_size="0.01"),
                order_type=OrderType.FAK,
            )
            self.logger.info("MARKET UP response: %s", resp)
            status_str = str(resp.get("status") or "")
            success = bool(
                resp.get("success", False)
                or status_str in ("matched", "filled", "delayed", "MMA")
            )
            def _parse_filled(r: dict) -> float:
                for key in ("filledSize", "filled_size", "takingAmount",
                            "takerAmount", "filled"):
                    v = r.get(key)
                    if v is not None:
                        try: return float(v)
                        except (TypeError, ValueError): pass
                maker = r.get("makingAmount") or r.get("makerAmount")
                if maker is not None:
                    try: return float(maker) / max(limit_price, 1e-6)
                    except (TypeError, ValueError): pass
                return 0.0
            filled = _parse_filled(resp)
            maker_raw = resp.get("makingAmount") or resp.get("makerAmount")
            try: spent = float(maker_raw) if maker_raw is not None else 0.0
            except (TypeError, ValueError): spent = 0.0
            avg_price = spent / filled if spent > 0 and filled > 0 else limit_price
            return FillReceipt(
                success=success, dry_run=False,
                order_id=str(resp.get("orderID") or resp.get("orderId")
                            or resp.get("id") or ""),
                filled_size=filled, avg_price=avg_price,
                status=status_str or "submitted",
                raw_response=resp,
            )
        except Exception as exc:
            self.logger.exception("MARKET UP submission failed: %r", exc)
            return FillReceipt(
                success=False, dry_run=False, order_id=None,
                filled_size=0.0, avg_price=0.0, status="error",
                err=repr(exc),
            )

    # ── IOC/FAK TAKER (crossing limit) — never rests a ghost order ──────────
    #
    # This is a LIMIT BUY with post_only=False and a marketable buffer above the
    # quoted ask. It crosses the book up to the limit, but FAK cancels any
    # unfilled remainder immediately. That preserves the bot's slippage cap and
    # avoids "status=live, filled_size=0" orders being mistaken for positions.
    MARKETABLE_BUFFER = 0.02   # cents above the quoted ask to guarantee a cross

    @staticmethod
    def _best_ask_from_book(book: dict) -> float | None:
        asks = book.get("asks") if isinstance(book, dict) else getattr(book, "asks", None)
        best = None
        for level in asks or ():
            try:
                px = float(level.get("price") if isinstance(level, dict) else level.price)
                sz = float(level.get("size") if isinstance(level, dict) else level.size)
            except (AttributeError, TypeError, ValueError):
                continue
            if sz <= 0:
                continue
            best = px if best is None else min(best, px)
        return best

    def submit_buy_gtc(self, *,
                       token_id: str,
                       limit_price: float,
                       size_shares: float,
                       side_label: str = "?",
                       buffer: float | None = None,
                       fixed_shares: float | None = None) -> FillReceipt:
        """Buy `~limit_price * size_shares` USDC worth of `token_id` as an
        immediate-or-cancel crossing limit order (taker). Fills what it can
        immediately and cancels the rest. Returns a FillReceipt whose
        ``success`` means "non-zero fill", not merely "order accepted".

        `buffer` = max slippage above the quote we'll pay to cross (the wire
        limit is quote+buffer). Defaults to MARKETABLE_BUFFER (2c) so existing
        callers are unchanged; the fair_value bot passes a larger cap to reach
        deeper book and fill fully on thin tops."""
        quoted = round(float(limit_price), 2)
        if quoted <= 0 or quoted >= 1:
            return FillReceipt(success=False, dry_run=not self.live, order_id=None,
                               filled_size=0.0, avg_price=0.0, status="reject",
                               err=f"degenerate price {quoted}")
        # Cross the spread by a buffer (= max slippage) so the order takes.
        buf = self.MARKETABLE_BUFFER if buffer is None else float(buffer)
        exec_quote = quoted
        live_ask = None
        quote_check_enabled = (
            fixed_shares is not None
            and os.getenv("FV_EXEC_LIVE_QUOTE_CHECK", "1") == "1"
            and (self.live or os.getenv("FV_EXEC_QUOTE_CHECK_DRY_RUN", "0") == "1")
        )
        if quote_check_enabled:
            try:
                live_ask = self._best_ask_from_book(self._quote_book_client().get_order_book(token_id))
            except Exception as exc:
                self.logger.warning("fresh CLOB quote lookup failed token=%s: %r", token_id, exc)
                return FillReceipt(
                    success=False, dry_run=not self.live, order_id=None,
                    filled_size=0.0, avg_price=0.0, status="quote_error",
                    raw_response={"err": repr(exc)}, err="fresh quote lookup failed",
                )
            if live_ask is None or live_ask <= 0:
                return FillReceipt(
                    success=False, dry_run=not self.live, order_id=None,
                    filled_size=0.0, avg_price=0.0, status="no_live_ask",
                    raw_response={"quoted": quoted}, err="no fresh live ask",
                )
            max_quote_diff = float(os.getenv("FV_EXEC_MAX_QUOTE_DIFF", "0.02"))
            if abs(float(live_ask) - quoted) > max_quote_diff or float(live_ask) > quoted + buf:
                return FillReceipt(
                    success=False, dry_run=not self.live, order_id=None,
                    filled_size=0.0, avg_price=0.0, status="quote_mismatch",
                    raw_response={"quoted": quoted, "live_ask": float(live_ask),
                                  "max_quote_diff": max_quote_diff, "max_slippage": buf},
                    err=f"fresh live ask {float(live_ask):.2f} != recorder quote {quoted:.2f}",
                )
            exec_quote = round(float(live_ask), 2)
        # In fixed-share mode, do not use the full slippage cap as the signed
        # limit by default. Polymarket BUY matching preserves USDC spend and can
        # return *more* shares on price improvement. A wide limit therefore
        # turns "5.1 shares" into "spend up to quote+buffer"; use the fresh ask
        # as the wire limit unless explicitly overridden.
        fixed_limit_buffer = float(os.getenv("FV_EXEC_LIMIT_BUFFER", "0.00")) if fixed_shares is not None else buf
        wire_price = round(min(exec_quote + fixed_limit_buffer, quoted + buf, 0.97), 2)
        if fixed_shares is not None:
            # FIXED-SHARE sizing: hold the share count ~constant and clear the
            # 5-share floor, instead of the notional round-trip that shrinks it.
            size_q, notional_q = self._quantize_shares(
                wire_price, float(fixed_shares), min_shares=self.POLYMARKET_MIN_SHARES)
            if os.getenv("FV_REQUIRE_EXACT_SHARES", "0") == "1" and abs(size_q - float(fixed_shares)) > 1e-9:
                return FillReceipt(
                    success=False, dry_run=not self.live, order_id=None,
                    filled_size=0.0, avg_price=0.0, status="size_not_exact",
                    raw_response={"target_shares": float(fixed_shares),
                                  "quantized_shares": size_q,
                                  "wire_price": wire_price},
                    err=f"cannot place exactly {float(fixed_shares):.4f} shares at price {wire_price:.2f}",
                )
        else:
            target_notional = max(quoted * float(size_shares),
                                  self.POLYMARKET_MIN_NOTIONAL_USD)
            size_q, notional_q = self._quantize_order(
                wire_price, target_notional, min_notional=self.POLYMARKET_MIN_NOTIONAL_USD)
        if size_q <= 0:
            return FillReceipt(success=False, dry_run=not self.live, order_id=None,
                               filled_size=0.0, avg_price=0.0, status="reject",
                               err=f"no valid size at price={wire_price}")

        if not self.live:
            return FillReceipt(
                success=True, dry_run=True, order_id="DRY-RUN-GTC",
                filled_size=size_q, avg_price=wire_price,
                status="dry_run_simulated_gtc_fill",
                raw_response={"token_id": token_id, "price": wire_price,
                               "quoted": quoted, "live_ask": live_ask,
                               "size": size_q, "notional": notional_q,
                               "side": "BUY", "outcome": side_label,
                               "order_type": "FAK", "post_only": False},
            )

        try:
            from py_clob_client_v2 import (
                OrderArgs, OrderType, PartialCreateOrderOptions, Side
            )
            order_args = OrderArgs(
                token_id=token_id,
                price=wire_price,
                size=size_q,
                side=Side.BUY,
            )
            self.logger.info(
                "FAK BUY %s: token=%s quote=%.2f live_ask=%s wire=%.2f size=%.4f (~$%.2f, crossing)",
                side_label, token_id, quoted,
                f"{live_ask:.2f}" if live_ask is not None else "?",
                wire_price, size_q, notional_q,
            )
            resp = self._client.create_and_post_order(
                order_args=order_args,
                options=PartialCreateOrderOptions(tick_size="0.01"),
                order_type=OrderType.FAK,
                post_only=False,
            )
            self.logger.info("FAK BUY %s response: %s", side_label, resp)
            status_str = str(resp.get("status") or "")

            def _parse_filled(r: dict) -> float:
                for key in ("filledSize", "filled_size", "takingAmount",
                            "takerAmount", "filled"):
                    v = r.get(key)
                    if v is not None:
                        try: return float(v)
                        except (TypeError, ValueError): pass
                maker = r.get("makingAmount") or r.get("makerAmount")
                if maker is not None:
                    try: return float(maker) / max(wire_price, 1e-6)
                    except (TypeError, ValueError): pass
                return 0.0

            filled = _parse_filled(resp)
            maker_raw = resp.get("makingAmount") or resp.get("makerAmount")
            try: spent = float(maker_raw) if maker_raw is not None else 0.0
            except (TypeError, ValueError): spent = 0.0
            avg_price = spent / filled if spent > 0 and filled > 0 else wire_price
            success = filled > 0.0
            return FillReceipt(
                success=success, dry_run=False,
                order_id=str(resp.get("orderID") or resp.get("orderId")
                            or resp.get("id") or ""),
                filled_size=filled, avg_price=avg_price,
                status=status_str or ("filled" if success else "no_fill"),
                raw_response=resp,
                err=None if success else "no immediate fill",
            )
        except Exception as exc:
            if exc.__class__.__name__ == "PolyApiException":
                error_msg = getattr(exc, "error_msg", None)
                status_code = getattr(exc, "status_code", None)
                text = str(error_msg)
                if status_code == 400 and "no orders found to match" in text:
                    order_id = ""
                    if isinstance(error_msg, dict):
                        order_id = str(error_msg.get("orderID") or error_msg.get("orderId") or "")
                    self.logger.warning("FAK BUY %s no immediate match: %s", side_label, error_msg)
                    return FillReceipt(
                        success=False, dry_run=False, order_id=order_id,
                        filled_size=0.0, avg_price=0.0, status="no_fill",
                        raw_response=error_msg if isinstance(error_msg, dict) else {"error": text},
                        err="no immediate fill",
                    )
            self.logger.exception("FAK BUY %s submission failed: %r", side_label, exc)
            return FillReceipt(
                success=False, dry_run=False, order_id=None,
                filled_size=0.0, avg_price=0.0, status="error", err=repr(exc),
            )

    # ── NEW: POST_ONLY (maker) submission and cancellation ──────────────────
    #
    # These methods exist purely to support the parallel maker strategy added
    # in live_bot.maker_strategy. They do NOT touch the FAK path above. The
    # contract is: submit_buy_down_post_only places a GTC bid that is REJECTED
    # by the matching engine if it would cross the book (so we are guaranteed
    # to be the maker, never a taker by accident). It lives in the book until
    # someone hits it or we call cancel_order to pull it.
    def submit_buy_down_post_only(self, *,
                                  down_token_id: str,
                                  limit_price: float,
                                  size_shares: float) -> FillReceipt:
        """Place a GTC POST_ONLY BUY bid for `size_shares` of the DOWN token
        at `limit_price` USDC each. Returns a FillReceipt where:
          * success=True  => order successfully placed on the book.
                             filled_size at this moment is almost always 0
                             (POST_ONLY rejects any immediate match).
          * success=False => placement rejected (would have crossed, balance
                             issue, validation error, etc.).
        Read `raw_response["orderID"]` to cancel/inspect later."""
        limit_price = round(float(limit_price), 2)
        target_notional = limit_price * float(size_shares)
        size_shares, target_notional = self._quantize_order(
            limit_price, target_notional,
            min_notional=self.POLYMARKET_MIN_NOTIONAL_USD)

        if size_shares <= 0 or limit_price <= 0 or limit_price >= 1:
            return FillReceipt(success=False, dry_run=not self.live, order_id=None,
                               filled_size=0.0, avg_price=0.0, status="reject",
                               err=f"degenerate: price={limit_price} size={size_shares}")

        if not self.live:
            return FillReceipt(
                success=True, dry_run=True, order_id="DRY-RUN-POSTONLY",
                filled_size=0.0,   # POST_ONLY almost never fills on submit
                avg_price=limit_price, status="dry_run_posted",
                raw_response={"token_id": down_token_id, "price": limit_price,
                              "size": size_shares, "side": "BUY",
                              "order_type": "GTC", "post_only": True},
            )

        try:
            from py_clob_client_v2 import (
                OrderArgs, OrderType, PartialCreateOrderOptions, Side
            )
            order_args = OrderArgs(
                token_id=down_token_id,
                price=limit_price,
                size=size_shares,
                side=Side.BUY,
            )
            self.logger.info(
                "POST_ONLY GTC order: token=%s price=%.4f size=%.4f",
                down_token_id, limit_price, size_shares,
            )
            resp = self._client.create_and_post_order(
                order_args=order_args,
                options=PartialCreateOrderOptions(tick_size="0.01"),
                order_type=OrderType.GTC,
                post_only=True,
            )
            self.logger.info("POST_ONLY response: %s", resp)
            status_str = str(resp.get("status") or "")
            # For POST_ONLY GTC, success means "successfully placed on book".
            # Polymarket may return status "live", "delayed", or even "matched"
            # if there was a tiny partial fill before the order rested.
            success = bool(
                resp.get("success", False)
                or status_str in ("live", "delayed", "matched")
                or resp.get("orderID") or resp.get("orderId") or resp.get("id")
            )
            # Parse any initial fill (rare for POST_ONLY but possible).
            def _parse_filled(r: dict) -> float:
                for key in ("filledSize", "filled_size", "takingAmount",
                            "takerAmount", "filled"):
                    v = r.get(key)
                    if v is not None:
                        return float(v)
                maker = r.get("makingAmount") or r.get("makerAmount")
                if maker is not None:
                    return float(maker) / max(limit_price, 1e-6)
                return 0.0

            filled = _parse_filled(resp)
            maker_raw = resp.get("makingAmount") or resp.get("makerAmount")
            spent = float(maker_raw) if maker_raw is not None else 0.0
            avg_price = spent / filled if spent > 0 and filled > 0 else limit_price
            return FillReceipt(
                success=success, dry_run=False,
                order_id=str(resp.get("orderID") or resp.get("orderId")
                            or resp.get("id") or ""),
                filled_size=filled, avg_price=avg_price,
                status=status_str or "posted",
                raw_response=resp,
            )
        except Exception as exc:
            self.logger.exception("POST_ONLY submission failed: %r", exc)
            return FillReceipt(
                success=False, dry_run=False, order_id=None,
                filled_size=0.0, avg_price=0.0, status="error",
                err=repr(exc),
            )

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order. Returns the SDK's raw response dict,
        which on Polymarket includes the order's final filled_size /
        remaining_size so the caller can detect partial fills before cancel."""
        if not self.live:
            return {"dry_run": True, "orderID": order_id, "status": "dry_cancelled"}
        try:
            from py_clob_client_v2 import OrderPayload
            resp = self._client.cancel_order(OrderPayload(orderID=order_id))
            self.logger.info("cancel_order response (id=%s): %s", order_id, resp)
            return resp if isinstance(resp, dict) else {"raw": resp}
        except Exception as exc:
            self.logger.warning("cancel_order failed (id=%s): %r", order_id, exc)
            return {"error": repr(exc)}

    def get_order(self, order_id: str) -> dict:
        """Look up an order's current state on Polymarket. Used by the
        maker strategy to reconcile fills before pulling unfilled remainder."""
        if not self.live:
            return {"dry_run": True, "orderID": order_id, "status": "unknown"}
        try:
            resp = self._client.get_order(order_id)
            return resp if isinstance(resp, dict) else {"raw": resp}
        except Exception as exc:
            self.logger.warning("get_order failed (id=%s): %r", order_id, exc)
            return {"error": repr(exc)}

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_token_balance(self, token_id: str) -> Optional[float]:
        if not self.live: return None
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
            return float(self._client.get_balance_allowance(params).get("balance", 0))
        except Exception as exc:
            self.logger.warning("balance lookup failed: %r", exc)
            return None

    def preflight(self) -> dict:
        """Read-only sanity check before any orders. Returns a dict
        describing wallet state. Logs warnings on common gotchas."""
        report: dict = {"live": self.live}
        if not self.live:
            report["note"] = "dry-run mode; preflight is informational only"
            return report
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            report["eoa_address"] = self._client.get_address()
            report["funder"] = self.creds.funder
            usdc = self._client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
            balance = int(usdc.get("balance", "0")) / 1_000_000     # USDC = 6 decimals
            allowances_raw = usdc.get("allowances", {}) or {}
            allowances = {addr: int(amt) / 1_000_000 for addr, amt in allowances_raw.items()}
            report["usdc_balance"] = balance
            report["usdc_allowances"] = allowances
            if balance < 1.0:
                self.logger.warning("USDC balance is %.2f — fund the safe wallet %s before trading",
                                    balance, self.creds.funder)
            zero_allow = [a for a, v in allowances.items() if v < 1.0]
            if zero_allow:
                self.logger.warning("USDC allowance is 0 for %d exchange contract(s) — "
                                    "orders WILL be rejected until you call "
                                    "client.update_balance_allowance(...) once. "
                                    "Contracts needing allowance: %s",
                                    len(zero_allow), zero_allow)
        except Exception as exc:
            self.logger.exception("preflight failed: %r", exc)
            report["error"] = repr(exc)
        return report

    def set_max_usdc_allowance(self) -> bool:
        """One-shot helper: set USDC allowance to max for all Polymarket
        exchange contracts. Call once after first USDC deposit."""
        if not self.live:
            self.logger.error("set_max_usdc_allowance requires live=True"); return False
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = self._client.update_balance_allowance(params)
            self.logger.info("allowance update submitted: %r", resp)
            return True
        except Exception as exc:
            self.logger.exception("allowance update failed: %r", exc)
            return False
