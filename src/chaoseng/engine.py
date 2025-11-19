import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chaoseng.model import OmegaModel
from chaoseng.alpaca_client import AlpacaClient


class ChaosEngineOmegaHybrid:
    """
    Ultra-adaptive hybrid engine:

    - Multi-ticker
    - Multi-timeframe-ish context via recent bars
    - AI-driven long/short decisions
    - AI-driven position sizing (S3)
    - 1–2 day max hold horizon logic
    - Live Alpaca execution (paper/live controlled via config/env)
    - Terminal-style log output
    - Trade history + PnL summary
    - Optional Telegram alerts on ENTER/EXIT
    """

    def __init__(self, config_path: str = "config.json"):
        self.device = torch.device("cpu")
        self.model = OmegaModel().to(self.device)
        self.model.eval()  # inference mode

        # Load config (optional)
        cfg: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                try:
                    cfg = json.load(f)
                except Exception:
                    cfg = {}
        self.cfg = cfg

        # Alpaca client (keys from config or env)
        self.alpaca = AlpacaClient(config_path=config_path)
        self.tickers: List[str] = self.alpaca.tickers

        # Internal state
        self.position_meta: Dict[str, Dict[str, Any]] = {}  # symbol -> {entry_time, side, entry_price, qty}
        self.start_equity = self.alpaca.get_account_equity()

        # Trade history (in-memory)
        self.trade_log: List[Dict[str, Any]] = []

        # Telegram setup
        self.telegram_token = cfg.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = cfg.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _prepare_tensor(self, df: pd.DataFrame, window: int = 60) -> torch.Tensor:
        """
        Turn OHLCV DataFrame into normalized tensor for OmegaModel.
        """
        if df is None or len(df) < 10:
            return None

        if len(df) > window:
            df = df.iloc[-window:]
        arr = df[["open", "high", "low", "close", "volume"]].values.astype("float32")

        # Normalize prices by last close, volume by mean
        ref = arr[-1, 3] if arr[-1, 3] != 0 else 1.0
        arr[:, :4] /= ref
        vol_ref = max(arr[:, 4].mean(), 1.0)
        arr[:, 4] /= vol_ref

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # [seq, batch=1, feat]
        return x.to(self.device)

    def _ai_position_size(self, symbol: str, equity: float, probs: np.ndarray, df: pd.DataFrame) -> int:
        """
        S3: AI-based position sizing.
        Uses:
        - confidence (max prob)
        - volatility (ATR-ish)
        - risk fraction of equity (adaptive)
        """
        conf = float(probs.max())
        base_risk_frac = 0.01 + 0.03 * (conf - 0.5)  # roughly 1% → 4% as confidence rises
        base_risk_frac = max(0.005, min(base_risk_frac, 0.05))  # clamp 0.5% – 5%

        closes = df["close"].values
        if len(closes) < 15:
            vol = 0.01
        else:
            rets = np.diff(closes[-15:]) / closes[-15:-1]
            vol = float(np.std(rets)) or 0.01

        # Higher volatility → smaller position
        vol_adj = 1.0 / (1.0 + 20.0 * vol)
        risk_frac = base_risk_frac * vol_adj

        assumed_stop = 0.015  # assume ~1.5% stop distance
        dollar_risk = equity * risk_frac
        if dollar_risk <= 0:
            return 0

        price = closes[-1]
        qty = dollar_risk / (price * assumed_stop)
        qty = math.floor(qty)
        return max(qty, 0)

    def _should_exit_position(self, symbol: str, side: str, probs: np.ndarray, df: pd.DataFrame) -> bool:
        """
        1–2 day horizon exit logic + safety exits.
        """
        meta = self.position_meta.get(symbol)
        now = datetime.now(timezone.utc)

        if meta:
            entry_time = meta["entry_time"]
            held_hours = (now - entry_time).total_seconds() / 3600.0
            # Hard max hold: ~36 hours (within your 1–2 day rule)
            if held_hours > 36:
                return True

        buy_p, hold_p, sell_p = probs

        # Flip if opposite side probability gets strong
        if side == "long" and sell_p > 0.55:
            return True
        if side == "short" and buy_p > 0.55:
            return True

        closes = df["close"].values
        if len(closes) >= 3:
            c1, c2 = closes[-2], closes[-1]
            ret = (c2 - c1) / c1
            # If moving aggressively against us in one bar
            if side == "long" and ret < -0.01:
                return True
            if side == "short" and ret > 0.01:
                return True

        return False

    # ---------------- Pattern / score helpers --------------------
    def _pattern_bias(self, df: pd.DataFrame) -> float:
        """
        Very simple price action bias:
        - Looks at last candle body + wicks
        - Looks at short-term trend via slope of closes
        Returns a bias in [-0.3, +0.3] (negative = bearish, positive = bullish).
        """
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values

        if len(closes) < 5:
            return 0.0

        last_open = opens[-1]
        last_close = closes[-1]
        last_high = highs[-1]
        last_low = lows[-1]

        body = last_close - last_open
        upper_wick = last_high - max(last_close, last_open)
        lower_wick = min(last_close, last_open) - last_low

        # Normalize by price
        ref_price = max(last_close, 1e-6)
        body_n = body / ref_price
        uw_n = upper_wick / ref_price
        lw_n = abs(lower_wick) / ref_price

        # Trend via simple slope of last 10 closes
        n = min(len(closes), 10)
        xs = np.arange(n)
        slope = 0.0
        try:
            slope = float(np.polyfit(xs, closes[-n:], 1)[0]) / ref_price
        except Exception:
            slope = 0.0

        bias = 0.0

        # Bullish body + long lower wick + upward slope
        if body_n > 0 and lw_n > uw_n * 1.3 and slope > 0:
            bias += 0.15
        # Bearish body + long upper wick + downward slope
        if body_n < 0 and uw_n > lw_n * 1.3 and slope < 0:
            bias -= 0.15

        # Extra bias if slope is strong
        if slope > 0.001:
            bias += 0.1
        if slope < -0.001:
            bias -= 0.1

        # Clamp
        bias = max(-0.3, min(0.3, bias))
        return float(bias)

    def _send_telegram(self, text: str):
        """
        Fire-and-forget Telegram alert if configured.
        """
        if not self.telegram_enabled:
            return
        try:
            import requests

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {"chat_id": self.telegram_chat_id, "text": text}
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[Telegram] Error sending message: {e}")

    def _log_trade(
        self,
        kind: str,
        symbol: str,
        side: str,
        qty: float,
        score: float = None,
        probs: np.ndarray = None,
        entry_price: float = None,
        exit_price: float = None,
        pnl: float = None,
    ):
        """
        Append a trade event to in-memory history and optionally send Telegram.
        kind: "ENTER" or "EXIT"
        """
        now = datetime.now(timezone.utc).isoformat()
        buy_p = hold_p = sell_p = None
        if probs is not None and len(probs) == 3:
            buy_p, hold_p, sell_p = [float(x) for x in probs]

        entry = {
            "timestamp": now,
            "kind": kind,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "score": float(score) if score is not None else None,
            "buy_p": buy_p,
            "hold_p": hold_p,
            "sell_p": sell_p,
            "entry_price": float(entry_price) if entry_price is not None else None,
            "exit_price": float(exit_price) if exit_price is not None else None,
            "pnl": float(pnl) if pnl is not None else None,
        }
        self.trade_log.append(entry)

        # Telegram alert
        if kind == "ENTER":
            msg = f"OMEGA ENTER {symbol} {side.upper()} x{qty} @ {entry_price:.2f if entry_price else 0.0}"
        elif kind == "EXIT":
            msg = f"OMEGA EXIT {symbol} {side.upper()} x{qty} @ {exit_price:.2f if exit_price else 0.0} PnL={pnl:.2f if pnl else 0.0}"
        else:
            msg = None

        if msg:
            try:
                self._send_telegram(msg)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Public API for FastAPI
    # -------------------------------------------------------------------------
    def get_trade_history(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Return the most recent 'limit' trade log entries.
        """
        if limit <= 0:
            return []
        return self.trade_log[-limit:]

    def get_pnl_summary(self) -> Dict[str, Any]:
        """
        Compute simple realized PnL stats from EXIT trades.
        """
        exits = [t for t in self.trade_log if t.get("kind") == "EXIT" and t.get("pnl") is not None]
        if not exits:
            return {
                "total_realized_pnl": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }

        pnls = [t["pnl"] for t in exits]
        total_pnl = float(sum(pnls))
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        num_trades = len(pnls)
        win_rate = float(len(wins) / num_trades) if num_trades > 0 else 0.0
        avg_win = float(sum(wins) / len(wins)) if wins else 0.0
        avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
        max_win = float(max(wins)) if wins else 0.0
        max_loss = float(min(losses)) if losses else 0.0

        return {
            "total_realized_pnl": total_pnl,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
        }

    # -------------------------------------------------------------------------
    # Main live loop step
    # -------------------------------------------------------------------------
    async def live_step(self) -> str:
        """
        One live evaluation step across tickers:
        - Pull recent bars for each ticker
        - Score with Omega model (+ pattern bias)
        - Decide best action
        - Execute via Alpaca
        - Return terminal-style log string
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        equity = self.alpaca.get_account_equity()

        ticker_logs: List[str] = []
        best_symbol = None
        best_score = 0.0
        best_side = "hold"
        best_probs = None
        best_df = None

        # ---------------- Evaluate each ticker ----------------
        for symbol in self.tickers:
            df = self.alpaca.get_recent_bars(symbol, timeframe="1Min", limit=120)
            if df is None or len(df) < 20:
                ticker_logs.append(f"{symbol}: insufficient data")
                continue

            x = self._prepare_tensor(df)
            if x is None:
                ticker_logs.append(f"{symbol}: not enough candles")
                continue

            with torch.no_grad():
                logits = self.model(x)
                probs_t = F.softmax(logits, dim=-1)[0]
                probs = probs_t.detach().cpu().numpy()

            buy_p, hold_p, sell_p = probs

            # Pattern bias modifies buy/sell weighting (Ultra adaptive flavor)
            bias = self._pattern_bias(df)
            buy_p_adj = buy_p + max(0.0, bias)
            sell_p_adj = sell_p + max(0.0, -bias)
            hold_p_adj = hold_p

            # Re-normalize
            total_adj = buy_p_adj + hold_p_adj + sell_p_adj
            if total_adj > 0:
                buy_p_adj /= total_adj
                hold_p_adj /= total_adj
                sell_p_adj /= total_adj

            probs_adj = np.array([buy_p_adj, hold_p_adj, sell_p_adj], dtype=float)

            # Decide side
            max_p = float(probs_adj.max())
            if max_p < 0.5:
                side = "hold"
            elif buy_p_adj > sell_p_adj:
                side = "long"
            else:
                side = "short"

            # Volatility factor
            closes = df["close"].values
            if len(closes) > 10:
                rets = np.diff(closes[-10:]) / closes[-10:-1]
                vol = float(np.std(rets)) or 0.01
            else:
                vol = 0.01
            vol_score = 1.0 + min(vol * 100, 1.0)  # up to +1

            # Pattern bonus
            pattern_bonus = 1.0 + abs(bias)  # 1.0–1.3 roughly
            score = max_p * vol_score * pattern_bonus

            ticker_logs.append(
                f"{symbol}: BUY={buy_p_adj:.2f} HOLD={hold_p_adj:.2f} SELL={sell_p_adj:.2f} "
                f"SIDE={side} SCORE={score:.2f}"
            )

            if score > best_score and side != "hold":
                best_score = score
                best_symbol = symbol
                best_side = side
                best_probs = probs_adj
                best_df = df

        # ---------------- Manage existing positions ----------------
        action_log_lines: List[str] = []

        for symbol in self.tickers:
            qty, avg_price = self.alpaca.get_open_position(symbol)
            held = qty != 0.0
            meta = self.position_meta.get(symbol)

            if held and meta:
                side = meta["side"]
                df = self.alpaca.get_recent_bars(symbol, timeframe="1Min", limit=120)
                x = self._prepare_tensor(df)
                if x is None:
                    continue

                with torch.no_grad():
                    logits = self.model(x)
                    probs_t = F.softmax(logits, dim=-1)[0]
                    probs = probs_t.detach().cpu().numpy()

                if self._should_exit_position(symbol, side, probs, df):
                    closes = df["close"].values
                    exit_price = float(closes[-1])
                    entry_price = float(meta.get("entry_price", exit_price))
                    trade_qty = float(meta.get("qty", qty))

                    if side == "long":
                        pnl = (exit_price - entry_price) * trade_qty
                    else:
                        pnl = (entry_price - exit_price) * trade_qty

                    self.alpaca.close_position(symbol)
                    self.position_meta.pop(symbol, None)

                    self._log_trade(
                        kind="EXIT",
                        symbol=symbol,
                        side=side,
                        qty=trade_qty,
                        score=None,
                        probs=probs,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                    )

                    action_log_lines.append(
                        f"EXIT {symbol} {side.upper()} x{int(trade_qty)} @ {exit_price:.2f} PnL={pnl:.2f} "
                        f"- securing profit / cutting risk."
                    )

        # ---------------- Enter new position if edge is strong ----------------
        if best_symbol is not None and best_probs is not None and best_df is not None:
            qty, _ = self.alpaca.get_open_position(best_symbol)
            if qty == 0.0:
                size = self._ai_position_size(best_symbol, equity, best_probs, best_df)
                if size > 0:
                    closes = best_df["close"].values
                    entry_price = float(closes[-1])
                    side_str = "buy" if best_side == "long" else "sell"

                    order = self.alpaca.submit_market_order(best_symbol, size, side_str)

                    self.position_meta[best_symbol] = {
                        "entry_time": datetime.now(timezone.utc),
                        "side": best_side,
                        "entry_price": entry_price,
                        "qty": float(size),
                    }

                    self._log_trade(
                        kind="ENTER",
                        symbol=best_symbol,
                        side=best_side,
                        qty=float(size),
                        score=best_score,
                        probs=best_probs,
                        entry_price=entry_price,
                    )

                    action_log_lines.append(
                        f"ENTER {best_symbol} {side_str.upper()} x{size} @ {entry_price:.2f} "
                        f"- AI hybrid decision (score={best_score:.2f})."
                    )

        # ---------------- Build terminal log ----------------
        header = f"[{now_str}] EQUITY={equity:.2f}"
        tickers_block = "\n".join(ticker_logs)
        actions_block = "\n".join(action_log_lines) if action_log_lines else "No new orders this cycle."
        separator = "-" * 40
        log = f"{header}\n{tickers_block}\n{actions_block}\n{separator}"
        return log
