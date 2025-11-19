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
    Omega AI Hybrid + Level 1 Adaptive Learning
    """

    def __init__(self, config_path: str = "config.json"):
        self.device = torch.device("cpu")
        self.model = OmegaModel().to(self.device)
        self.model.eval()

        # Load config file
        cfg: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                try:
                    cfg = json.load(f)
                except Exception:
                    cfg = {}
        self.cfg = cfg

        # Alpaca client
        self.alpaca = AlpacaClient(config_path=config_path)
        self.tickers: List[str] = self.alpaca.tickers

        # Position memory
        self.position_meta: Dict[str, Dict[str, Any]] = {}
        self.start_equity = self.alpaca.get_account_equity()

        # Trade history log
        self.trade_log: List[Dict[str, Any]] = []

        # Telegram setup
        self.telegram_token = cfg.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = cfg.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)

        # ----------------------------------------------------------------------
        # LEVEL 1 ADAPTIVE LEARNING MEMORY
        # ----------------------------------------------------------------------
        self.pattern_memory = {}              # pattern_sig -> list of pnls
        self.symbol_performance = {t: [] for t in self.tickers}   # ticker -> pnls
        self.time_of_day_bias = {}            # hour -> pnls
        self.volatility_memory = []           # (vol, pnl)
        self.dynamic_threshold = 0.50         # adjusted by performance

    # ----------------------------------------------------------------------
    # Helper: Convert candles to tensor for the AI model
    # ----------------------------------------------------------------------
    def _prepare_tensor(self, df: pd.DataFrame, window: int = 60) -> torch.Tensor:
        if df is None or len(df) < 10:
            return None

        df = df.iloc[-window:] if len(df) > window else df
        arr = df[["open", "high", "low", "close", "volume"]].values.astype("float32")

        ref = arr[-1, 3] if arr[-1, 3] != 0 else 1.0
        arr[:, :4] /= ref
        vol_ref = max(arr[:, 4].mean(), 1.0)
        arr[:, 4] /= vol_ref

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
        return x.to(self.device)

    # ----------------------------------------------------------------------
    # Helper: Pattern Signature for Adaptive Learning
    # ----------------------------------------------------------------------
    def _pattern_signature(self, df: pd.DataFrame) -> str:
        """
        Converts last 3 candles into a compact pattern signature:
        Example: "B-|BW|R-"
        """
        try:
            last = df.iloc[-3:]
            sigs = []
            for _, row in last.iterrows():
                body = row["close"] - row["open"]
                wick = (row["high"] - row["low"]) / max(row["close"], 1e-6)
                sig = ("B" if body > 0 else "R") + ("W" if wick > 0.02 else "-")
                sigs.append(sig)
            return "|".join(sigs)
        except:
            return "UNK"

    # ----------------------------------------------------------------------
    # Helper: AI Position Sizing
    # ----------------------------------------------------------------------
    def _ai_position_size(self, symbol: str, equity: float, probs: np.ndarray, df: pd.DataFrame) -> int:
        conf = float(probs.max())
        base_risk_frac = 0.01 + 0.03 * (conf - 0.5)
        base_risk_frac = max(0.005, min(base_risk_frac, 0.05))

        closes = df["close"].values
        if len(closes) < 15:
            vol = 0.01
        else:
            rets = np.diff(closes[-15:]) / closes[-15:-1]
            vol = float(np.std(rets)) or 0.01

        vol_adj = 1.0 / (1.0 + 20.0 * vol)
        risk_frac = base_risk_frac * vol_adj

        assumed_stop = 0.015
        price = closes[-1]
        dollar_risk = equity * risk_frac
        qty = dollar_risk / (price * assumed_stop)
        return max(math.floor(qty), 0)

    # ----------------------------------------------------------------------
    # Helper: Exit Logic
    # ----------------------------------------------------------------------
    def _should_exit_position(self, symbol: str, side: str, probs: np.ndarray, df: pd.DataFrame) -> bool:
        meta = self.position_meta.get(symbol)
        now = datetime.now(timezone.utc)

        if meta:
            hours = (now - meta["entry_time"]).total_seconds() / 3600.0
            if hours > 36:
                return True

        buy_p, _, sell_p = probs

        if side == "long" and sell_p > 0.55:
            return True
        if side == "short" and buy_p > 0.55:
            return True

        closes = df["close"].values
        if len(closes) >= 3:
            ret = (closes[-1] - closes[-2]) / closes[-2]
            if side == "long" and ret < -0.01:
                return True
            if side == "short" and ret > 0.01:
                return True

        return False

    # ----------------------------------------------------------------------
    # Pattern Bias (existing logic)
    # ----------------------------------------------------------------------
    def _pattern_bias(self, df: pd.DataFrame) -> float:
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values

        if len(closes) < 5:
            return 0.0

        body = closes[-1] - opens[-1]
        upper = highs[-1] - max(closes[-1], opens[-1])
        lower = min(closes[-1], opens[-1]) - lows[-1]

        ref = max(closes[-1], 1e-6)
        body_n = body / ref
        uw_n = upper / ref
        lw_n = abs(lower) / ref

        xs = np.arange(min(len(closes), 10))
        try:
            slope = float(np.polyfit(xs, closes[-len(xs):], 1)[0]) / ref
        except:
            slope = 0.0

        bias = 0.0
        if body_n > 0 and lw_n > uw_n * 1.3 and slope > 0:
            bias += 0.15
        if body_n < 0 and uw_n > lw_n * 1.3 and slope < 0:
            bias -= 0.15

        if slope > 0.001:
            bias += 0.1
        if slope < -0.001:
            bias -= 0.1

        return max(-0.3, min(0.3, bias))
    # ----------------------------------------------------------------------
    # Main Live Step (Adaptive Learning Integrated)
    # ----------------------------------------------------------------------
    async def live_step(self) -> str:
        """
        One live evaluation step across all tickers:
        - Fetch bars
        - Predict with AI
        - Apply adaptive learning biases
        - Score tickers
        - Select best opportunity
        - Manage existing positions
        """
        now = datetime.now(timezone.utc)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        equity = self.alpaca.get_account_equity()

        ticker_logs: List[str] = []
        best_symbol = None
        best_score = 0.0
        best_side = "hold"
        best_probs = None
        best_df = None

        # ------------------------------------------------------------------
        # Evaluate each ticker
        # ------------------------------------------------------------------
        for symbol in self.tickers:
            df = self.alpaca.get_recent_bars(symbol, timeframe="1Min", limit=120)
            if df is None or len(df) < 20:
                ticker_logs.append(f"{symbol}: insufficient data")
                continue

            x = self._prepare_tensor(df)
            if x is None:
                ticker_logs.append(f"{symbol}: not enough candles")
                continue

            # AI model prediction
            with torch.no_grad():
                logits = self.model(x)
                probs_t = F.softmax(logits, dim=-1)[0]
                probs = probs_t.detach().cpu().numpy()

            buy_p, hold_p, sell_p = probs

            # --------------------------------------
            # Pattern bias (existing logic)
            # --------------------------------------
            bias = self._pattern_bias(df)

            buy_p_adj = buy_p + max(0.0, bias)
            sell_p_adj = sell_p + max(0.0, -bias)
            hold_p_adj = hold_p

            # Normalize
            total_adj = buy_p_adj + hold_p_adj + sell_p_adj
            if total_adj > 0:
                buy_p_adj /= total_adj
                hold_p_adj /= total_adj
                sell_p_adj /= total_adj

            probs_adj = np.array([buy_p_adj, hold_p_adj, sell_p_adj])

            # --------------------------------------
            # ADAPTIVE LEARNING: Dynamic Threshold
            # --------------------------------------
            max_p = float(probs_adj.max())
            if max_p < self.dynamic_threshold:
                side = "hold"
            elif buy_p_adj > sell_p_adj:
                side = "long"
            else:
                side = "short"

            # --------------------------------------
            # Volatility Score
            # --------------------------------------
            closes = df["close"].values
            if len(closes) > 10:
                rets = np.diff(closes[-10:]) / closes[-10:-1]
                vol = float(np.std(rets)) or 0.01
            else:
                vol = 0.01

            vol_score = 1.0 + min(vol * 100, 1.0)

            # --------------------------------------
            # Base pattern bonus
            # --------------------------------------
            pattern = self._pattern_signature(df)
            pattern_bonus = 1.0 + abs(bias)

            # --------------------------------------
            # ADAPTIVE LEARNING: Pattern Memory Boost
            # --------------------------------------
            if pattern in self.pattern_memory and len(self.pattern_memory[pattern]) >= 2:
                avg_pattern_pnl = sum(self.pattern_memory[pattern]) / len(self.pattern_memory[pattern])
                if avg_pattern_pnl > 0:
                    pattern_bonus *= 1.10
                else:
                    pattern_bonus *= 0.90

            # --------------------------------------
            # ADAPTIVE LEARNING: Symbol Performance
            # --------------------------------------
            perf_list = self.symbol_performance.get(symbol, [])
            if len(perf_list) >= 3:
                avg_perf = sum(perf_list[-3:]) / 3
                if avg_perf > 0:
                    pattern_bonus *= 1.05
                else:
                    pattern_bonus *= 0.95

            # --------------------------------------
            # ADAPTIVE LEARNING: Time-of-Day Bias
            # --------------------------------------
            hour = now.hour
            if hour in self.time_of_day_bias and len(self.time_of_day_bias[hour]) >= 2:
                avg_hour_pnl = sum(self.time_of_day_bias[hour]) / len(self.time_of_day_bias[hour])
                if avg_hour_pnl > 0:
                    pattern_bonus *= 1.03
                else:
                    pattern_bonus *= 0.97

            # --------------------------------------
            # ADAPTIVE LEARNING: Volatility Regime Learning
            # --------------------------------------
            if len(self.volatility_memory) >= 5:
                close_vol_matches = [
                    p for v, p in self.volatility_memory[-20:]
                    if abs(v - vol) < 0.005
                ]

                if close_vol_matches:
                    avg_vol_pnl = sum(close_vol_matches) / len(close_vol_matches)
                    if avg_vol_pnl > 0:
                        pattern_bonus *= 1.04
                    else:
                        pattern_bonus *= 0.96

            # --------------------------------------
            # Final Score Calculation
            # --------------------------------------
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
                best_vol = vol
                best_pattern = pattern
        # ------------------------------------------------------------------
        # Manage Existing Positions (includes adaptive updates on EXIT)
        # ------------------------------------------------------------------
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
                    probs_exit = probs_t.detach().cpu().numpy()

                # --------------------- Exit decision ---------------------
                if self._should_exit_position(symbol, side, probs_exit, df):
                    closes = df["close"].values
                    exit_price = float(closes[-1])
                    entry_price = float(meta.get("entry_price", exit_price))
                    trade_qty = float(meta.get("qty", qty))

                    # Realized PnL
                    if side == "long":
                        pnl = (exit_price - entry_price) * trade_qty
                    else:
                        pnl = (entry_price - exit_price) * trade_qty

                    # Close position
                    self.alpaca.close_position(symbol)
                    self.position_meta.pop(symbol, None)

                    # Log trade
                    self._log_trade(
                        kind="EXIT",
                        symbol=symbol,
                        side=side,
                        qty=trade_qty,
                        score=None,
                        probs=probs_exit,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                    )

                    # --------------------- ADAPTIVE LEARNING UPDATES ---------------------

                    # Symbol-level performance
                    self.symbol_performance.setdefault(symbol, []).append(pnl)

                    # Time-of-Day performance
                    hour = now.hour
                    self.time_of_day_bias.setdefault(hour, []).append(pnl)

                    # Volatility learning
                    if "best_vol" in locals():
                        self.volatility_memory.append((best_vol, pnl))

                    # Pattern memory
                    if "best_pattern" in locals():
                        self.pattern_memory.setdefault(best_pattern, []).append(pnl)

                    # Dynamic threshold tuning
                    if pnl > 0:
                        self.dynamic_threshold = max(0.45, self.dynamic_threshold - 0.01)
                    else:
                        self.dynamic_threshold = min(0.60, self.dynamic_threshold + 0.01)

                    # Add exit message
                    action_log_lines.append(
                        f"EXIT {symbol} {side.upper()} x{int(trade_qty)} "
                        f"@ {exit_price:.2f} PnL={pnl:.2f} (Adaptive learning updated)"
                    )

        # ------------------------------------------------------------------
        # Enter New Position (if best score found)
        # ------------------------------------------------------------------
        if best_symbol is not None and best_probs is not None and best_df is not None:
            qty, _ = self.alpaca.get_open_position(best_symbol)

            # Only enter if not already holding the symbol
            if qty == 0.0:
                size = self._ai_position_size(best_symbol, equity, best_probs, best_df)

                if size > 0:
                    closes = best_df["close"].values
                    entry_price = float(closes[-1])
                    side_str = "buy" if best_side == "long" else "sell"

                    # Execute market order
                    order = self.alpaca.submit_market_order(best_symbol, size, side_str)

                    # Save meta
                    self.position_meta[best_symbol] = {
                        "entry_time": datetime.now(timezone.utc),
                        "side": best_side,
                        "entry_price": entry_price,
                        "qty": float(size),
                    }

                    # Log entry
                    self._log_trade(
                        kind="ENTER",
                        symbol=best_symbol,
                        side=best_side,
                        qty=float(size),
                        score=best_score,
                        probs=best_probs,
                        entry_price=entry_price,
                    )

                    # Add entry log
                    action_log_lines.append(
                        f"ENTER {best_symbol} {side_str.upper()} x{size} @ {entry_price:.2f} "
                        f"- AI hybrid decision (score={best_score:.2f})"
                    )

        # ------------------------------------------------------------------
        # Build Terminal Log Output
        # ------------------------------------------------------------------
        header = f"[{now_str}] EQUITY={equity:.2f}"
        tickers_block = "\n".join(ticker_logs)
        actions_block = "\n".join(action_log_lines) if action_log_lines else "No new orders this cycle."
        separator = "-" * 40

        log = f"{header}\n{tickers_block}\n{actions_block}\n{separator}"
        return log
