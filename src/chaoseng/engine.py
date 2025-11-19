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
    Omega AI Hybrid Engine + Level-1 Adaptive Learning
    Autonomous, pattern-aware, volatility-aware, symbol-memory AI.
    """

    def __init__(self, config_path: str = "config.json"):
        self.device = torch.device("cpu")
        self.model = OmegaModel().to(self.device)
        self.model.eval()

        # Load config
        cfg = {}
        if os.path.exists(config_path):
            try:
                cfg = json.load(open(config_path))
            except:
                cfg = {}
        self.cfg = cfg

        # Alpaca client
        self.alpaca = AlpacaClient(config_path=config_path)
        self.tickers = self.alpaca.tickers

        # Runtime state
        self.position_meta: Dict[str, Dict[str, Any]] = {}
        self.start_equity = self.alpaca.get_account_equity()
        self.trade_log: List[Dict[str, Any]] = []

        # Telegram
        self.telegram_token = cfg.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = cfg.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)

        # ----------------------------------------------------------------------
        # LEVEL 1 ADAPTIVE LEARNING MEMORY
        # ----------------------------------------------------------------------
        self.pattern_memory = {}                  # pattern_sig -> PnLs
        self.symbol_performance = {t: [] for t in self.tickers}
        self.time_of_day_bias = {}                # hour -> PnLs
        self.volatility_memory = []               # (vol, pnl)
        self.dynamic_threshold = 0.50             # AI aggressiveness threshold

    # ==========================================================================
    # ----------------------  DATA PREP / HELPERS  -----------------------------
    # ==========================================================================

    def _prepare_tensor(self, df: pd.DataFrame, window: int = 60) -> torch.Tensor:
        if df is None or len(df) < 10:
            return None
        if len(df) > window:
            df = df.iloc[-window:]

        arr = df[["open", "high", "low", "close", "volume"]].values.astype("float32")
        ref = arr[-1, 3] or 1.0
        arr[:, :4] /= ref
        vol_ref = max(arr[:, 4].mean(), 1.0)
        arr[:, 4] /= vol_ref

        return torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(self.device)

    def _pattern_signature(self, df: pd.DataFrame) -> str:
        """3-candle signature for pattern memory."""
        try:
            last = df.iloc[-3:]
            sigs = []
            for _, row in last.iterrows():
                body = row["close"] - row["open"]
                wick_ratio = (row["high"] - row["low"]) / max(row["close"], 1e-6)
                color = "B" if body > 0 else "R"
                wick = "W" if wick_ratio > 0.02 else "_"
                sigs.append(color + wick)
            return "|".join(sigs)
        except:
            return "UNK"

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

        vol_adj = 1 / (1 + 20 * vol)
        risk_frac = base_risk_frac * vol_adj

        price = closes[-1]
        assumed_stop = 0.015
        dollar_risk = equity * risk_frac
        qty = dollar_risk / (price * assumed_stop)

        return max(int(qty), 0)

    def _should_exit_position(self, symbol: str, side: str, probs: np.ndarray, df: pd.DataFrame) -> bool:
        meta = self.position_meta.get(symbol)
        now = datetime.now(timezone.utc)

        if meta:
            hrs = (now - meta["entry_time"]).total_seconds() / 3600
            if hrs > 36:
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
        ref = closes[-1] or 1e-6

        body_n = body / ref
        uw = upper / ref
        lw = abs(lower) / ref

        xs = np.arange(min(len(closes), 10))
        try:
            slope = float(np.polyfit(xs, closes[-len(xs):], 1)[0]) / ref
        except:
            slope = 0

        bias = 0.0
        if body_n > 0 and lw > uw * 1.3 and slope > 0:
            bias += 0.15
        if body_n < 0 and uw > lw * 1.3 and slope < 0:
            bias -= 0.15

        if slope > 0.001:
            bias += 0.10
        if slope < -0.001:
            bias -= 0.10

        return max(-0.3, min(0.3, bias))

    # ==========================================================================
    # ---------------------------- LOGGING -------------------------------------
    # ==========================================================================

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
        now = datetime.now(timezone.utc).isoformat()
        buy_p = hold_p = sell_p = None

        if probs is not None and len(probs) == 3:
            buy_p, hold_p, sell_p = [float(x) for x in probs]

        self.trade_log.append({
            "timestamp": now,
            "kind": kind,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "score": float(score) if score else None,
            "buy_p": buy_p,
            "hold_p": hold_p,
            "sell_p": sell_p,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
        })

        # Telegram
        try:
            if kind == "ENTER":
                msg = f"ENTER {symbol} {side} x{qty} @ {entry_price:.2f}"
            elif kind == "EXIT":
                msg = f"EXIT {symbol} {side} x{qty} @ {exit_price:.2f} PnL={pnl:.2f}"
            else:
                msg = None

            if msg:
                self._send_telegram(msg)
        except:
            pass

    def _send_telegram(self, text: str):
        if not self.telegram_enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, data={"chat_id": self.telegram_chat_id, "text": text}, timeout=5)
        except:
            pass

    # ==========================================================================
    # ---------------------- PUBLIC API (Dashboard) ----------------------------
    # ==========================================================================

    def get_trade_history(self, limit=200):
        return self.trade_log[-limit:]

    def get_pnl_summary(self):
        exits = [t for t in self.trade_log if t["kind"] == "EXIT" and t["pnl"] is not None]
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
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            "total_realized_pnl": sum(pnls),
            "num_trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_win": sum(wins)/len(wins) if wins else 0,
            "avg_loss": sum(losses)/len(losses) if losses else 0,
            "max_win": max(wins) if wins else 0,
            "max_loss": min(losses) if losses else 0,
        }

    # ==========================================================================
    # ---------------------------- MAIN LIVE STEP ------------------------------
    # ==========================================================================

    async def live_step(self) -> str:
        now = datetime.now(timezone.utc)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        equity = self.alpaca.get_account_equity()

        ticker_logs = []
        best_symbol = None
        best_score = 0
        best_side = "hold"
        best_probs = None
        best_df = None
        best_vol = None
        best_pattern = None

        # ------------------------- Evaluate each ticker -----------------------
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
                probs = F.softmax(self.model(x), dim=-1)[0].cpu().numpy()

            buy_p, hold_p, sell_p = probs

            # Pattern bias
            bias = self._pattern_bias(df)

            buy_adj = buy_p + max(0, bias)
            sell_adj = sell_p + max(0, -bias)
            hold_adj = hold_p
            total = buy_adj + hold_adj + sell_adj
            if total > 0:
                buy_adj /= total
                hold_adj /= total
                sell_adj /= total

            probs_adj = np.array([buy_adj, hold_adj, sell_adj])
            max_p = probs_adj.max()

            # Dynamic threshold decides if attempt is valid
            if max_p < self.dynamic_threshold:
                side = "hold"
            elif buy_adj > sell_adj:
                side = "long"
            else:
                side = "short"

            closes = df["close"].values
            if len(closes) < 10:
                vol = 0.01
            else:
                rets = np.diff(closes[-10:]) / closes[-10:-1]
                vol = float(np.std(rets)) or 0.01

            vol_score = 1 + min(vol*100, 1)

            pattern = self._pattern_signature(df)
            pattern_bonus = 1 + abs(bias)

            # Adaptive learning influences
            # Pattern memory
            if pattern in self.pattern_memory and len(self.pattern_memory[pattern]) >= 2:
                avgp = sum(self.pattern_memory[pattern]) / len(self.pattern_memory[pattern])
                pattern_bonus *= 1.10 if avgp > 0 else 0.90

            # Symbol performance
            if len(self.symbol_performance[symbol]) >= 3:
                avg_perf = sum(self.symbol_performance[symbol][-3:]) / 3
                pattern_bonus *= 1.05 if avg_perf > 0 else 0.95

            # Time-of-day
            hr = now.hour
            if hr in self.time_of_day_bias and len(self.time_of_day_bias[hr]) >= 2:
                avg_hr = sum(self.time_of_day_bias[hr]) / len(self.time_of_day_bias[hr])
                pattern_bonus *= 1.03 if avg_hr > 0 else 0.97

            # Volatility memory
            if len(self.volatility_memory) >= 5:
                close_vol = [p for v, p in self.volatility_memory[-20:] if abs(v - vol) < 0.005]
                if close_vol:
                    avg_volp = sum(close_vol)/len(close_vol)
                    pattern_bonus *= 1.04 if avg_volp > 0 else 0.96

            score = max_p * vol_score * pattern_bonus

            ticker_logs.append(
                f"{symbol}: BUY={buy_adj:.2f} HOLD={hold_adj:.2f} SELL={sell_adj:.2f} "
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

        # ==========================================================================
        # ----------------------------- EXIT LOGIC ---------------------------------
        # ==========================================================================
        action_logs = []

        for symbol in self.tickers:
            qty, _ = self.alpaca.get_open_position(symbol)
            if qty == 0:
                continue

            meta = self.position_meta.get(symbol)
            if not meta:
                continue

            side = meta["side"]

            df = self.alpaca.get_recent_bars(symbol, "1Min", 120)
            x = self._prepare_tensor(df)
            if x is None:
                continue

            with torch.no_grad():
                p = F.softmax(self.model(x), dim=-1)[0].cpu().numpy()

            if self._should_exit_position(symbol, side, p, df):
                closes = df["close"].values
                exit_price = float(closes[-1])
                entry_price = float(meta["entry_price"])
                size = float(meta["qty"])

                if side == "long":
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size

                self.alpaca.close_position(symbol)
                self.position_meta.pop(symbol, None)

                self._log_trade(
                    "EXIT",
                    symbol,
                    side,
                    size,
                    probs=p,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                )

                # Adaptive updates
                self.symbol_performance[symbol].append(pnl)
                hour = now.hour
                self.time_of_day_bias.setdefault(hour, []).append(pnl)
                if best_vol is not None:
                    self.volatility_memory.append((best_vol, pnl))
                if best_pattern is not None:
                    self.pattern_memory.setdefault(best_pattern, []).append(pnl)

                if pnl > 0:
                    self.dynamic_threshold = max(0.45, self.dynamic_threshold - 0.01)
                else:
                    self.dynamic_threshold = min(0.60, self.dynamic_threshold + 0.01)

                action_logs.append(
                    f"EXIT {symbol.upper()} {side.upper()} x{int(size)} @ {exit_price:.2f} "
                    f"PnL={pnl:.2f} (learning updated)"
                )

        # ==========================================================================
        # ----------------------------- ENTRY LOGIC --------------------------------
        # ==========================================================================
        if best_symbol and best_probs is not None and best_df is not None:
            qty, _ = self.alpaca.get_open_position(best_symbol)
            if qty == 0:
                size = self._ai_position_size(best_symbol, equity, best_probs, best_df)

                if size > 0:
                    entry_price = float(best_df["close"].values[-1])
                    side_str = "buy" if best_side == "long" else "sell"

                    self.alpaca.submit_market_order(best_symbol, size, side_str)

                    self.position_meta[best_symbol] = {
                        "entry_time": datetime.now(timezone.utc),
                        "side": best_side,
                        "entry_price": entry_price,
                        "qty": size,
                    }

                    self._log_trade(
                        "ENTER",
                        best_symbol,
                        best_side,
                        size,
                        score=best_score,
                        probs=best_probs,
                        entry_price=entry_price,
                    )

                    action_logs.append(
                        f"ENTER {best_symbol.upper()} {side_str.upper()} x{size} "
                        f"@ {entry_price:.2f} (score={best_score:.2f})"
                    )

        # ==========================================================================
        # ----------------------------- OUTPUT LOG ---------------------------------
        # ==========================================================================
        header = f"[{now_str}] EQUITY={equity:.2f}"
        tick_block = "\n".join(ticker_logs)
        act_block = "\n".join(action_logs) if action_logs else "No new orders this cycle."
        sep = "-"*40

        return f"{header}\n{tick_block}\n{act_block}\n{sep}"