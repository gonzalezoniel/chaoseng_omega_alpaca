import json
import math
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chaoseng.model import OmegaModel
from chaoseng.alpaca_client import AlpacaClient


class ChaosEngineOmegaHybrid:
    '''
    Ultra-adaptive hybrid engine:
    - Multi-ticker
    - Multi-timeframe context (simplified but extendable)
    - AI-driven long/short decisions
    - AI-driven position sizing (S3)
    - 1-2 day max hold horizon logic
    - Live Alpaca execution
    - Terminal-style log output
    '''

    def __init__(self, config_path: str = "config.json"):
        self.device = torch.device("cpu")
        self.model = OmegaModel().to(self.device)
        self.model.eval()  # inference mode by default

        # Load config
        if not os.path.exists(config_path):
            # Create a sample config if not exists
            sample = {
                "ALPACA_API_KEY": "YOUR_KEY_HERE",
                "ALPACA_SECRET_KEY": "YOUR_SECRET_HERE",
                "ALPACA_PAPER": True,
                "TICKERS": ["SPY","QQQ","AAPL","TSLA","NVDA","AMD","AMZN","META","MSFT","NFLX"]
            }
            with open(config_path, "w") as f:
                json.dump(sample, f, indent=2)
            raise FileNotFoundError("config.json created. Fill in your Alpaca keys and restart.")

        with open(config_path) as f:
            self.cfg = json.load(f)

        self.tickers: List[str] = self.cfg.get("TICKERS", ["SPY","QQQ","AAPL","TSLA","NVDA","AMD","AMZN","META","MSFT","NFLX"])

        # Alpaca client
        self.alpaca = AlpacaClient(config_path=config_path)

        # Internal state
        self.last_decisions: Dict[str, Dict] = {}
        self.position_meta: Dict[str, Dict] = {}  # symbol -> {entry_time, entry_price, side}
        self.start_equity = self.alpaca.get_account_equity()

    def _prepare_tensor(self, df: pd.DataFrame, window: int = 60) -> torch.Tensor:
        if len(df) < 10:
            # too little data
            return None
        if len(df) < window:
            df = df.iloc[-window:]
        else:
            df = df.iloc[-window:]
        arr = df[["open","high","low","close","volume"]].values.astype("float32")
        # Normalize roughly by last close
        ref = arr[-1, 3] if arr[-1, 3] != 0 else 1.0
        arr[:, :4] /= ref
        arr[:, 4] /= max(arr[:, 4].mean(), 1.0)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # [seq, batch=1, feat]
        return x.to(self.device)

    def _ai_position_size(self, symbol: str, equity: float, probs: np.ndarray, df: pd.DataFrame) -> float:
        '''
        S3: AI-based position sizing.
        Uses:
        - confidence (max prob)
        - volatility (ATR-like)
        - rough risk fraction of equity
        '''
        conf = float(probs.max())
        base_risk_frac = 0.01 + 0.03 * (conf - 0.5)  # between ~1% and 4% as confidence rises
        base_risk_frac = max(0.005, min(base_risk_frac, 0.05))  # clamp 0.5% - 5%

        closes = df["close"].values
        if len(closes) < 15:
            vol = 1.0
        else:
            rets = np.diff(closes[-15:]) / closes[-15:-1]
            vol = float(np.std(rets)) if np.std(rets) > 0 else 0.01

        # Higher vol -> smaller size
        vol_adj = 1.0 / (1.0 + 20.0 * vol)
        risk_frac = base_risk_frac * vol_adj

        # Assume stop ~1.5% away on average
        assumed_stop = 0.015
        dollar_risk = equity * risk_frac
        if dollar_risk <= 0:
            return 0.0
        price = closes[-1]
        qty = dollar_risk / (price * assumed_stop)
        qty = math.floor(qty)
        return max(qty, 0)

    def _should_exit_position(self, symbol: str, side: str, probs: np.ndarray, df: pd.DataFrame) -> bool:
        '''
        1-2 day horizon exit logic plus safety.
        '''
        meta = self.position_meta.get(symbol)
        now = datetime.now(timezone.utc)
        if meta:
            entry_time = meta["entry_time"]
            held_hours = (now - entry_time).total_seconds() / 3600.0
            # Hard max: about 36 hours
            if held_hours > 36:
                return True

        # Exit if opposite side confidence strong
        buy_p, hold_p, sell_p = probs
        if side == "long" and sell_p > 0.55:
            return True
        if side == "short" and buy_p > 0.55:
            return True

        closes = df["close"].values
        if len(closes) >= 3:
            c1, c2 = closes[-2], closes[-1]
            ret = (c2 - c1) / c1
            # If moving aggressively against us
            if side == "long" and ret < -0.01:  # -1 percent bar
                return True
            if side == "short" and ret > 0.01:
                return True

        return False

    async def live_step(self) -> str:
        '''
        One live evaluation step across tickers:
        - Pull recent bars for each ticker
        - Score with Omega model
        - Decide best action
        - Execute via Alpaca (paper or live depending on config)
        - Return terminal-style log string
        '''
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        equity = self.alpaca.get_account_equity()

        ticker_logs: List[str] = []
        best_symbol = None
        best_score = 0.0
        best_side = "hold"
        best_probs = None
        best_df = None

        # Evaluate each ticker
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
            # Ultra adaptive: side preference by relative strength
            if max(probs) < 0.5:
                side = "hold"
            elif buy_p > sell_p:
                side = "long"
            else:
                side = "short"

            # Score with simple heuristic: (max prob) * volatility factor
            closes = df["close"].values
            if len(closes) > 10:
                rets = np.diff(closes[-10:]) / closes[-10:-1]
                vol = float(np.std(rets))
            else:
                vol = 0.01
            vol_score = 1.0 + min(vol * 100, 1.0)  # up to +1 multiplier
            score = float(max(probs) * vol_score)

            ticker_logs.append(
                f"{symbol}: BUY={buy_p:.2f} HOLD={hold_p:.2f} SELL={sell_p:.2f} SIDE={side} SCORE={score:.2f}"
            )

            if score > best_score and side != "hold":
                best_score = score
                best_symbol = symbol
                best_side = side
                best_probs = probs
                best_df = df

        action_log_lines: List[str] = []
        # Handle positions per symbol
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
                    self.alpaca.close_position(symbol)
                    self.position_meta.pop(symbol, None)
                    action_log_lines.append(f"EXIT {symbol} {side.upper()} - securing profit / cutting risk.")

        # Enter new position if best_symbol exists and no open position in it
        if best_symbol is not None:
            qty, _ = self.alpaca.get_open_position(best_symbol)
            if qty == 0.0 and best_probs is not None and best_df is not None:
                # AI-based sizing
                size = self._ai_position_size(best_symbol, equity, best_probs, best_df)
                if size > 0:
                    side = "buy" if best_side == "long" else "sell"
                    order = self.alpaca.submit_market_order(best_symbol, size, side)
                    self.position_meta[best_symbol] = {
                        "entry_time": datetime.now(timezone.utc),
                        "side": "long" if side == "buy" else "short"
                    }
                    action_log_lines.append(
                        f"ENTER {best_symbol} {side.upper()} x{size} - AI hybrid decision (score={best_score:.2f})."
                    )

        # Build terminal-style log
        header = f"[{now}] EQUITY={equity:.2f}"
        tickers_block = "\n".join(ticker_logs)
        actions_block = "\n".join(action_log_lines) if action_log_lines else "No new orders this cycle."
        separator = "-" * 40
        log = f"{header}\n{tickers_block}\n{actions_block}\n{separator}"
        return log
