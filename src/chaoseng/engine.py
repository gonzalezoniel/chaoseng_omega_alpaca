# chaoseng/engine.py

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chaoseng.model import OmegaModel
from chaoseng.alpaca_client import AlpacaClient


# -------------------------------
# Config dataclass
# -------------------------------

@dataclass
class DayTraderConfig:
    symbols: List[str]

    # Risk
    base_risk_per_trade: float = 0.003    # 0.3% of equity per long trade
    short_risk_factor: float = 0.6        # shorts risk = base_risk * factor
    max_open_trades: int = 3

    # Time rules (US Eastern)
    morning_start: time = time(9, 34)
    morning_end: time = time(11, 0)
    afternoon_start: time = time(13, 30)
    afternoon_end: time = time(15, 45)
    flatten_all_at: time = time(15, 55)   # force close everything

    # Holding time
    max_hold_minutes: int = 60

    # Shorting
    enable_shorting: bool = True

    # Data / features
    lookback_bars: int = 50


# -------------------------------
# Helper: market regime
# -------------------------------

class MarketRegime:
    UP_TREND = "UP_TREND"
    DOWN_TREND = "DOWN_TREND"
    RANGE = "RANGE"
    CHOP = "CHOP"
    LOW_VOL = "LOW_VOL"
    UNKNOWN = "UNKNOWN"


# -------------------------------
# Core Engine
# -------------------------------

class ChaosEngineOmegaHybrid:
    """
    Day-Trader ChaosEngine (REPLACES previous engine):

    - Intraday only (no overnight holds)
    - Restricted trading windows (AM + PM session)
    - Max hold duration per position
    - AI-driven long/short decisions
    - Shorting enabled with safety rules
    - Market regime detection
    - Pattern confidence scoring
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        model: Optional[OmegaModel] = None,
        config: Optional[DayTraderConfig] = None,
        tzinfo: timezone = timezone.utc,
    ):
        """
        Make this compatible with your existing main.py which calls
        ChaosEngineOmegaHybrid() with no arguments.

        - If nothing is passed, we build AlpacaClient, OmegaModel, and a default
          DayTraderConfig internally.
        - If you later want to construct it manually, you can still do:
            ChaosEngineOmegaHybrid(alpaca_client=..., model=..., config=...)
        """
        if config is None:
            # You can edit this symbol list to whatever you're actually trading
            config = DayTraderConfig(symbols=["SPY", "QQQ", "TSLA", "NVDA"])

        if alpaca_client is None:
            alpaca_client = AlpacaClient()  # uses your existing implementation

        if model is None:
            model = OmegaModel()  # uses your existing default/weights loading

        self.alpaca = alpaca_client
        self.model = model
        self.cfg = config
        self.tzinfo = tzinfo


    # ---------------------------
    # Public entrypoint
    # ---------------------------

    def run_cycle(self) -> None:
        """
        One full decision cycle:
        - Get positions
        - Intraday housekeeping (flatten near close, max hold)
        - If within trading hours: evaluate signals and trade
        """
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(self.tzinfo)

        # Fetch current positions once
        positions = self.alpaca.get_open_positions()  # expected: dict[symbol] -> info

        # Housekeeping: flatten near close + max hold check
        self._intraday_housekeeping(now_local, positions)

        # If market closed or outside trading window: stop here
        if not self._is_market_open() or not self._within_trading_hours(now_local):
            return

        # Pull latest market data
        bars_by_symbol = self._get_recent_bars()

        # Detect regime (per symbol or combined)
        regimes = {
            symbol: self._detect_market_regime(df)
            for symbol, df in bars_by_symbol.items()
            if not df.empty
        }

        # Evaluate and trade per symbol
        equity = self.alpaca.get_equity()  # account equity float

        for symbol, df in bars_by_symbol.items():
            if df.empty:
                continue

            regime = regimes.get(symbol, MarketRegime.UNKNOWN)
            pattern_score, direction_hint = self._pattern_signal(df, regime)

            # Evaluate model (if you want AI + patterns hybrid)
            model_dir, model_conf = self._model_signal(df)

            decision = self._combine_signals(
                regime=regime,
                pattern_score=pattern_score,
                pattern_dir=direction_hint,
                model_dir=model_dir,
                model_conf=model_conf,
            )

            # Execute decision with risk rules
            self._execute_decision(
                symbol=symbol,
                decision=decision,
                df=df,
                equity=equity,
                positions=positions,
            )

    # ---------------------------
    # Time & schedule logic
    # ---------------------------

    def _is_market_open(self) -> bool:
        try:
            clock = self.alpaca.get_clock()
            return bool(getattr(clock, "is_open", False))
        except Exception:
            return False

    def _within_trading_hours(self, now_local: datetime) -> bool:
        t = now_local.time()

        in_morning = self.cfg.morning_start <= t <= self.cfg.morning_end
        in_afternoon = self.cfg.afternoon_start <= t <= self.cfg.afternoon_end
        return in_morning or in_afternoon

    def _intraday_housekeeping(self, now_local: datetime, positions: Dict[str, Any]) -> None:
        """
        - Flatten everything at configured 'flatten_all_at' time.
        - Enforce max holding time per position.
        """
        # 1) Hard flatten time
        if now_local.time() >= self.cfg.flatten_all_at:
            if positions:
                for symbol in list(positions.keys()):
                    self._close_position(symbol, reason="End of day flatten")
            return

        # 2) Max holding time enforcement
        max_delta = timedelta(minutes=self.cfg.max_hold_minutes)
        for symbol, pos in positions.items():
            entry_time = self._extract_entry_time(pos)
            if entry_time is None:
                continue
            hold_time = now_local - entry_time.astimezone(self.tzinfo)
            if hold_time >= max_delta:
                self._close_position(symbol, reason="Max hold time reached")

    def _extract_entry_time(self, pos: Any) -> Optional[datetime]:
        """
        Extract entry time from position object/dict.
        Adjust according to how AlpacaClient returns it.
        """
        t = getattr(pos, "entry_time", None) or getattr(pos, "filled_at", None)
        if isinstance(t, datetime):
            return t
        # or parse ISO-8601 string if needed
        if isinstance(t, str):
            try:
                return datetime.fromisoformat(t.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    # ---------------------------
    # Data fetch
    # ---------------------------

    def _get_recent_bars(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for symbol in self.cfg.symbols:
            try:
                df = self.alpaca.get_recent_bars(
                    symbol,
                    limit=self.cfg.lookback_bars,
                )
                # Expect df with columns: ["time", "open", "high", "low", "close", "volume"]
                if not isinstance(df, pd.DataFrame) or df.empty:
                    df = pd.DataFrame()
                data[symbol] = df
            except Exception:
                data[symbol] = pd.DataFrame()
        return data

    # ---------------------------
    # Market regime detection
    # ---------------------------

    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 10:
            return MarketRegime.UNKNOWN

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        window = min(len(closes), 40)
        sub = closes[-window:]
        x = np.arange(window)
        # simple slope of price vs bar index
        slope = np.polyfit(x, sub, 1)[0]

        # volatility via high/low range
        rng = (highs[-window:] - lows[-window:]).mean()
        price = closes[-1]
        vol_ratio = rng / price if price > 0 else 0.0

        # crude classification
        if vol_ratio < 0.001:
            return MarketRegime.LOW_VOL

        # thresholds you can tune
        if slope > 0 and abs(slope) > price * 0.0003:
            return MarketRegime.UP_TREND
        if slope < 0 and abs(slope) > price * 0.0003:
            return MarketRegime.DOWN_TREND

        # Range / chop
        # If last N bars are oscillating around mean
        last = sub
        mean = last.mean()
        dist = np.abs(last - mean)
        mean_dist = dist.mean()
        # If range small -> RANGE, if noisy -> CHOP
        if mean_dist < price * 0.002:
            return MarketRegime.RANGE
        else:
            return MarketRegime.CHOP

    # ---------------------------
    # Pattern signal
    # ---------------------------

    def _pattern_signal(self, df: pd.DataFrame, regime: str) -> Tuple[float, Optional[str]]:
        """
        Returns:
          pattern_score: 0–1 float
          direction_hint: "LONG" | "SHORT" | None
        """
        if df.empty or len(df) < 5:
            return 0.0, None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        o, h, l, c = last["open"], last["high"], last["low"], last["close"]
        body = abs(c - o)
        range_ = max(h - l, 1e-6)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l

        score = 0.0
        direction: Optional[str] = None

        # Hammer (long lower wick) in uptrend / range → bullish
        if lower_wick > 0.6 * range_ and body < 0.3 * range_:
            if regime in (MarketRegime.UP_TREND, MarketRegime.RANGE):
                score += 0.4
                direction = "LONG"

        # Shooting star (long upper wick) in downtrend / range → bearish
        if upper_wick > 0.6 * range_ and body < 0.3 * range_:
            if regime in (MarketRegime.DOWN_TREND, MarketRegime.RANGE):
                score += 0.4
                direction = "SHORT"

        # Bullish engulfing
        if (
            c > o
            and prev["close"] < prev["open"]
            and c >= prev["open"]
            and o <= prev["close"]
        ):
            if regime == MarketRegime.UP_TREND:
                score += 0.3
                direction = "LONG"

        # Bearish engulfing
        if (
            c < o
            and prev["close"] > prev["open"]
            and c <= prev["open"]
            and o >= prev["close"]
        ):
            if regime == MarketRegime.DOWN_TREND:
                score += 0.3
                direction = "SHORT"

        # Normalize score
        score = max(0.0, min(1.0, score))
        return score, direction

    # ---------------------------
    # Model signal (AI component)
    # ---------------------------

    def _model_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], float]:
        """
        Use your OmegaModel to get direction + confidence.
        Adjust this to match how your model expects features.
        """
        if df.empty:
            return None, 0.0

        try:
            features = self._build_features(df)
            logits = self.model(features)  # shape [1, 3] for [short, flat, long] for example
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            short_p, flat_p, long_p = probs.tolist()
            max_p = max(probs)
            if max_p < 0.4:
                return None, max_p

            if long_p == max_p:
                return "LONG", max_p
            if short_p == max_p:
                return "SHORT", max_p
            return None, max_p
        except Exception:
            return None, 0.0

    def _build_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Simple feature builder. Replace/extend with your actual one.
        """
        closes = df["close"].values.astype(np.float32)
        returns = np.diff(closes) / closes[:-1]
        pad_len = max(0, 50 - len(returns))
        if pad_len > 0:
            returns = np.pad(returns, (pad_len, 0), mode="constant")
        else:
            returns = returns[-50:]

        x = torch.from_numpy(returns).unsqueeze(0)  # shape [1, 50]
        return x

    # ---------------------------
    # Decision fusion
    # ---------------------------

    def _combine_signals(
        self,
        regime: str,
        pattern_score: float,
        pattern_dir: Optional[str],
        model_dir: Optional[str],
        model_conf: float,
    ) -> Dict[str, Any]:
        """
        Combines pattern + model into a unified decision dict.
        """
        # If regime is low-vol or chop, be very conservative
        if regime in (MarketRegime.LOW_VOL, MarketRegime.CHOP):
            if pattern_score < 0.7 or (model_conf < 0.55):
                return {"action": "HOLD"}

        # Base decision from pattern
        decision_dir = pattern_dir
        confidence = pattern_score

        # If model agrees with pattern, boost confidence
        if model_dir is not None and pattern_dir is not None and model_dir == pattern_dir:
            confidence = min(1.0, pattern_score + 0.25 * model_conf)

        # If model contradicts strongly, neutralize
        if model_dir is not None and pattern_dir is not None and model_dir != pattern_dir:
            if model_conf > 0.6:
                return {"action": "HOLD"}

        if confidence < 0.5 or decision_dir is None:
            return {"action": "HOLD"}

        return {"action": "ENTER", "direction": decision_dir, "confidence": confidence}

    # ---------------------------
    # Execution & risk mgmt
    # ---------------------------

    def _execute_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        df: pd.DataFrame,
        equity: float,
        positions: Dict[str, Any],
    ) -> None:
        action = decision.get("action", "HOLD")

        if action == "HOLD":
            return

        # Don't exceed max open trades
        if len(positions) >= self.cfg.max_open_trades:
            return

        direction = decision.get("direction")
        confidence = decision.get("confidence", 0.0)

        # If direction is SHORT but shorting disabled or conditions unsafe, skip
        if direction == "SHORT" and not self.cfg.enable_shorting:
            return

        price = float(df["close"].iloc[-1])

        # Determine stop distance based on volatility
        atr = self._estimate_atr(df)
        if atr is None or atr <= 0:
            stop_dist = price * 0.004  # fallback ~0.4%
        else:
            stop_dist = max(price * 0.002, atr * 0.6)

        # Risk per trade
        risk_fraction = self.cfg.base_risk_per_trade
        if direction == "SHORT":
            risk_fraction *= self.cfg.short_risk_factor

        risk_dollars = equity * risk_fraction

        qty = max(1, int(risk_dollars / stop_dist))

        # Safety: don't send absurd size
        if qty <= 0:
            return

        # Entry + stop/target
        if direction == "LONG":
            stop_price = price - stop_dist
            # simple reward: 1.5R
            take_profit = price + stop_dist * 1.5
            self.alpaca.submit_bracket_order(
                symbol=symbol,
                side="buy",
                qty=qty,
                entry_price=None,  # market
                stop_price=stop_price,
                take_profit_price=take_profit,
            )
        elif direction == "SHORT":
            # extra guard: don't short in strong uptrend regime
            regime = self._detect_market_regime(df)
            if regime == MarketRegime.UP_TREND:
                return

            stop_price = price + stop_dist
            take_profit = price - stop_dist * 1.5
            self.alpaca.submit_bracket_order(
                symbol=symbol,
                side="sell",
                qty=qty,
                entry_price=None,
                stop_price=stop_price,
                take_profit_price=take_profit,
            )

    def _estimate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        if len(df) < period + 1:
            return None
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        trs = []
        for i in range(1, period + 1):
            high = highs[-i]
            low = lows[-i]
            prev_close = closes[-i - 1]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)
        return float(np.mean(trs))

    def _close_position(self, symbol: str, reason: str = "") -> None:
        try:
            self.alpaca.close_position(symbol)
        except Exception:
            pass
