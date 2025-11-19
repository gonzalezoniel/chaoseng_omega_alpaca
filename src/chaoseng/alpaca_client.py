import os
import json
from datetime import datetime, timezone

import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None


class AlpacaClient:
    """
    Thin wrapper around Alpaca's REST API.

    - Reads keys from config.json first, then falls back to environment variables:
        ALPACA_API_KEY
        ALPACA_SECRET_KEY
        ALPACA_PAPER
    - Uses paper endpoint when paper mode is enabled.
    - Provides convenience helpers for:
        * recent bars (OHLCV)
        * account equity
        * open positions
        * closing positions
        * submitting market orders
    """

    def __init__(self, config_path: str = "config.json"):
        # Load optional config.json
        cfg = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                try:
                    cfg = json.load(f)
                except Exception:
                    cfg = {}
        else:
            # Not fatal; we can still use env vars
            cfg = {}

        # Prefer config.json, fall back to env vars
        self.api_key = cfg.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
        self.secret_key = cfg.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

        # Paper/live flag
        if "ALPACA_PAPER" in cfg:
            self.paper = bool(cfg.get("ALPACA_PAPER"))
        else:
            # env var string -> bool
            env_paper = os.getenv("ALPACA_PAPER", "true").lower()
            self.paper = env_paper in ("1", "true", "yes", "y")

        self.tickers = cfg.get(
            "TICKERS",
            ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "AMZN", "META", "MSFT", "NFLX"],
        )

        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Alpaca API key/secret not found. "
                "Set them in config.json or as environment variables "
                "ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        if tradeapi is None:
            raise ImportError(
                "alpaca_trade_api is not installed. "
                "Make sure it's listed in requirements.txt."
            )

        base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"

        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url,
            api_version="v2",
        )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    def get_recent_bars(self, symbol: str, timeframe: str = "1Min", limit: int = 120) -> pd.DataFrame:
        """
        Fetch the most recent 'limit' bars for a symbol.

        IMPORTANT:
        We do NOT pass custom start/end timestamps here to avoid RFC3339
        formatting issues. Alpaca will simply return the latest 'limit' bars.
        """
        # Alpaca v2 get_bars can accept string timeframe like "1Min"
        bars = self.api.get_bars(symbol, timeframe, limit=limit)

        # In recent alpaca_trade_api versions, .df is the pandas DataFrame
        df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Ensure consistent columns
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0

        return df[cols].copy()

    # -------------------------------------------------------------------------
    # Account / positions
    # -------------------------------------------------------------------------
    def get_account_equity(self) -> float:
        acct = self.api.get_account()
        return float(acct.equity)

    def get_open_position(self, symbol: str):
        """
        Returns (qty, avg_entry_price) or (0.0, 0.0) if no open position.
        """
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty), float(pos.avg_entry_price)
        except Exception:
            return 0.0, 0.0

    def close_position(self, symbol: str):
        """
        Close any open position for the symbol.
        """
        try:
            self.api.close_position(symbol)
        except Exception:
            # No open position or other non-fatal error.
            pass

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------
    def submit_market_order(self, symbol: str, qty: float, side: str):
        """
        Submit a simple market order.

        side: "buy" or "sell"
        qty: positive number of shares (will be rounded down).
        """
        side = side.lower().strip()
        if qty <= 0:
            return None
        if side not in ("buy", "sell"):
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            return order
        except Exception as e:
            # Log server-side; in terminal output you'll see this if anything breaks.
            print(f"[AlpacaClient] Order error for {symbol}: {e}")
            return None

