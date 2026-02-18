import os
import json
from datetime import datetime, timezone

import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


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
        self.paper = _as_bool(cfg.get("ALPACA_PAPER", os.getenv("ALPACA_PAPER", "true")), default=True)

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

    def get_open_positions(self) -> dict:
        """
        Return a dict of open positions keyed by symbol.
        """
        try:
            positions = self.api.list_positions()
        except Exception:
            return {}

        return {pos.symbol: pos for pos in positions}

    def get_clock(self):
        return self.api.get_clock()

    def get_account(self):
        return self.api.get_account()

    def is_paper(self) -> bool:
        return bool(self.paper)

    # -------------------------------------------------------------------------
    # Account / positions
    # -------------------------------------------------------------------------
    def get_account_equity(self) -> float:
        acct = self.api.get_account()
        return float(acct.equity)

    def get_equity(self) -> float:
        return self.get_account_equity()

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

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float | None,
        stop_price: float,
        take_profit_price: float,
    ):
        """
        Submit a bracket order with optional limit entry.
        """
        side = side.lower().strip()
        if qty <= 0 or side not in ("buy", "sell"):
            return None

        order_kwargs = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": take_profit_price},
            "stop_loss": {"stop_price": stop_price},
        }

        if entry_price is None:
            order_kwargs["type"] = "market"
        else:
            order_kwargs["type"] = "limit"
            order_kwargs["limit_price"] = entry_price

        try:
            return self.api.submit_order(**order_kwargs)
        except Exception as e:
            print(f"[AlpacaClient] Bracket order error for {symbol}: {e}")
            return None

    def get_trade_history(self, limit: int = 200) -> list[dict]:
        """
        Return a recent list of filled orders (enter/exit).
        """
        try:
            orders = self.api.list_orders(status="closed", limit=limit)
        except Exception:
            return []

        history = []
        for order in orders:
            filled_at = getattr(order, "filled_at", None)
            if filled_at is None:
                continue
            history.append(
                {
                    "id": getattr(order, "id", None),
                    "symbol": getattr(order, "symbol", None),
                    "side": getattr(order, "side", None),
                    "qty": getattr(order, "filled_qty", None),
                    "filled_avg_price": getattr(order, "filled_avg_price", None),
                    "filled_at": filled_at.isoformat() if hasattr(filled_at, "isoformat") else str(filled_at),
                    "type": getattr(order, "type", None),
                    "status": getattr(order, "status", None),
                }
            )
        return history

    def get_pnl_summary(self) -> dict:
        """
        Return a simple PnL/equity snapshot.
        """
        acct = self.api.get_account()
        summary = {
            "equity": float(getattr(acct, "equity", 0.0)),
            "last_equity": float(getattr(acct, "last_equity", 0.0)),
            "portfolio_value": float(getattr(acct, "portfolio_value", 0.0)),
            "cash": float(getattr(acct, "cash", 0.0)),
            "buying_power": float(getattr(acct, "buying_power", 0.0)),
            "unrealized_pl": float(getattr(acct, "unrealized_pl", 0.0)),
            "unrealized_plpc": float(getattr(acct, "unrealized_plpc", 0.0)),
        }
        return summary
        
