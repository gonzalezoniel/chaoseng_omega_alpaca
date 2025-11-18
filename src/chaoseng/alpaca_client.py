import os
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None


class AlpacaClient:
    def __init__(self, config_path: str = "config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError("config.json not found. Please create it with your Alpaca keys.")
        with open(config_path) as f:
            cfg = json.load(f)

        self.api_key = cfg.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
        self.secret_key = cfg.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        self.paper = cfg.get("ALPACA_PAPER", True)
        self.tickers = cfg.get("TICKERS", ["SPY","QQQ","AAPL","TSLA","NVDA","AMD","AMZN","META","MSFT","NFLX"])

        if tradeapi is None:
            raise ImportError("alpaca_trade_api is not installed. Add it to requirements and reinstall.")

        base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        self.api = tradeapi.REST(self.api_key, self.secret_key, base_url, api_version="v2")

    def get_recent_bars(self, symbol: str, timeframe: str = "1Min", limit: int = 120) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=3)
        bars = self.api.get_bars(symbol, timeframe, start=start, end=end, limit=limit)
        df = bars.df
        if len(df) == 0:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        # Ensure columns
        return df[["open","high","low","close","volume"]].copy()

    def get_account_equity(self) -> float:
        acct = self.api.get_account()
        return float(acct.equity)

    def get_open_position(self, symbol: str):
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty), float(pos.avg_entry_price)
        except Exception:
            return 0.0, 0.0

    def close_position(self, symbol: str):
        try:
            self.api.close_position(symbol)
        except Exception:
            pass

    def submit_market_order(self, symbol: str, qty: float, side: str):
        side = side.lower()
        if qty <= 0:
            return None
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
            return order
        except Exception as e:
            print("Order error:", e)
            return None

