from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame


def fetch_yahoo_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys(list(tickers)))
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if isinstance(px.columns, pd.MultiIndex):
        adj = {}
        for t in tickers:
            if (t, "Close") in px.columns:
                adj[t] = px[(t, "Close")]
            elif (t, "Adj Close") in px.columns:
                adj[t] = px[(t, "Adj Close")]
        out = pd.DataFrame(adj)
    else:
        out = px[["Close"]].rename(columns={"Close": tickers[0]})

    out = out.dropna(how="all").sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    return rets.dropna(how="all")


def fetch_fred_series(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    idx = [pd.to_datetime(o["date"]) for o in obs]
    vals = []
    for o in obs:
        v = o.get("value")
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    s = pd.Series(vals, index=idx, name=series_id).sort_index()
    return s.loc[pd.to_datetime(start) : pd.to_datetime(end)]


def load_market_data(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> MarketData:
    prices = fetch_yahoo_prices(tickers, start, end)
    returns = prices_to_returns(prices)
    return MarketData(prices=prices, returns=returns)


def align_daily(*series: pd.Series) -> pd.DataFrame:
    df = pd.concat(series, axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def infer_rf_daily(rf_annual: float) -> float:
    return (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0


def safe_get_env_api_key(env: dict, key: str) -> Optional[str]:
    v = env.get(key)
    if v is None:
        return None
    v = str(v).strip()
    return v or None
