from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    x = daily_returns.dropna() - rf_daily
    if x.std(ddof=0) == 0:
        return float("nan")
    return float(np.sqrt(252.0) * x.mean() / x.std(ddof=0))


def sortino(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    x = daily_returns.dropna() - rf_daily
    downside = x[x < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return float("nan")
    return float(np.sqrt(252.0) * x.mean() / dd)


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return float("nan")
    return float(cagr(equity) / dd)


def avg_turnover(turnover: pd.Series) -> float:
    if turnover is None or len(turnover) == 0:
        return float("nan")
    return float(turnover.mean())


def tear_sheet(
    daily_returns: pd.Series,
    turnover: pd.Series | None = None,
    rf_daily: float = 0.0,
) -> pd.Series:
    eq = (1.0 + daily_returns.dropna()).cumprod()
    out = {
        "CAGR": cagr(eq),
        "Sharpe": sharpe(daily_returns, rf_daily=rf_daily),
        "Sortino": sortino(daily_returns, rf_daily=rf_daily),
        "MaxDrawdown": max_drawdown(eq),
        "Calmar": calmar(eq),
    }
    if turnover is not None:
        out["AvgTurnover"] = avg_turnover(turnover)
    return pd.Series(out)
