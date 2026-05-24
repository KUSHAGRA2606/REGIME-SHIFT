from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .backtest import baseline_static, walk_forward_backtest
from .config import BacktestConfig, OptimizerConfig, Universe
from .data import fetch_fred_series, fetch_yahoo_prices, prices_to_returns, safe_get_env_api_key
from .metrics import tear_sheet
from .report import make_paths, plot_equity_curves, plot_regimes, write_artifacts


@dataclass(frozen=True)
class RunResult:
    curves: pd.DataFrame
    metrics: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.Series
    regimes: pd.Series
    transition: pd.DataFrame


def load_macro_fred(start: str, end: str) -> Optional[pd.DataFrame]:
    api_key = safe_get_env_api_key(os.environ, "FRED_API_KEY")
    if api_key is None:
        return None

    cpi = fetch_fred_series("CPIAUCSL", start, end, api_key=api_key)
    dgs10 = fetch_fred_series("DGS10", start, end, api_key=api_key)
    dgs2 = fetch_fred_series("DGS2", start, end, api_key=api_key)

    macro = pd.concat(
        {
            "cpi_yoy": cpi.pct_change(12),
            "term_spread": dgs10 - dgs2,
        },
        axis=1,
    )
    macro = macro.dropna(how="all")
    macro.index = pd.to_datetime(macro.index)
    return macro


def run_engine(
    bt: BacktestConfig,
    opt: OptimizerConfig,
    universe: Universe = Universe(),
    out_dir: str = "outputs",
) -> RunResult:
    tickers = [universe.risky, universe.rates, universe.haven, universe.vix]
    px = fetch_yahoo_prices(tickers, bt.start, bt.end)

    prices_assets = px[[c for c in [universe.risky, universe.rates, universe.haven] if c in px.columns]]
    vix_px = px[universe.vix] if universe.vix in px.columns else None

    macro = load_macro_fred(bt.start, bt.end)

    res = walk_forward_backtest(
        prices=prices_assets,
        vix_prices=vix_px,
        macro=macro,
        universe=universe,
        bt=bt,
        opt=opt,
    )

    asset_rets = prices_to_returns(prices_assets).reindex(res.daily_returns.index).dropna()

    w_6040 = pd.Series({universe.risky: 0.6, universe.rates: 0.4, universe.haven: 0.0})
    w_eq = pd.Series(1.0 / asset_rets.shape[1], index=asset_rets.columns)

    r_dyn = res.daily_returns
    r_6040 = baseline_static(asset_rets, w_6040)
    r_eq = baseline_static(asset_rets, w_eq)

    curves = pd.DataFrame(
        {
            "Dynamic": (1.0 + r_dyn).cumprod(),
            "60_40": (1.0 + r_6040).cumprod(),
            "Equal": (1.0 + r_eq).cumprod(),
        }
    ).dropna()

    rf_daily = (1.0 + bt.rf_annual) ** (1.0 / 252.0) - 1.0

    metrics = pd.DataFrame(
        {
            "Dynamic": tear_sheet(r_dyn, turnover=res.turnover, rf_daily=rf_daily),
            "60_40": tear_sheet(r_6040, rf_daily=rf_daily),
            "Equal": tear_sheet(r_eq, rf_daily=rf_daily),
        }
    )

    paths = make_paths(out_dir)
    plot_equity_curves(curves, paths.equity_png)
    plot_regimes(prices_assets[universe.risky].reindex(res.regimes.index).ffill(), res.regimes, paths.regime_png)
    write_artifacts(paths, metrics, res.weights, res.turnover, transition=res.transition)

    return RunResult(
        curves=curves,
        metrics=metrics,
        weights=res.weights,
        turnover=res.turnover,
        regimes=res.regimes,
        transition=res.transition,
    )
