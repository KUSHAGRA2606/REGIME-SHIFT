from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Universe:
    risky: str = "SPY"
    rates: str = "TLT"
    haven: str = "GLD"
    vix: str = "^VIX"


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2006-01-01"
    end: str = "2025-12-31"
    rebalance: str = "ME"  # pandas offset alias (month-end)
    train_years: int = 3
    n_states: int = 3
    cost_bps: float = 7.5
    min_obs: int = 252
    rf_annual: float = 0.02


@dataclass(frozen=True)
class OptimizerConfig:
    long_only: bool = True
    w_min: float = 0.0
    w_max: float = 1.0
    eq_max_bull: float = 0.75
    eq_max_bear: float = 0.45
    eq_max_crisis: float = 0.20
    haven_min_crisis: float = 0.25
    rates_min_crisis: float = 0.30
    lambda_bull: float = 2.0
    lambda_bear: float = 6.0
    solver: str = "ECOS"
