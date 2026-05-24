from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, OptimizerConfig, Universe
from .data import infer_rf_daily
from .features import FeatureSet, build_features
from .hmm_regime import fit_and_label, infer_regime_series
from .optimizer import solve_allocation


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    regimes: pd.Series
    turnover: pd.Series
    transition: pd.DataFrame


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("M").to_timestamp("M").unique()


def rebalance_schedule(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    sched = pd.date_range(idx.min(), idx.max(), freq=freq)
    return pd.DatetimeIndex(sched)


def _turnover(w_new: pd.Series, w_old: pd.Series) -> float:
    d = (w_new - w_old).abs().sum()
    return float(d)


def walk_forward_backtest(
    prices: pd.DataFrame,
    vix_prices: Optional[pd.Series],
    macro: Optional[pd.DataFrame],
    universe: Universe,
    bt: BacktestConfig,
    opt: OptimizerConfig,
    seed: int = 7,
) -> BacktestResult:
    asset_prices = prices[[universe.risky, universe.rates, universe.haven]].dropna()
    asset_rets = asset_prices.pct_change().dropna()

    vix_level = None
    if vix_prices is not None:
        vix_level = vix_prices.reindex(asset_rets.index).ffill()

    feats: FeatureSet = build_features(asset_rets, vix_level=vix_level, macro=macro)
    X = feats.X

    idx = asset_rets.index.intersection(X.index).sort_values()
    asset_rets = asset_rets.loc[idx]
    X = X.loc[idx]

    rebals = rebalance_schedule(idx, bt.rebalance)
    rebals = rebals[(rebals >= idx.min()) & (rebals <= idx.max())]

    train_days = int(bt.train_years * 252)
    rf_daily = infer_rf_daily(bt.rf_annual)
    cost = bt.cost_bps / 10000.0

    asset_roles = {universe.risky: "risky", universe.rates: "rates", universe.haven: "haven"}

    w_prev = pd.Series(1.0 / 3.0, index=asset_rets.columns)
    w_hist = []
    t_hist = []
    regime_hist = []
    turnover_hist = []
    status_hist = []

    daily_port_rets = pd.Series(index=asset_rets.index, dtype=float)

    last_model = None
    last_map: Dict[int, str] = {}
    last_transition = pd.DataFrame()

    for r in rebals:
        if r not in idx:
            r = idx[idx.get_indexer([r], method="ffill")[0]]

        pos = idx.get_loc(r)
        train_start = max(0, pos - train_days)
        train_slice = slice(idx[train_start], idx[pos])

        X_train = X.loc[train_slice]
        R_train = asset_rets.loc[train_slice]
        if len(X_train) < bt.min_obs or len(R_train) < bt.min_obs:
            continue

        model, mapping = fit_and_label(
            X=X_train,
            asset_returns_for_label=R_train[[universe.risky]],
            n_states=bt.n_states,
            seed=seed,
        )
        last_model, last_map = model, mapping
        hmm_res = infer_regime_series(model, X_train, mapping)
        last_transition = hmm_res.transition

        state_now = model.predict(X.loc[[idx[pos]]].values)[0]
        regime_now = mapping.get(int(state_now), "Bear")

        opt_res = solve_allocation(
            returns_window=R_train,
            regime=regime_now,
            cfg=opt,
            asset_roles=asset_roles,
        )

        w_new = opt_res.weights.copy()
        if w_new.isna().any() or not np.isfinite(w_new.values).all():
            w_new = w_prev.copy()

        to = _turnover(w_new, w_prev)
        tc = cost * to

        w_hist.append(w_new)
        t_hist.append(r)
        regime_hist.append(regime_now)
        turnover_hist.append(to)
        status_hist.append(opt_res.status)

        next_pos = pos + 1
        next_r = idx[next_pos] if next_pos < len(idx) else None
        end = idx[-1]
        hold_start = next_r if next_r is not None else end
        hold_end = end

        next_rebals = rebals[rebals > r]
        if len(next_rebals) > 0:
            nr = next_rebals[0]
            nr = nr if nr in idx else idx[idx.get_indexer([nr], method="ffill")[0]]
            hold_end = nr

        hold_idx = asset_rets.loc[hold_start:hold_end].index
        if len(hold_idx) == 0:
            w_prev = w_new
            continue

        pr = (asset_rets.loc[hold_idx] * w_new).sum(axis=1)
        pr.iloc[0] = pr.iloc[0] - tc
        daily_port_rets.loc[hold_idx] = pr

        w_prev = w_new

    daily_port_rets = daily_port_rets.dropna()
    equity = (1.0 + daily_port_rets).cumprod().rename("equity")

    w_df = pd.DataFrame(w_hist, index=pd.to_datetime(t_hist))
    regimes = pd.Series(regime_hist, index=pd.to_datetime(t_hist), name="regime")
    turnover_s = pd.Series(turnover_hist, index=pd.to_datetime(t_hist), name="turnover")

    return BacktestResult(
        equity_curve=equity,
        daily_returns=daily_port_rets,
        weights=w_df,
        regimes=regimes,
        turnover=turnover_s,
        transition=last_transition,
    )


def baseline_static(asset_returns: pd.DataFrame, w: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    w = w.reindex(asset_returns.columns).fillna(0.0)
    w = w / w.sum()
    r = (asset_returns * w).sum(axis=1)
    if cost_bps > 0:
        r = r.copy()
        r.iloc[0] = r.iloc[0] - (cost_bps / 10000.0) * 0.0
    return r
