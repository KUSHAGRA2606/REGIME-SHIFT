from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import OptimizerConfig


@dataclass(frozen=True)
class OptResult:
    weights: pd.Series
    status: str


def _ensure_psd(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def estimate_mu_sigma(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mu = returns.mean()
    Sigma = returns.cov()
    return mu, Sigma


def regime_constraints(
    regime: str,
    assets: list[str],
    cfg: OptimizerConfig,
    asset_roles: Dict[str, str],
) -> list:
    n = len(assets)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if cfg.long_only:
        cons += [w >= cfg.w_min, w <= cfg.w_max]

    role_idx = {asset_roles[a]: i for i, a in enumerate(assets)}
    eq_i = role_idx.get("risky")
    rates_i = role_idx.get("rates")
    haven_i = role_idx.get("haven")

    if eq_i is not None:
        if regime == "Bull":
            cons += [w[eq_i] <= cfg.eq_max_bull]
        elif regime == "Bear":
            cons += [w[eq_i] <= cfg.eq_max_bear]
        elif regime == "Crisis":
            cons += [w[eq_i] <= cfg.eq_max_crisis]

    if regime == "Crisis":
        if haven_i is not None:
            cons += [w[haven_i] >= cfg.haven_min_crisis]
        if rates_i is not None:
            cons += [w[rates_i] >= cfg.rates_min_crisis]

    return w, cons


def solve_allocation(
    returns_window: pd.DataFrame,
    regime: str,
    cfg: OptimizerConfig,
    asset_roles: Dict[str, str],
    mu_override: Optional[pd.Series] = None,
    Sigma_override: Optional[pd.DataFrame] = None,
) -> OptResult:
    assets = list(returns_window.columns)
    mu, Sigma = estimate_mu_sigma(returns_window)
    if mu_override is not None:
        mu = mu_override.reindex(assets)
    if Sigma_override is not None:
        Sigma = Sigma_override.reindex(index=assets, columns=assets)

    S = _ensure_psd(Sigma.values)
    mu_v = mu.values

    w, cons = regime_constraints(regime=regime, assets=assets, cfg=cfg, asset_roles=asset_roles)

    if regime == "Crisis":
        obj = cp.Minimize(cp.quad_form(w, S))
    else:
        lam = cfg.lambda_bull if regime == "Bull" else cfg.lambda_bear
        obj = cp.Maximize(mu_v @ w - lam * cp.quad_form(w, S))

    prob = cp.Problem(obj, cons)

    solver = cfg.solver
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if w.value is None:
        return OptResult(weights=pd.Series(np.nan, index=assets), status=str(prob.status))

    wv = np.asarray(w.value).reshape(-1)
    wv = np.clip(wv, 0.0, 1.0)
    if wv.sum() > 0:
        wv = wv / wv.sum()
    return OptResult(weights=pd.Series(wv, index=assets), status=str(prob.status))
