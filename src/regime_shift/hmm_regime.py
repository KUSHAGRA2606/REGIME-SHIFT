from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


RegimeName = str


@dataclass(frozen=True)
class HMMResult:
    model: GaussianHMM
    hidden_states: pd.Series
    regime: pd.Series
    state_to_regime: Dict[int, RegimeName]
    transition: pd.DataFrame


def fit_hmm(X: pd.DataFrame, n_states: int, seed: int = 7) -> GaussianHMM:
    m = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=seed,
    )
    m.fit(X.values)
    return m


def infer_states(model: GaussianHMM, X: pd.DataFrame) -> pd.Series:
    states = model.predict(X.values)
    return pd.Series(states, index=X.index, name="state")


def _score_states(asset_returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    df = pd.concat([asset_returns, states.rename("state")], axis=1).dropna()
    g = df.groupby("state")
    stats = []
    for s, sub in g:
        eq = sub.iloc[:, 0]
        eq_mu = float(eq.mean())
        eq_vol = float(eq.std(ddof=0))
        stats.append((int(s), eq_mu, eq_vol))
    out = pd.DataFrame(stats, columns=["state", "eq_mu", "eq_vol"]).set_index("state")
    return out


def label_regimes(
    asset_returns: pd.DataFrame,
    states: pd.Series,
) -> Dict[int, RegimeName]:
    stats = _score_states(asset_returns, states)
    crisis_state = int(stats.sort_values(["eq_mu", "eq_vol"], ascending=[True, False]).index[0])
    bull_state = int(stats.sort_values(["eq_mu", "eq_vol"], ascending=[False, True]).index[0])
    other = [s for s in stats.index.tolist() if s not in {crisis_state, bull_state}]
    bear_state = int(other[0]) if len(other) else bull_state
    return {bull_state: "Bull", bear_state: "Bear", crisis_state: "Crisis"}


def build_transition_matrix(model: GaussianHMM, state_to_regime: Dict[int, RegimeName]) -> pd.DataFrame:
    A = model.transmat_.copy()
    idx = [f"{i}:{state_to_regime.get(i, str(i))}" for i in range(A.shape[0])]
    return pd.DataFrame(A, index=idx, columns=idx)


def fit_and_label(
    X: pd.DataFrame,
    asset_returns_for_label: pd.DataFrame,
    n_states: int,
    seed: int = 7,
) -> Tuple[GaussianHMM, Dict[int, RegimeName]]:
    model = fit_hmm(X, n_states=n_states, seed=seed)
    st = infer_states(model, X)
    mapping = label_regimes(asset_returns_for_label.loc[st.index], st)
    return model, mapping


def infer_regime_series(model: GaussianHMM, X: pd.DataFrame, state_to_regime: Dict[int, RegimeName]) -> HMMResult:
    states = infer_states(model, X)
    regime = states.map(state_to_regime).rename("regime")
    transition = build_transition_matrix(model, state_to_regime)
    return HMMResult(
        model=model,
        hidden_states=states,
        regime=regime,
        state_to_regime=state_to_regime,
        transition=transition,
    )
