from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame
    feature_cols: list[str]


def build_features(
    asset_returns: pd.DataFrame,
    vix_level: Optional[pd.Series] = None,
    macro: Optional[pd.DataFrame] = None,
) -> FeatureSet:
    df = pd.DataFrame(index=asset_returns.index)

    for c in asset_returns.columns:
        df[f"ret_{c}"] = asset_returns[c]

    if vix_level is not None:
        vix_level = vix_level.reindex(df.index).ffill()
        df["vix"] = vix_level
        df["vix_chg"] = vix_level.pct_change()

    if macro is not None and len(macro.columns) > 0:
        m = macro.reindex(df.index).ffill()
        for c in m.columns:
            df[c] = m[c]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    cols = list(df.columns)
    X = (df - df.mean()) / (df.std(ddof=0) + 1e-12)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return FeatureSet(X=X, feature_cols=cols)
