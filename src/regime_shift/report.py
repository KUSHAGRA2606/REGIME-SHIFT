from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ReportPaths:
    out_dir: Path
    equity_png: Path
    regime_png: Path
    metrics_csv: Path
    weights_csv: Path
    turnover_csv: Path
    transition_csv: Path


def ensure_outputs(out_dir: str | Path = "outputs") -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_paths(out_dir: str | Path = "outputs") -> ReportPaths:
    p = ensure_outputs(out_dir)
    return ReportPaths(
        out_dir=p,
        equity_png=p / "equity_curves.png",
        regime_png=p / "regimes.png",
        metrics_csv=p / "metrics.csv",
        weights_csv=p / "weights.csv",
        turnover_csv=p / "turnover.csv",
        transition_csv=p / "transition.csv",
    )


def plot_equity_curves(curves: pd.DataFrame, path: Path) -> None:
    ax = curves.plot(figsize=(11, 5))
    ax.set_title("Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_regimes(price: pd.Series, regimes: pd.Series, path: Path) -> None:
    df = pd.DataFrame({"price": price}).dropna()
    r = regimes.reindex(df.index).ffill()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["price"], lw=1.3)

    colors = {"Bull": "#2ca02c", "Bear": "#ff7f0e", "Crisis": "#d62728"}
    for name, col in colors.items():
        mask = r == name
        if mask.any():
            ax.fill_between(df.index, df["price"].min(), df["price"].max(), where=mask.values, color=col, alpha=0.08)

    ax.set_title("Regimes Overlay")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_artifacts(
    paths: ReportPaths,
    metrics: pd.DataFrame,
    weights: pd.DataFrame,
    turnover: pd.Series,
    transition: Optional[pd.DataFrame] = None,
) -> None:
    metrics.to_csv(paths.metrics_csv, index=True)
    weights.to_csv(paths.weights_csv, index=True)
    turnover.to_csv(paths.turnover_csv, index=True)
    if transition is not None and len(transition) > 0:
        transition.to_csv(paths.transition_csv, index=True)
