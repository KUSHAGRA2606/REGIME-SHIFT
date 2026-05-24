from __future__ import annotations

import argparse

from regime_shift.config import BacktestConfig, OptimizerConfig
from regime_shift.pipeline import run_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default=BacktestConfig.start)
    p.add_argument("--end", type=str, default=BacktestConfig.end)
    p.add_argument("--train-years", type=int, default=BacktestConfig.train_years)
    p.add_argument("--states", type=int, default=BacktestConfig.n_states)
    p.add_argument("--rebalance", type=str, default=BacktestConfig.rebalance)
    p.add_argument("--cost-bps", type=float, default=BacktestConfig.cost_bps)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bt = BacktestConfig(
        start=args.start,
        end=args.end,
        train_years=args.train_years,
        n_states=args.states,
        rebalance=args.rebalance,
        cost_bps=args.cost_bps,
    )
    opt = OptimizerConfig()

    res = run_engine(bt=bt, opt=opt)

    print(res.metrics.round(4))
    print("\nSaved artifacts to outputs/")


if __name__ == "__main__":
    main()
