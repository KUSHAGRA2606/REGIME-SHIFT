# REGIME-SHIFT — Macro-Aware Tactical Asset Allocation Engine

Dynamic allocation engine using HMM regime detection + convex portfolio optimization with strict walk-forward validation and explicit transaction friction.

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Optional (macro via FRED): create `.env` with:

```text
FRED_API_KEY=your_key
```

## Run (script)

```bash
python scripts/run_backtest.py --start 2006-01-01 --end 2025-12-31 --cost-bps 7.5
```

Outputs saved to `outputs/` (equity curves, regime chart, metrics table, weights, turnover).

## Notebook

Open `notebooks/regime_shift.ipynb` and run top-to-bottom.

## Architecture

- `src/regime_shift/data.py` — data ingestion (Yahoo + optional FRED)
- `src/regime_shift/hmm_regime.py` — HMM fit/inference + regime labeling
- `src/regime_shift/optimizer.py` — CVXPY allocation objectives + regime constraints
- `src/regime_shift/backtest.py` — walk-forward backtest + transaction cost model
- `src/regime_shift/metrics.py` — Sharpe/Sortino/Calmar/max drawdown/turnover
- `src/regime_shift/report.py` — plots + tear sheet export

## Notes

- Walk-forward logic uses only data available up to $t-1$ to decide weights for day $t$.
- Rebalancing is monthly by default.
