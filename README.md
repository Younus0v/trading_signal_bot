# ML Trading Signal System: A Cross-Asset Validation Case Study

**Younus Hassen Abdulkadir** — Etehaad International School

An independent research project investigating a specific, common failure mode in applied machine learning for finance: models that look highly successful in backtesting but fail to generalize to new data or new assets ("backtest overfitting").

**Read the full research paper:** [`research_paper.docx`](./research_paper.docx)

**Note on this repository's history:** this project was fully researched, built, and tested privately over several weeks before being uploaded to GitHub as a completed body of work. The commit history reflects the upload of a finished project, not the full incremental development process.

## Disclaimer — Not Financial Advice

**This project is for educational and research purposes only. It is not financial advice, and the system it describes is not recommended for real trading with real money.** All results in this repository come from backtesting and paper trading (simulated money only) — no real funds were ever traded. Past backtested or paper-traded performance does not indicate future results, and the paper itself documents multiple cases where seemingly strong results failed to generalize. Nothing in this repository, its code, or its accompanying paper should be interpreted as investment advice or a recommendation to buy, sell, or hold any security. Anyone using this code with real money does so entirely at their own risk and discretion.

## What this project actually found

1. **A hyperparameter search on Apple (AAPL) stock found a configuration with a Sharpe ratio of 2.82** — a very strong result. Applying that exact same configuration to Microsoft (MSFT), a stock it had never seen, produced a Sharpe ratio of **-0.62** — one of the worst results in the entire search. This is a direct, empirical demonstration of overfitting to a single asset's specific historical path.

2. **In response, the system was rebuilt to validate across 15 stocks simultaneously** rather than tuning on one at a time. This produced a more modest but far more credible result: 46.2% classification accuracy and an average Sharpe ratio of 0.85, beating buy-and-hold on 53% of tested stocks — though formal significance testing shows neither figure is statistically distinguishable from a trivial baseline at this sample size, which the paper reports directly rather than hiding.

3. **A second, more precise diagnostic was run using the Deflated Sharpe Ratio framework** (Bailey & López de Prado, 2014), showing the AAPL failure was driven less by pure random search noise than by the search finding configurations exposed to AAPL's strong upward trend during that specific test window — a subtler and more accurate explanation than "overfitting" alone.

4. **An alternative labeling method (triple-barrier) was tested and rejected** after it achieved *higher* raw accuracy (52.3% vs. 46.2%) while performing *worse* as an actual trading strategy — because it had learned to predict "SELL" almost every time, inflating its accuracy score while providing little useful signal. Reported as a negative result rather than omitted.

5. **The validated system was deployed as a live, automated paper-trading bot** (Alpaca Markets) with real risk controls (stop-loss, take-profit, daily loss limits), for ongoing forward validation with real, out-of-sample market data — simulated money only.

## Repository contents

| File | Purpose |
|---|---|
| `data.py` | Fetches historical price data (Yahoo Finance) |
| `features.py` | Computes technical indicator features |
| `labels.py` | Threshold-based and triple-barrier labeling methods |
| `model.py` | Random Forest / Gradient Boosting training and walk-forward validation |
| `backtest.py` | Chronological out-of-sample backtesting with transaction costs |
| `main.py` | Runs the full single-asset pipeline end to end |
| `train_pooled.py` | Trains and evaluates the cross-asset pooled model (Experiment 2) |
| `tune.py` / `tune_pooled.py` | Hyperparameter search, single-asset and pooled |
| `alpaca_trader.py` / `run_bot.py` | Paper-trading deployment with risk management |
| `research_paper.docx` | The full write-up of methodology, results, and findings |

## Methodology notes

- All results use a **strict chronological train/test split** — the model never sees the future during training.
- Walk-forward cross-validation (`TimeSeriesSplit`) is used during development; no random shuffling of time-series data anywhere in this project.
- Every reported result, including the negative ones, is included. Nothing that contradicted the initial hypothesis was omitted.

## A note on how this was built

The decision to test and reject the triple-barrier hypothesis, building the whole system, the interpretation of all results and etc are the author's own.
