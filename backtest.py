"""
Backtesting: simulate actually trading the model's signals, with costs.
"""
import numpy as np
import pandas as pd


def run_backtest(df, signals, transaction_cost=0.001, initial_capital=10000):
    """
    Simple long-only simulation:
      BUY  -> enter a full position (if currently in cash)
      SELL -> exit to cash (if currently holding)
      HOLD -> do nothing
    `transaction_cost` (e.g. 0.001 = 0.1%) is deducted on every entry and exit,
    to avoid pretending trading is free.
    """
    position = 0
    cash = initial_capital
    shares = 0
    entry_price = None
    equity_curve = []
    trades = []

    for date, row in df.iterrows():
        price = row["close"]
        signal = signals.loc[date] if date in signals.index else "HOLD"

        if signal == "BUY" and position == 0:
            shares = (cash * (1 - transaction_cost)) / price
            cash = 0
            position = 1
            entry_price = price
            trades.append({"date": date, "action": "BUY", "price": price})

        elif signal == "SELL" and position == 1:
            cash = shares * price * (1 - transaction_cost)
            trades.append({
                "date": date, "action": "SELL", "price": price,
                "return_pct": (price / entry_price - 1) * 100,
            })
            shares = 0
            position = 0
            entry_price = None

        equity_curve.append({"date": date, "equity": cash + shares * price})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    return equity_df, trades_df


def performance_summary(equity_df, trades_df, df, initial_capital=10000):
    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    buy_hold_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    daily_returns = equity_df["equity"].pct_change().dropna()
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    n_trades = 0
    win_rate = "N/A"
    if not trades_df.empty:
        completed = trades_df[trades_df["action"] == "SELL"]
        n_trades = len(completed)
        if n_trades > 0:
            win_rate = round((completed["return_pct"] > 0).mean() * 100, 2)

    return {
        "Total Return (%)": round(total_return, 2),
        "Buy & Hold Return (%)": round(buy_hold_return, 2),
        "Sharpe Ratio (annualized)": round(sharpe, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Number of Completed Trades": n_trades,
        "Win Rate (%)": win_rate,
        "Final Equity ($)": round(final_equity, 2),
    }
