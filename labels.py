"""
Label creation: what the model is trying to predict.
"""
import numpy as np


def add_labels(df, horizon=5, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Label each day using the ACTUAL forward return over `horizon` trading days
    (this is only usable for training, since it looks into the future).

    BUY  = forward return > buy_threshold
    SELL = forward return < sell_threshold
    HOLD = otherwise
    """
    df = df.copy()
    fwd_return = df["close"].shift(-horizon) / df["close"] - 1
    df["forward_return"] = fwd_return

    conditions = [
        fwd_return > buy_threshold,
        fwd_return < sell_threshold,
    ]
    choices = ["BUY", "SELL"]
    df["label"] = np.select(conditions, choices, default="HOLD")
    return df


def add_triple_barrier_labels(df, take_profit_pct=0.03, stop_loss_pct=0.02, max_holding_days=10):
    """
    A more realistic label: simulates an actual trade from each day, using
    daily high/low to check which barrier gets touched first.

    BUY  = take-profit would have been hit first (this was a good entry point)
    SELL = stop-loss would have been hit first (this was a bad entry point)
    HOLD = neither hit within max_holding_days (inconclusive -- no strong
           opportunity either way)

    LIMITATION, stated plainly: with only daily bars, if BOTH the take-profit
    and stop-loss price are touched on the SAME day, there's no way to know
    which happened first without intraday data. This conservatively assumes
    the stop-loss was hit first in that case, since assuming the better
    outcome would make the labels overly optimistic.
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    labels = np.array(["HOLD"] * n, dtype=object)

    for i in range(n - 1):
        entry_price = close[i]
        tp_price = entry_price * (1 + take_profit_pct)
        sl_price = entry_price * (1 - stop_loss_pct)
        end = min(i + 1 + max_holding_days, n)

        label = "HOLD"
        for j in range(i + 1, end):
            hit_tp = high[j] >= tp_price
            hit_sl = low[j] <= sl_price
            if hit_tp and hit_sl:
                label = "SELL"  # conservative: can't know order, assume the worse case
                break
            elif hit_tp:
                label = "BUY"
                break
            elif hit_sl:
                label = "SELL"
                break
        labels[i] = label

    df = df.copy()
    df["label"] = labels
    return df
