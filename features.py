"""
Technical indicator feature engineering.
"""
import numpy as np


def add_features(df):
    df = df.copy()
    close = df["close"]

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)

    for w in (10, 20, 50):
        sma = close.rolling(w).mean()
        df[f"sma_{w}_ratio"] = close / sma - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_width"] = (bb_upper - bb_lower) / sma20
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    df["volume_change"] = df["volume"].pct_change()
    volume_sma_20 = df["volume"].rolling(20).mean()
    df["relative_volume"] = df["volume"] / volume_sma_20

    df["momentum_10d"] = close / close.shift(10) - 1

    return df


FEATURE_COLUMNS = [
    "return_1d", "return_5d",
    "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_width", "bb_position",
    "volatility_10d", "volatility_20d",
    "volume_change", "relative_volume",
    "momentum_10d",
]
