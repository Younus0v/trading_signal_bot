"""
Data loading for the trading signal system.
"""
import pandas as pd
import numpy as np


def fetch_data(ticker, period="5y", interval="1d"):
    """Fetch historical OHLCV data using yfinance. Requires internet access."""
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def generate_synthetic_data(n_days=1500, seed=42):
    """
    Generate synthetic OHLCV data so the pipeline can be tested end-to-end
    without internet access or a real ticker. NOT real market data --
    for verifying the code works, not for drawing conclusions.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n_days = len(dates)  # bdate_range can return a slightly different count than requested

    base_returns = rng.normal(0.0004, 0.015, n_days)
    regime = np.sin(np.linspace(0, 15, n_days)) * 0.001  # gives the model *something* learnable
    returns = base_returns + regime

    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    volume = rng.integers(1_000_000, 10_000_000, n_days)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df
