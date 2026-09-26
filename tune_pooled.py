"""
Pooled hyperparameter search.

tune.py searches on ONE stock -- which is exactly what produced a false
signal earlier (the AAPL "winner" failed completely on MSFT). This script
searches for settings that work well ACROSS a basket of different stocks
AT THE SAME TIME. A config only wins here if it holds up across many
different companies, which is a much harder, more honest bar to clear.

Stock data is fetched and feature-engineered ONCE per ticker and reused
across every config tested (features don't depend on horizon/thresholds,
only labels do), so this stays reasonably fast despite testing many configs.

Usage:
    python tune_pooled.py
"""
import argparse
import itertools
import pandas as pd

from data import fetch_data
from features import add_features, FEATURE_COLUMNS
from labels import add_labels
from model import train_final_model
from backtest import run_backtest, performance_summary

# Financial-services names (conventional banks, payment networks) removed --
# interest-based revenue typically excludes them from Shariah screens.
# Verify any ticker's actual compliance yourself via Zoya / Musaffa / Islamicly
# before trading it -- this list is a reasonable starting basket, not a fatwa.
DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # tech
    "JNJ", "UNH",                                      # healthcare
    "COST", "WMT", "PG", "KO", "HD",                   # consumer
    "XOM",                                             # energy
    "DIS",                                             # media
]


def prefetch(tickers):
    data = {}
    for ticker in tickers:
        try:
            df = fetch_data(ticker, period="5y")
            df = add_features(df)
            data[ticker] = df
            print(f"  Loaded {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  {ticker}: skipped ({e})")
    return data


def evaluate_config(feature_data, model_type, horizon, buy_threshold, sell_threshold, train_frac=0.75):
    train_frames, test_frames = [], []
    for ticker, df in feature_data.items():
        labeled = add_labels(df, horizon=horizon, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        df_model = labeled.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
        if len(df_model) < 200:
            continue
        df_model["ticker"] = ticker
        split_idx = int(len(df_model) * train_frac)
        train_frames.append(df_model.iloc[:split_idx])
        test_frames.append(df_model.iloc[split_idx:])

    if not train_frames:
        return None
    train_df, test_df = pd.concat(train_frames), pd.concat(test_frames)
    if train_df["label"].nunique() < 2:
        return None

    model = train_final_model(train_df[FEATURE_COLUMNS], train_df["label"], model_type=model_type)

    sharpes, beat_flags = [], []
    for ticker in feature_data:
        ticker_test = test_df[test_df["ticker"] == ticker]
        if len(ticker_test) < 20:
            continue
        preds = model.predict(ticker_test[FEATURE_COLUMNS])
        signals = pd.Series(preds, index=ticker_test.index)
        equity_df, trades_df = run_backtest(ticker_test, signals)
        summary = performance_summary(equity_df, trades_df, ticker_test)
        sharpes.append(summary["Sharpe Ratio (annualized)"])
        beat_flags.append(summary["Total Return (%)"] > summary["Buy & Hold Return (%)"])

    if not sharpes:
        return None
    return {
        "model_type": model_type, "horizon": horizon,
        "buy_threshold": buy_threshold, "sell_threshold": sell_threshold,
        "avg_sharpe": sum(sharpes) / len(sharpes),
        "beat_buyhold_pct": sum(beat_flags) / len(beat_flags) * 100,
        "n_stocks": len(sharpes),
    }


def main():
    parser = argparse.ArgumentParser(description="Search for settings that work across many stocks at once")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_BASKET))
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"=== Loading & feature-engineering {len(tickers)} stocks (done once, reused for every config) ===")
    feature_data = prefetch(tickers)
    if len(feature_data) < 3:
        print("Not enough tickers loaded. Check your internet connection.")
        return

    model_types = ["rf", "gb"]
    horizons = [3, 5, 10]
    threshold_pairs = [(0.015, -0.015), (0.02, -0.02), (0.03, -0.03)]
    combos = list(itertools.product(model_types, horizons, threshold_pairs))

    print(f"\n=== Testing {len(combos)} configs, each evaluated across all {len(feature_data)} stocks ===\n")
    results = []
    for i, (model_type, horizon, (buy_t, sell_t)) in enumerate(combos, 1):
        print(f"  [{i}/{len(combos)}] model={model_type} horizon={horizon} buy>{buy_t} sell<{sell_t} ...", end=" ")
        result = evaluate_config(feature_data, model_type, horizon, buy_t, sell_t)
        if result:
            print(f"avg Sharpe={result['avg_sharpe']:.2f}  beat buy-hold on {result['beat_buyhold_pct']:.0f}% of stocks")
            results.append(result)
        else:
            print("skipped")

    if not results:
        print("No valid results.")
        return

    results_df = pd.DataFrame(results).sort_values("avg_sharpe", ascending=False)
    print("\n" + "=" * 70)
    print("TOP 5 CONFIGS BY AVERAGE SHARPE ACROSS ALL STOCKS")
    print("=" * 70)
    print(results_df.head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest pooled config: --model_type {best['model_type']} --horizon {int(best['horizon'])} "
          f"--buy_threshold {best['buy_threshold']} --sell_threshold {best['sell_threshold']}")
    print(f"Beat buy-and-hold on {best['beat_buyhold_pct']:.0f}% of the {int(best['n_stocks'])} stocks tested --")
    print("a cross-stock result, not a single-ticker fluke like the earlier AAPL-only search.")

    results_df.to_csv("tune_pooled_results.csv", index=False)
    print("\nFull results saved to tune_pooled_results.csv")


if __name__ == "__main__":
    main()
