"""
Systematic search: instead of guessing at settings, this tries many honest
combinations and reports which ones actually performed best out-of-sample.

This is the real answer to "make it more accurate" -- not a bigger promise,
but an honest search over what's actually testable. Every result here is
still out-of-sample (the model never sees the test period during training),
so this won't produce fantasy numbers -- it'll show you real trade-offs.

Usage:
    python tune.py --ticker AAPL
"""
import argparse
import itertools
import pandas as pd

from data import fetch_data
from features import add_features, FEATURE_COLUMNS
from labels import add_labels
from model import train_final_model
from backtest import run_backtest, performance_summary


def evaluate_config(df_raw, model_type, horizon, buy_threshold, sell_threshold, train_frac=0.75):
    df = add_features(df_raw)
    df = add_labels(df, horizon=horizon, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    df_model = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()

    if len(df_model) < 200:
        return None

    split_idx = int(len(df_model) * train_frac)
    train_df, test_df = df_model.iloc[:split_idx], df_model.iloc[split_idx:]
    if len(test_df) < 30:
        return None

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test = test_df[FEATURE_COLUMNS]

    if y_train.nunique() < 2:
        return None

    model = train_final_model(X_train, y_train, model_type=model_type)
    preds = model.predict(X_test)

    signals = pd.Series(preds, index=X_test.index)
    equity_df, trades_df = run_backtest(test_df, signals)
    summary = performance_summary(equity_df, trades_df, test_df)

    return {
        "model_type": model_type,
        "horizon": horizon,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        **summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Search over model/threshold combinations for the best out-of-sample risk-adjusted return")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--period", type=str, default="5y")
    args = parser.parse_args()

    print(f"Fetching data for {args.ticker}...")
    df_raw = fetch_data(args.ticker, period=args.period)
    print(f"Loaded {len(df_raw)} rows.\n")

    model_types = ["rf", "gb"]
    horizons = [3, 5, 10]
    threshold_pairs = [(0.015, -0.015), (0.02, -0.02), (0.03, -0.03)]

    combos = list(itertools.product(model_types, horizons, threshold_pairs))
    print(f"Testing {len(combos)} combinations (this will take a few minutes)...\n")

    results = []
    for i, (model_type, horizon, (buy_t, sell_t)) in enumerate(combos, 1):
        print(f"  [{i}/{len(combos)}] model={model_type} horizon={horizon} buy>{buy_t} sell<{sell_t} ...", end=" ")
        try:
            result = evaluate_config(df_raw, model_type, horizon, buy_t, sell_t)
            if result:
                print(f"Sharpe={result['Sharpe Ratio (annualized)']:.2f}  Return={result['Total Return (%)']:.1f}%")
                results.append(result)
            else:
                print("skipped (not enough data)")
        except Exception as e:
            print(f"failed ({e})")

    if not results:
        print("\nNo valid results -- try a ticker with more history.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe Ratio (annualized)", ascending=False)

    print("\n" + "=" * 70)
    print(f"TOP 5 CONFIGS BY SHARPE RATIO (risk-adjusted return) for {args.ticker}")
    print("=" * 70)
    cols = ["model_type", "horizon", "buy_threshold", "sell_threshold",
            "Sharpe Ratio (annualized)", "Total Return (%)", "Buy & Hold Return (%)",
            "Max Drawdown (%)", "Win Rate (%)"]
    print(results_df[cols].head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest config: --model_type {best['model_type']} --horizon {int(best['horizon'])} "
          f"--buy_threshold {best['buy_threshold']} --sell_threshold {best['sell_threshold']}")
    print("\nIMPORTANT: this searched many combinations against the SAME test period,")
    print("which means there's some risk of picking a config that got lucky on that")
    print("specific stretch of time, not one that's truly better. Treat the winner as")
    print("a promising candidate to paper-trade and re-check, not a proven answer.")

    out_path = f"tune_results_{args.ticker}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
