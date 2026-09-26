"""
Pooled multi-stock training.

The AAPL/MSFT test showed the real problem: a model tuned to one stock's
specific history can fail completely on a different stock. This script
trains ONE model across many stocks at once, so it has to find patterns
that hold up broadly -- not quirks of a single company's particular path.

This is a legitimate way to build a more ROBUST model. It is still not a
route to 80%+ accuracy -- nothing is, for the reasons already discussed.
What this can realistically deliver: a model that performs consistently
(even if modestly) across many stocks, which is worth far more than a
model that looks amazing on one stock and falls apart on the next.

Usage:
    python train_pooled.py --model_type gb --horizon 5
"""
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from data import fetch_data
from features import add_features, FEATURE_COLUMNS
from labels import add_labels, add_triple_barrier_labels
from model import train_final_model
from backtest import run_backtest, performance_summary

DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # tech
    "JNJ", "UNH",                                      # healthcare
    "COST", "WMT", "PG", "KO", "HD",                   # consumer
    "XOM",                                             # energy
    "DIS",                                             # media
]


def build_pooled_dataset(tickers, label_method, horizon, buy_threshold, sell_threshold,
                          take_profit_pct, stop_loss_pct, max_holding_days, train_frac=0.75):
    train_frames, test_frames = [], []
    for ticker in tickers:
        try:
            df = fetch_data(ticker, period="5y")
        except Exception as e:
            print(f"  {ticker}: skipped ({e})")
            continue
        df = add_features(df)
        if label_method == "triple_barrier":
            df = add_triple_barrier_labels(df, take_profit_pct=take_profit_pct,
                                            stop_loss_pct=stop_loss_pct, max_holding_days=max_holding_days)
        else:
            df = add_labels(df, horizon=horizon, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        df_model = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
        if len(df_model) < 200:
            print(f"  {ticker}: skipped (not enough history)")
            continue
        df_model["ticker"] = ticker

        split_idx = int(len(df_model) * train_frac)
        train_frames.append(df_model.iloc[:split_idx])
        test_frames.append(df_model.iloc[split_idx:])
        print(f"  {ticker}: {len(df_model)} usable rows")

    if not train_frames:
        raise RuntimeError("No usable data for any ticker -- check your internet connection and ticker symbols.")

    return pd.concat(train_frames), pd.concat(test_frames)


def main():
    parser = argparse.ArgumentParser(description="Train one model pooled across many stocks for more robust patterns")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_BASKET),
                         help="Comma-separated basket of stocks to pool together")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--buy_threshold", type=float, default=0.02)
    parser.add_argument("--sell_threshold", type=float, default=-0.02)
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "gb"])
    parser.add_argument("--label_method", type=str, default="threshold", choices=["threshold", "triple_barrier"],
                         help="threshold = simple forward-return labels, triple_barrier = simulated take-profit/stop-loss outcome")
    parser.add_argument("--take_profit_pct", type=float, default=0.03, help="Triple-barrier: take-profit level (0.03 = 3 percent)")
    parser.add_argument("--stop_loss_pct", type=float, default=0.02, help="Triple-barrier: stop-loss level (0.02 = 2 percent)")
    parser.add_argument("--max_holding_days", type=int, default=10, help="Triple-barrier: max days to wait for either barrier")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"=== Building pooled dataset across {len(tickers)} stocks (labels: {args.label_method}) ===")
    train_df, test_df = build_pooled_dataset(
        tickers, args.label_method, args.horizon, args.buy_threshold, args.sell_threshold,
        args.take_profit_pct, args.stop_loss_pct, args.max_holding_days,
    )
    print(f"\nPooled training rows: {len(train_df)}   Pooled test rows: {len(test_df)}")
    print("Pooled training label distribution:\n" + train_df["label"].value_counts().to_string())

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    print(f"\n=== Training ONE {args.model_type.upper()} model on all {len(tickers)} stocks pooled together ===")
    model = train_final_model(X_train, y_train, model_type=args.model_type)

    preds = model.predict(X_test)
    print(f"\nOut-of-sample accuracy across ALL pooled test data: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    print("=== Per-ticker out-of-sample backtest, using the SAME single model ===")
    results = []
    for ticker in tickers:
        ticker_test = test_df[test_df["ticker"] == ticker]
        if len(ticker_test) < 20:
            continue
        ticker_preds = model.predict(ticker_test[FEATURE_COLUMNS])
        signals = pd.Series(ticker_preds, index=ticker_test.index)
        equity_df, trades_df = run_backtest(ticker_test, signals)
        summary = performance_summary(equity_df, trades_df, ticker_test)
        summary["ticker"] = ticker
        results.append(summary)
        print(f"  {ticker:6s} Sharpe={summary['Sharpe Ratio (annualized)']:6.2f}  "
              f"Return={summary['Total Return (%)']:7.1f}%  "
              f"BuyHold={summary['Buy & Hold Return (%)']:7.1f}%  "
              f"Trades={summary['Number of Completed Trades']}")

    if not results:
        print("No tickers had enough out-of-sample data to report.")
        return

    results_df = pd.DataFrame(results)
    avg_sharpe = results_df["Sharpe Ratio (annualized)"].mean()
    beat_pct = (results_df["Total Return (%)"] > results_df["Buy & Hold Return (%)"]).mean() * 100

    print(f"\n=== Summary across {len(results_df)} stocks ===")
    print(f"Average Sharpe across all stocks: {avg_sharpe:.2f}")
    print(f"Beat buy-and-hold on {beat_pct:.0f}% of stocks")
    print("\nThis is the real, honest test: one model, trained once, checked across many")
    print("different stocks it wasn't individually tuned for. A model that's mildly")
    print("useful on most stocks is more trustworthy than one that's spectacular on a")
    print("single stock and falls apart on the next -- which is what we just saw happen")
    print("with the AAPL-tuned config on MSFT.")

    results_df.to_csv("pooled_results.csv", index=False)
    print("\nFull per-ticker results saved to pooled_results.csv")


if __name__ == "__main__":
    main()
