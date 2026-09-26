"""
ML Buy/Sell/Hold signal system -- an educational/research tool, NOT a
guaranteed-profit machine. Read the README before using this with real money.

Usage:
    python main.py --ticker SYNTHETIC          # test the pipeline, no internet needed
    python main.py --ticker AAPL --period 5y   # real data (needs `pip install yfinance`)
"""
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from data import fetch_data, generate_synthetic_data
from features import add_features, FEATURE_COLUMNS
from labels import add_labels
from model import walk_forward_validate, train_final_model
from backtest import run_backtest, performance_summary


def main():
    parser = argparse.ArgumentParser(description="ML Buy/Sell/Hold signal system")
    parser.add_argument("--ticker", type=str, default="SYNTHETIC", help="Stock ticker, e.g. AAPL (or SYNTHETIC to test without internet)")
    parser.add_argument("--period", type=str, default="5y", help="History period for yfinance, e.g. 2y, 5y, max")
    parser.add_argument("--horizon", type=int, default=5, help="Trading days ahead used to define the label")
    parser.add_argument("--buy_threshold", type=float, default=0.02, help="Forward return above this -> BUY label")
    parser.add_argument("--sell_threshold", type=float, default=-0.02, help="Forward return below this -> SELL label")
    parser.add_argument("--train_frac", type=float, default=0.75, help="Fraction of history used for training; the rest is held out")
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "gb"], help="rf = Random Forest, gb = Gradient Boosting")
    args = parser.parse_args()

    print(f"\n=== Loading data: {args.ticker} ===")
    if args.ticker.upper() == "SYNTHETIC":
        df = generate_synthetic_data()
        print("Using SYNTHETIC data -- for testing the pipeline only, not real market data.")
    else:
        df = fetch_data(args.ticker, period=args.period)
    print(f"{len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

    print("\n=== Building features & labels ===")
    df = add_features(df)
    df = add_labels(df, horizon=args.horizon, buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)
    df_model = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
    print(f"Usable rows after indicator warm-up: {len(df_model)}")
    print("Label distribution:\n" + df_model["label"].value_counts().to_string())

    # Chronological split -- the test period is data the model NEVER sees during training.
    split_idx = int(len(df_model) * args.train_frac)
    train_df, test_df = df_model.iloc[:split_idx], df_model.iloc[split_idx:]
    print(f"\nTrain period: {train_df.index[0].date()} -> {train_df.index[-1].date()}  ({len(train_df)} rows)")
    print(f"Test period (out-of-sample, unseen by the model): {test_df.index[0].date()} -> {test_df.index[-1].date()}  ({len(test_df)} rows)")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    print("\n=== Walk-forward validation within training data ===")
    walk_forward_validate(X_train, y_train, n_splits=5, model_type=args.model_type)

    print(f"\n=== Training {args.model_type.upper()} model on training data only ===")
    model = train_final_model(X_train, y_train, model_type=args.model_type)

    test_preds = model.predict(X_test)
    print(f"\nOut-of-sample accuracy (the honest number, not the training accuracy): {accuracy_score(y_test, test_preds):.3f}")
    print(classification_report(y_test, test_preds, zero_division=0))

    print("=== Backtest on the out-of-sample test period ===")
    signals = pd.Series(test_preds, index=X_test.index)
    equity_df, trades_df = run_backtest(test_df, signals)
    summary = performance_summary(equity_df, trades_df, test_df)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== Latest signal (model retrained on ALL available data through today) ===")
    X_full, y_full = df_model[FEATURE_COLUMNS], df_model["label"]
    full_model = train_final_model(X_full, y_full, model_type=args.model_type)
    latest_X = df.dropna(subset=FEATURE_COLUMNS)[FEATURE_COLUMNS].iloc[[-1]]
    latest_pred = full_model.predict(latest_X)[0]
    latest_proba = full_model.predict_proba(latest_X)[0]
    print(f"Date: {latest_X.index[-1].date()}")
    print(f"Signal: {latest_pred}")
    for cls, p in zip(full_model.classes_, latest_proba):
        print(f"  {cls}: {p*100:.1f}% model confidence")

    print("\nThis is a research signal generated from historical patterns. It is not")
    print("financial advice and the out-of-sample results above -- not the training")
    print("accuracy -- are the number to trust (or distrust). See README.md.")


if __name__ == "__main__":
    main()
