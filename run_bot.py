"""
Automated decision cycle: scans a watchlist, checks risk limits FIRST, then
retrains the model fresh for each ticker and acts on the best signal via
Alpaca paper trading. Meant to be run once per trading day on a schedule.

Usage:
    python run_bot.py --tickers AAPL,MSFT,GOOGL,AMZN,SPY --trade_amount 5 \
        --model_type gb --horizon 5 --buy_threshold 0.02 --sell_threshold -0.02 \
        --stop_loss_pct 0.05 --take_profit_pct 0.10 --max_daily_loss_pct 0.03
"""
import argparse
import sys
from datetime import datetime

from data import fetch_data
from features import add_features, FEATURE_COLUMNS
from labels import add_labels
from model import train_final_model
import alpaca_trader as broker


def get_latest_signal(ticker, horizon, buy_threshold, sell_threshold, model_type="rf"):
    """Returns (signal, confidence_pct) for one ticker, or (None, None) if data can't be fetched."""
    try:
        df = fetch_data(ticker, period="5y")
    except Exception as e:
        print(f"  {ticker}: could not fetch data ({e}), skipping.")
        return None, None

    df = add_features(df)
    df_labeled = add_labels(df, horizon=horizon, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    df_model = df_labeled.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()

    if len(df_model) < 100:
        print(f"  {ticker}: not enough history, skipping.")
        return None, None

    model = train_final_model(df_model[FEATURE_COLUMNS], df_model["label"], model_type=model_type)

    latest_X = df.dropna(subset=FEATURE_COLUMNS)[FEATURE_COLUMNS].iloc[[-1]]
    pred = model.predict(latest_X)[0]
    proba = model.predict_proba(latest_X)[0]
    confidence = dict(zip(model.classes_, proba))[pred] * 100
    return pred, confidence


def run_cycle(tickers, trade_amount, horizon, buy_threshold, sell_threshold, max_positions,
              model_type="rf", stop_loss_pct=0.05, take_profit_pct=0.10, max_daily_loss_pct=0.03,
              position_size_pct=None):
    timestamp = datetime.now().isoformat(timespec="seconds")

    try:
        trading_client, is_paper = broker.get_trading_client()
    except RuntimeError as e:
        print(f"[{timestamp}] Setup error: {e}")
        sys.exit(1)

    mode = "PAPER (fake money)" if is_paper else "LIVE (REAL MONEY)"
    print(f"[{timestamp}] Mode: {mode}")

    if not broker.market_is_open(trading_client):
        print(f"[{timestamp}] Market is closed. No action taken.")
        for t in tickers:
            broker.log_decision(timestamp, t, "N/A", 0, "skipped_market_closed")
        return

    # --- Risk check #1: daily loss limit, checked before anything else ---
    daily_paused, daily_pnl_pct = broker.check_daily_loss_limit(trading_client, max_daily_loss_pct)
    if daily_paused:
        print(f"[{timestamp}] *** DAILY LOSS LIMIT HIT ({daily_pnl_pct*100:.1f}%, limit is "
              f"-{max_daily_loss_pct*100:.0f}%) *** -- no new BUYs today. "
              f"Existing positions are still protected by stop-loss below.")
    else:
        print(f"[{timestamp}] Daily P&L so far: {daily_pnl_pct*100:+.2f}% (limit: -{max_daily_loss_pct*100:.0f}%)")

    print(f"[{timestamp}] Scanning watchlist: {', '.join(tickers)}")
    signals = {}
    for ticker in tickers:
        signal, confidence = get_latest_signal(ticker, horizon, buy_threshold, sell_threshold, model_type)
        if signal is not None:
            signals[ticker] = (signal, confidence)
            print(f"  {ticker}: {signal} ({confidence:.1f}% confidence)")
        else:
            print(f"  {ticker}: FAILED to get a signal this cycle")
            broker.log_decision(timestamp, ticker, "N/A", 0, "fetch_failed",
                                 "Could not fetch/process data this cycle -- see console output if captured")

    account = broker.get_account_summary(trading_client)
    print(f"[{timestamp}] Account cash: ${account['cash']:.2f} | portfolio value: ${account['portfolio_value']:.2f}")

    # --- Handle existing positions: stop-loss / take-profit checked BEFORE the model's opinion ---
    held_count = 0
    for ticker in tickers:
        qty = broker.get_current_position(trading_client, ticker)
        if qty <= 0:
            continue
        held_count += 1
        signal, confidence = signals.get(ticker, ("UNKNOWN", 0))
        pos = broker.get_position_details(trading_client, ticker)
        pl_pct = pos["unrealized_plpc"] if pos else 0.0

        if pl_pct <= -abs(stop_loss_pct):
            order = broker.place_order(trading_client, ticker, "sell", qty=qty)
            broker.log_decision(timestamp, ticker, signal, confidence, "stop_loss_sold",
                                 f"pl_pct={pl_pct*100:.1f}%, qty={qty}, order_id={order.id}")
            print(f"[{timestamp}] STOP-LOSS triggered on {ticker} ({pl_pct*100:.1f}%) -- sold {qty} shares")
            held_count -= 1

        elif pl_pct >= abs(take_profit_pct):
            order = broker.place_order(trading_client, ticker, "sell", qty=qty)
            broker.log_decision(timestamp, ticker, signal, confidence, "take_profit_sold",
                                 f"pl_pct={pl_pct*100:.1f}%, qty={qty}, order_id={order.id}")
            print(f"[{timestamp}] TAKE-PROFIT triggered on {ticker} (+{pl_pct*100:.1f}%) -- sold {qty} shares")
            held_count -= 1

        elif signal == "SELL":
            order = broker.place_order(trading_client, ticker, "sell", qty=qty)
            broker.log_decision(timestamp, ticker, signal, confidence, "sold",
                                 f"qty={qty}, order_id={order.id}, realized_pl_pct={pl_pct*100:.1f}%")
            print(f"[{timestamp}] SOLD {qty} shares of {ticker} (model signal, P&L was {pl_pct*100:+.1f}%)")
            held_count -= 1

        else:
            broker.log_decision(timestamp, ticker, signal, confidence, "held_existing_position",
                                 f"pl_pct={pl_pct*100:.1f}%")

    # --- New BUYs: skipped entirely if the daily loss limit is active ---
    if daily_paused:
        print(f"[{timestamp}] Skipping new BUY evaluation -- daily loss limit is active.")
    else:
        open_slots = max_positions - held_count
        if open_slots > 0:
            buy_candidates = [
                (ticker, conf) for ticker, (sig, conf) in signals.items()
                if sig == "BUY" and broker.get_current_position(trading_client, ticker) == 0
            ]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)

            if buy_candidates:
                best_ticker, best_conf = buy_candidates[0]
                if position_size_pct is not None:
                    spend = account["cash"] * position_size_pct
                else:
                    spend = min(trade_amount, account["cash"])
                if spend >= 1:
                    order = broker.place_order(trading_client, best_ticker, "buy", notional=spend)
                    broker.log_decision(timestamp, best_ticker, "BUY", best_conf, "bought",
                                         f"notional=${spend:.2f}, order_id={order.id}")
                    print(f"[{timestamp}] BOUGHT ${spend:.2f} of {best_ticker} "
                          f"(best of {len(buy_candidates)} BUY signals) "
                          f"-- stop-loss at -{stop_loss_pct*100:.0f}%, take-profit at +{take_profit_pct*100:.0f}%")
                else:
                    print(f"[{timestamp}] BUY signal on {best_ticker} but insufficient cash (${account['cash']:.2f}).")
                    broker.log_decision(timestamp, best_ticker, "BUY", best_conf, "skipped_insufficient_cash")
            else:
                print(f"[{timestamp}] No new BUY candidates this cycle.")
                broker.log_decision(timestamp, "ALL", "N/A", 0, "no_action_no_candidates",
                                     f"scanned {len(signals)} tickers, no BUY signal and no existing position")
        else:
            print(f"[{timestamp}] Already holding max positions ({max_positions}), not opening new ones.")

    print(f"[{timestamp}] Cycle complete. See trade_log.csv for the full record.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one automated paper-trading decision cycle with risk management")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL,AMZN,SPY",
                         help="Comma-separated list of tickers to scan")
    parser.add_argument("--trade_amount", type=float, default=5.0, help="Dollar amount to spend on the best new BUY signal")
    parser.add_argument("--max_positions", type=int, default=1, help="Max number of tickers to hold at once")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--buy_threshold", type=float, default=0.02)
    parser.add_argument("--sell_threshold", type=float, default=-0.02)
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "gb"])
    parser.add_argument("--stop_loss_pct", type=float, default=0.05, help="Sell if a position drops this fraction below entry (0.05 = 5 percent)")
    parser.add_argument("--take_profit_pct", type=float, default=0.10, help="Sell if a position rises this fraction above entry (0.10 = 10 percent)")
    parser.add_argument("--max_daily_loss_pct", type=float, default=0.03, help="Pause new BUYs if the account is down this much today (0.03 = 3 percent)")
    parser.add_argument("--position_size_pct", type=float, default=None,
                         help="If set (0-1), spend this fraction of available cash per trade instead of a flat --trade_amount -- lets gains actually compound")
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    try:
        run_cycle(ticker_list, args.trade_amount, args.horizon, args.buy_threshold, args.sell_threshold,
                  args.max_positions, args.model_type, args.stop_loss_pct, args.take_profit_pct,
                  args.max_daily_loss_pct, args.position_size_pct)
    except Exception as e:
        import traceback
        crash_time = datetime.now().isoformat(timespec="seconds")
        print(f"[{crash_time}] CRASHED: {e}")
        traceback.print_exc()
        try:
            broker.log_decision(crash_time, "SYSTEM", "N/A", 0, "crashed", str(e))
        except Exception:
            pass  # even logging the crash failed -- nothing more we can safely do here
        raise
