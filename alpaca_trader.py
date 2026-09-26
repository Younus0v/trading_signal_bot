"""
Alpaca paper-trading integration.

SAFETY: this module is hardcoded to paper trading (paper=True) unless you
explicitly change ALPACA_PAPER=false in your environment AND acknowledge the
warning in run_bot.py. Paper trading uses fake money and real-time real
market data -- it is the right place to run this for a long time before ever
considering live money.

You will need free Alpaca API keys: https://alpaca.markets (sign up, create
a paper trading account, generate an API key + secret from the dashboard).
Never hardcode keys in this file -- set them as environment variables:
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"
"""
import os
import csv
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "trade_log.csv"
STATE_PATH = Path(__file__).parent / "bot_state.json"


def get_trading_client():
    from alpaca.trading.client import TradingClient

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables. "
            "Get free paper trading keys at https://alpaca.markets"
        )

    is_paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
    return TradingClient(api_key, secret_key, paper=is_paper), is_paper


def market_is_open(trading_client):
    clock = trading_client.get_clock()
    return clock.is_open


def get_account_summary(trading_client):
    account = trading_client.get_account()
    return {
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
    }


def get_current_position(trading_client, symbol):
    """Returns qty held (float) or 0.0 if no position."""
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0


def get_position_details(trading_client, symbol):
    """Returns {qty, avg_entry_price, unrealized_plpc} or None if no position.
    unrealized_plpc is a fraction, e.g. 0.05 = +5%, -0.03 = -3%."""
    try:
        position = trading_client.get_open_position(symbol)
        return {
            "qty": float(position.qty),
            "avg_entry_price": float(position.avg_entry_price),
            "unrealized_plpc": float(position.unrealized_plpc),
        }
    except Exception:
        return None


def load_daily_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_daily_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def check_daily_loss_limit(trading_client, max_daily_loss_pct):
    """
    Tracks portfolio value at the start of each calendar day. Returns
    (paused, daily_pnl_pct). `paused` is True if today's loss has already
    breached max_daily_loss_pct, meaning no new BUYs should be placed
    (existing positions are still protected separately by stop-loss checks).
    """
    today = datetime.now().date().isoformat()
    state = load_daily_state()
    account = get_account_summary(trading_client)
    current_equity = account["portfolio_value"]

    if state.get("date") != today:
        state = {"date": today, "start_of_day_equity": current_equity}
        save_daily_state(state)
        return False, 0.0

    start_equity = state.get("start_of_day_equity", current_equity)
    daily_pnl_pct = (current_equity - start_equity) / start_equity if start_equity > 0 else 0.0
    paused = daily_pnl_pct <= -abs(max_daily_loss_pct)
    return paused, daily_pnl_pct


def place_order(trading_client, symbol, side, notional=None, qty=None):
    """
    side: "buy" or "sell"
    Use `notional` (dollar amount) for fractional buys -- ideal for small accounts.
    Use `qty` to sell an exact share amount (e.g. closing a whole position).
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    if notional is not None:
        order_data = MarketOrderRequest(
            symbol=symbol, notional=round(notional, 2),
            side=order_side, time_in_force=TimeInForce.DAY,
        )
    else:
        order_data = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=order_side, time_in_force=TimeInForce.DAY,
        )

    return trading_client.submit_order(order_data)


def log_decision(timestamp, symbol, signal, confidence, action_taken, details=""):
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "symbol", "model_signal", "confidence_pct", "action_taken", "details"])
        writer.writerow([timestamp, symbol, signal, round(confidence, 1), action_taken, details])
