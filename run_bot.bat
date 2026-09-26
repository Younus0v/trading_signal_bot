@echo off
cd /d C:\trading_signal_bot
echo. >> run_output.log
echo ===== Run started: %date% %time% ===== >> run_output.log
C:\Users\dagoo\AppData\Local\Python\bin\python.exe run_bot.py --tickers AAPL,MSFT,GOOGL,AMZN,XOM --trade_amount 5 --model_type gb --horizon 5 --buy_threshold 0.02 --sell_threshold -0.02 --stop_loss_pct 0.05 --take_profit_pct 0.10 --max_daily_loss_pct 0.03 >> run_output.log 2>&1
echo ===== Run finished: %date% %time% ===== >> run_output.log
