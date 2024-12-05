import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_env.forex_env import ForexTradingEnv
from utils.indicators import add_indicators

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return add_indicators(df)

def run_backtest(model_path, data_path):
    # Load test data
    test_data = prepare_data(data_path)
    
    # Create environment
    env = ForexTradingEnv(
        df=test_data,
        initial_balance=10000,
        leverage=1000,
        max_daily_drawdown=0.05
    )
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Run backtest
    obs, _ = env.reset()
    done = False
    trades = []
    balance_history = [10000]
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        balance_history.append(env.balance)
        
        if env.position != 0:
            trades.append({
                'timestamp': test_data.index[env.current_step],
                'action': 'buy' if env.position > 0 else 'sell',
                'entry_price': env.entry_price,
                'exit_price': test_data.iloc[env.current_step]['close'],
                'position_size': abs(env.position),
                'profit_loss': reward
            })
    
    # Create trade report
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('src/backtest_report/trades.csv', index=False)
    
    # Calculate statistics
    total_trades = len(trades)
    profitable_trades = len([t for t in trades if t['profit_loss'] > 0])
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    total_profit = sum([t['profit_loss'] for t in trades])
    max_drawdown = min([b - max(balance_history[:i+1]) for i, b in enumerate(balance_history)])
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Balance History
    plt.subplot(2, 1, 1)
    plt.plot(balance_history)
    plt.title('Account Balance History')
    plt.xlabel('Steps')
    plt.ylabel('Balance')
    
    # Plot 2: Trade Distribution
    plt.subplot(2, 1, 2)
    profits = [t['profit_loss'] for t in trades]
    sns.histplot(profits, bins=50)
    plt.title('Trade Profit/Loss Distribution')
    plt.xlabel('Profit/Loss')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('src/backtest_report/backtest_results.png')
    
    # Print statistics
    print(f"\nBacktest Results:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Max Drawdown: ${abs(max_drawdown):.2f}")
    print(f"Final Balance: ${balance_history[-1]:.2f}")

if __name__ == "__main__":
    model_path = "src/rl_model/forex_trading_model"
    test_data_path = "src/data/raw/USDJPY_5M_2024.csv"
    run_backtest(model_path, test_data_path)
