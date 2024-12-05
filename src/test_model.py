import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime

from rl_env.forex_env import ForexTradingEnv

def prepare_data(file_path: str) -> pd.DataFrame:
    """Prepare and preprocess the data"""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def evaluate_model(model, env, n_episodes: int = 10):
    """Evaluate the trained model"""
    returns = []
    win_rate = 0
    trades = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_trades = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
            if 'trade_info' in info:
                episode_trades.append(info['trade_info'])
        
        returns.append(total_reward)
        if total_reward > 0:
            win_rate += 1
        trades.extend(episode_trades)
    
    win_rate = win_rate / n_episodes * 100
    
    # Print evaluation results
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Average return: {np.mean(returns):.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Standard deviation: {np.std(returns):.2f}")
    
    # Create trades DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f"backtest_report/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"\nTrade statistics:")
        print(f"Total trades: {len(trades_df)}")
        print(f"Win rate: {(trades_df['pnl'] > 0).mean():.2%}")
        print(f"Average profit: ${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}")
        print(f"Average loss: ${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}")
        print(f"Profit factor: {abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()):.2f}")
        print(f"Max drawdown: ${trades_df['pnl'].cumsum().min():.2f}")

def main():
    # Create directories if they don't exist
    os.makedirs("backtest_report", exist_ok=True)
    
    # Load and prepare data
    data_path = "../data/raw/USDJPY_5M_2023.csv"
    df = prepare_data(data_path)
    
    # Use test data (last 20% of data)
    train_size = int(len(df) * 0.8)
    test_df = df[train_size:]
    
    # Create test environment
    test_env = DummyVecEnv([
        lambda: ForexTradingEnv(
            df=test_df,
            initial_balance=100.0,
            leverage=1000.0,
            max_daily_drawdown=0.05
        )
    ])
    
    # Load the latest model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("trading_model_")]
    if not model_files:
        print("No trained models found!")
        return
    
    latest_model = max(model_files)
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    print("\nStarting backtesting...")
    evaluate_model(model, test_env)

if __name__ == "__main__":
    main()
