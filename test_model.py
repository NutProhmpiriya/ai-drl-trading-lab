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
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Calculate technical indicators
    df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['obv'] = calculate_obv(df)
    df['atr'] = calculate_atr(df, 14)
    
    # Remove NaN values
    df = df.dropna()
    
    return df

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV)"""
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['tick_volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['tick_volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['tick_volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def evaluate_model(model, env, n_episodes: int = 10):
    """Evaluate the trained model"""
    returns = []
    win_rate = 0
    trades = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        episode_trades = []
        
        while not (terminated or truncated):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
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
    data_path = "data/raw/USDJPY_5M_2023.csv"
    df = prepare_data(data_path)
    
    # Use test data (last 20% of data)
    train_size = int(len(df) * 0.8)
    test_df = df[train_size:]
    
    # Create test environment
    test_env = DummyVecEnv([lambda: ForexTradingEnv(
        df=test_df,
        initial_balance=100.0,
        leverage=1000.0,
        max_daily_drawdown=0.05
    )])
    
    # Load the latest model from rl_models directory
    models_dir = "rl_models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("forex_trading_model_")]
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
