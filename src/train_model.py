import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime
import pandas_ta as ta
import torch.nn as nn

from rl_env.forex_env import ForexTradingEnv
from rl_agent import DRLAgent

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    # Calculate EMAs
    df['ema7'] = ta.ema(df['close'], length=7)
    df['ema21'] = ta.ema(df['close'], length=21)
    
    # Calculate RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # Calculate OBV
    df['obv'] = ta.obv(df['close'], df['tick_volume'])
    
    # Calculate ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df

def prepare_data(file_path: str) -> pd.DataFrame:
    """Prepare and preprocess the data"""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Calculate technical indicators
    df = calculate_indicators(df)
    
    # Remove NaN values
    df = df.dropna()
    
    return df

def train_model(env, total_timesteps: int = 100000, save_path: str = None):
    """Train the RL model"""
    # Create and train DRL agent
    agent = DRLAgent(env)
    agent.train(total_timesteps=total_timesteps)
    
    # Save the trained model
    if save_path:
        agent.save(save_path)
    
    return agent

def main():
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_log", exist_ok=True)
    
    # Load and prepare data
    data_path = "data/raw/USDJPY_5M_2023.csv"
    df = prepare_data(data_path)
    
    # Check for NaN values
    print("\nChecking for NaN values:")
    print(df.isna().sum())
    
    # Print data info
    print("\nData info:")
    print(df.info())
    
    # Print first few rows
    print("\nFirst few rows:")
    print(df.head())
    
    # Print data statistics
    print("\nData statistics:")
    print(df.describe())
    
    # Split data into train and test sets (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    
    # Create training environment
    train_env = DummyVecEnv([
        lambda: ForexTradingEnv(
            df=train_df,
            initial_balance=100.0,
            leverage=1000.0,
            max_daily_drawdown=0.05
        )
    ])
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/trading_model_{timestamp}"
    
    print("Starting training...")
    agent = train_model(
        env=train_env,
        total_timesteps=100000,
        save_path=model_path
    )
    
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
