import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_env.forex_env import ForexTradingEnv
from utils.indicators import add_indicators

# Load and prepare data
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return add_indicators(df)

# Training parameters
TOTAL_TIMESTEPS = 100000
LEARNING_RATE = 0.0001
INITIAL_BALANCE = 10000
LEVERAGE = 1000
MAX_DAILY_DRAWDOWN = 0.05

def train_model():
    # Load training data
    train_data = prepare_data('src/data/raw/USDJPY_5M_2023.csv')
    
    # Create and wrap the environment
    env = ForexTradingEnv(
        df=train_data,
        initial_balance=INITIAL_BALANCE,
        leverage=LEVERAGE,
        max_daily_drawdown=MAX_DAILY_DRAWDOWN
    )
    env = DummyVecEnv([lambda: env])
    
    # Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        batch_size=64,
        ent_coef=0.01,
        verbose=1
    )
    
    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save the trained model
    model.save("src/rl_model/forex_trading_model")

if __name__ == "__main__":
    train_model()
