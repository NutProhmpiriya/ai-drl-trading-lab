import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class ForexTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, position, price_features]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(23,),  # Adjust based on your features
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.history = []
        return self._get_observation(), {}
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        done = False
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position <= 0:
                self.position = 1
                reward = -current_price * 0.0001  # Transaction cost
        elif action == 2:  # Sell
            if self.position >= 0:
                self.position = -1
                reward = -current_price * 0.0001  # Transaction cost
                
        # Move to next step
        self.current_step += 1
        
        # Calculate reward based on position and price movement
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]['Close']
            price_change = (next_price - current_price) / current_price
            reward += self.position * price_change * self.balance
            
            # Update balance
            self.balance += reward
        else:
            done = True
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        # Get current market data
        obs = self.df.iloc[self.current_step]
        
        # Combine position, balance, and market data
        return np.array([
            self.balance,
            self.position,
            *obs.values
        ], dtype=np.float32)
