import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from datetime import datetime

class ForexTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, leverage=1000, max_daily_drawdown=0.05):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_daily_drawdown = max_daily_drawdown
        self.current_step = 0
        self.daily_start_balance = initial_balance
        self.last_trade_date = None
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, position, price_features, technical_indicators]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(28,),  # [balance, position, OHLCV, EMA fast/slow, RSI, OBV, ATR]
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.position = 0
        self.last_trade_date = None
        self.history = []
        return self._get_observation(), {}
    
    def step(self, action):
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']
        current_date = pd.to_datetime(current_data.name).date()
        reward = 0
        done = False
        
        # Check for new trading day
        if self.last_trade_date is not None and current_date != self.last_trade_date:
            self.daily_start_balance = self.balance
        
        # Calculate position size based on ATR for risk management
        atr = current_data['atr']
        risk_amount = self.balance * 0.01  # Risk 1% per trade
        position_size = (risk_amount / atr) * self.leverage
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position <= 0:
                self.position = position_size
                self.entry_price = current_price
                self.stop_loss = current_price - 1.5 * atr  # 1.5 * ATR for stop loss
                self.take_profit = current_price + 2.25 * atr  # 1.5 * 1.5 * ATR for take profit (1:1.5 ratio)
                reward = -current_price * 0.0001  # Transaction cost
                
        elif action == 2:  # Sell
            if self.position >= 0:
                self.position = -position_size
                self.entry_price = current_price
                self.stop_loss = current_price + 1.5 * atr
                self.take_profit = current_price - 2.25 * atr
                reward = -current_price * 0.0001
        
        # Move to next step
        self.current_step += 1
        
        # Check if we have next data point
        if self.current_step < len(self.df):
            next_data = self.df.iloc[self.current_step]
            next_price = next_data['close']
            
            # Calculate reward based on position and price movement
            if self.position != 0:
                price_change = (next_price - current_price)
                reward = self.position * price_change
                
                # Check for SL/TP
                if self.position > 0:  # Long position
                    if next_price <= self.stop_loss or next_price >= self.take_profit:
                        reward = self.position * (next_price - self.entry_price)
                        self.position = 0
                else:  # Short position
                    if next_price >= self.stop_loss or next_price <= self.take_profit:
                        reward = self.position * (self.entry_price - next_price)
                        self.position = 0
            
            # Update balance
            self.balance += reward
            
            # Check daily drawdown limit
            daily_drawdown = (self.balance - self.daily_start_balance) / self.daily_start_balance
            if daily_drawdown <= -self.max_daily_drawdown:
                done = True
                reward -= 100  # Penalty for exceeding daily drawdown limit
            
        else:
            done = True
        
        self.last_trade_date = current_date
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        # Get current market data
        obs = self.df.iloc[self.current_step]
        
        # Combine position, balance, and market data
        return np.array([
            self.balance,
            self.position,
            obs['open'],
            obs['high'],
            obs['low'],
            obs['close'],
            obs['tick_volume'],
            obs['ema_fast'],
            obs['ema_slow'],
            obs['rsi'],
            obs['obv'],
            obs['atr']
        ], dtype=np.float32)
