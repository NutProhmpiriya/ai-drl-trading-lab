import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from datetime import datetime

class ForexTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100, leverage=1000, max_daily_drawdown=0.05):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_daily_drawdown = max_daily_drawdown
        self.current_step = 0
        self.daily_start_balance = initial_balance
        self.last_trade_date = None
        
        # Calculate min and max values for normalization
        price_cols = ['open', 'high', 'low', 'close']
        self.price_min = df[price_cols].min().min()
        self.price_max = df[price_cols].max().max()
        self.volume_min = df['tick_volume'].min()
        self.volume_max = df['tick_volume'].max()
        self.indicator_mins = {
            'ema7': df['ema7'].min(),
            'ema21': df['ema21'].min(),
            'rsi': df['rsi'].min(),
            'obv': df['obv'].min(),
            'atr': df['atr'].min()
        }
        self.indicator_maxs = {
            'ema7': df['ema7'].max(),
            'ema21': df['ema21'].max(),
            'rsi': df['rsi'].max(),
            'obv': df['obv'].max(),
            'atr': df['atr'].max()
        }
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, position, price_features, technical_indicators]
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(13,),  # [balance, position, OHLCV, ema7, ema21, rsi, obv, atr]
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.position = 0  # No position
        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.last_trade_date = None
        self.daily_start_balance = self.initial_balance
        
        return self._get_observation(), {}  # Return observation and info dict
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}
        reward = 0  # Initialize reward
        
        # Get current price data
        current_price = self.df['close'].iloc[self.current_step]
        next_price = self.df['close'].iloc[self.current_step + 1] if self.current_step + 1 < len(self.df) else current_price
        atr = self.df['atr'].iloc[self.current_step]
        
        # Store previous balance for reward calculation
        self.prev_balance = self.balance
        
        # Check if episode should end (end of data)
        if self.current_step >= len(self.df) - 1:
            terminated = True
            return self._get_observation(), reward, terminated, truncated, info
        
        # Calculate position size based on ATR for risk management
        risk_amount = self.balance * 0.01  # Risk 1% per trade
        position_size = (risk_amount / atr) * self.leverage
        
        # Entry conditions
        ema_cross = self.df['ema7'].iloc[self.current_step] > self.df['ema21'].iloc[self.current_step]
        rsi = self.df['rsi'].iloc[self.current_step]
        obv_trend = self.df['obv'].iloc[self.current_step] > self.df['obv'].iloc[self.current_step - 1] if self.current_step > 0 else False
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position <= 0 and ema_cross and rsi > 40 and obv_trend:
                self.position = position_size
                self.entry_price = current_price
                self.stop_loss = current_price - 1.5 * atr
                self.take_profit = current_price + 2.5 * atr
                info['trade_info'] = {
                    'action': 'buy',
                    'position_type': 'long',
                    'entry_price': self.entry_price,
                    'entry_time': self.df.index[self.current_step],
                    'stop_loss': self.stop_loss,
                    'take_profit': self.take_profit
                }
                
        elif action == 2:  # Sell
            if self.position >= 0 and not ema_cross and rsi < 60 and not obv_trend:
                self.position = -position_size
                self.entry_price = current_price
                self.stop_loss = current_price + 1.5 * atr
                self.take_profit = current_price - 2.5 * atr
                info['trade_info'] = {
                    'action': 'sell',
                    'position_type': 'short',
                    'entry_price': self.entry_price,
                    'entry_time': self.df.index[self.current_step],
                    'stop_loss': self.stop_loss,
                    'take_profit': self.take_profit
                }
        
        # Calculate reward based on position and price movement
        if self.position != 0:
            price_change = (next_price - current_price) / current_price
            reward = self._calculate_reward()
            
            # Check for SL/TP
            if self.position > 0:  # Long position
                if next_price <= self.stop_loss:
                    reward = -1  # Fixed penalty for stop loss
                    self.position = 0
                    info['trade_info'] = {
                        'action': 'stop_loss',
                        'position_type': 'long',
                        'entry_price': self.entry_price,
                        'exit_price': next_price,
                        'pnl': reward,
                        'entry_time': self.df.index[self.current_step - 1],
                        'exit_time': self.df.index[self.current_step],
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    }
                elif next_price >= self.take_profit:
                    reward = 1.5  # Fixed reward for take profit
                    self.position = 0
                    info['trade_info'] = {
                        'action': 'take_profit',
                        'position_type': 'long',
                        'entry_price': self.entry_price,
                        'exit_price': next_price,
                        'pnl': reward,
                        'entry_time': self.df.index[self.current_step - 1],
                        'exit_time': self.df.index[self.current_step],
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    }
                    
                # Update trailing stop if in profit
                elif (next_price - self.entry_price) >= atr:
                    new_stop = next_price - atr
                    if new_stop > self.stop_loss:
                        self.stop_loss = new_stop
                        
            else:  # Short position
                if next_price >= self.stop_loss:
                    reward = -1
                    self.position = 0
                    info['trade_info'] = {
                        'action': 'stop_loss',
                        'position_type': 'short',
                        'entry_price': self.entry_price,
                        'exit_price': next_price,
                        'pnl': reward,
                        'entry_time': self.df.index[self.current_step - 1],
                        'exit_time': self.df.index[self.current_step],
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    }
                elif next_price <= self.take_profit:
                    reward = 1.5
                    self.position = 0
                    info['trade_info'] = {
                        'action': 'take_profit',
                        'position_type': 'short',
                        'entry_price': self.entry_price,
                        'exit_price': next_price,
                        'pnl': reward,
                        'entry_time': self.df.index[self.current_step - 1],
                        'exit_time': self.df.index[self.current_step],
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    }
                    
                # Update trailing stop if in profit
                elif (self.entry_price - next_price) >= atr:
                    new_stop = next_price + atr
                    if new_stop < self.stop_loss:
                        self.stop_loss = new_stop
            
            # Update balance
            self.balance += reward * risk_amount
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current step"""
        # Get the current balance and previous balance
        current_balance = self.balance
        prev_balance = self.prev_balance
        
        # Calculate profit/loss
        profit_loss = current_balance - prev_balance
        
        # Calculate drawdown
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        
        # Base reward is the normalized profit/loss
        reward = profit_loss / self.initial_balance
        
        # Apply penalties
        if drawdown >= self.max_daily_drawdown:
            # Severe penalty for exceeding max drawdown
            reward = -1.0
        else:
            # Scale reward based on drawdown
            drawdown_penalty = (drawdown / self.max_daily_drawdown) ** 2
            reward = reward * (1 - drawdown_penalty)
        
        # Add small time penalty to encourage faster trading
        time_penalty = -0.0001
        reward += time_penalty
        
        # Add exploration bonus for taking positions
        if self.position != 0:
            # Small positive reward for having an open position
            position_bonus = 0.0001
            reward += position_bonus
        else:
            # Small negative reward for not having a position
            inaction_penalty = -0.0001
            reward += inaction_penalty
        
        # Clip reward to [-1, 1] range
        reward = np.clip(reward, -1.0, 1.0)
        
        return float(reward)
    
    def _get_observation(self):
        # Check if current_step is valid
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1  # Keep at last valid index
            
        current_data = self.df.iloc[self.current_step]
        
        # Helper function to safely normalize data
        def safe_normalize(value, min_val, max_val, clip=True):
            if np.isclose(min_val, max_val):
                return 0.0
            normalized = (value - min_val) / (max_val - min_val) * 2 - 1
            if clip:
                normalized = np.clip(normalized, -1.0, 1.0)
            return normalized
        
        # Normalize the data
        normalized_balance = np.clip(self.balance / (self.initial_balance * 2) - 1, -1.0, 1.0)
        normalized_position = np.clip(self.position / (self.initial_balance * self.leverage), -1.0, 1.0)
        
        # Normalize price data
        normalized_open = safe_normalize(current_data['open'], self.price_min, self.price_max)
        normalized_high = safe_normalize(current_data['high'], self.price_min, self.price_max)
        normalized_low = safe_normalize(current_data['low'], self.price_min, self.price_max)
        normalized_close = safe_normalize(current_data['close'], self.price_min, self.price_max)
        
        # Normalize volume
        normalized_volume = safe_normalize(current_data['tick_volume'], self.volume_min, self.volume_max)
        
        # Normalize technical indicators
        normalized_ema7 = safe_normalize(current_data['ema7'], self.indicator_mins['ema7'], self.indicator_maxs['ema7'])
        normalized_ema21 = safe_normalize(current_data['ema21'], self.indicator_mins['ema21'], self.indicator_maxs['ema21'])
        normalized_rsi = safe_normalize(current_data['rsi'], self.indicator_mins['rsi'], self.indicator_maxs['rsi'])
        normalized_obv = safe_normalize(current_data['obv'], self.indicator_mins['obv'], self.indicator_maxs['obv'])
        normalized_atr = safe_normalize(current_data['atr'], self.indicator_mins['atr'], self.indicator_maxs['atr'])
        
        # Normalize daily PnL
        normalized_daily_pnl = np.clip((self.balance - self.daily_start_balance) / self.initial_balance, -1.0, 1.0)
        
        obs = np.array([
            normalized_balance,
            normalized_position,
            normalized_open,
            normalized_high,
            normalized_low,
            normalized_close,
            normalized_volume,
            normalized_ema7,
            normalized_ema21,
            normalized_rsi,
            normalized_obv,
            normalized_atr,
            normalized_daily_pnl
        ], dtype=np.float32)
        
        return obs
