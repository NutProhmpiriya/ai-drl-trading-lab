# AI DRL Trading System Concept

## Overview
This system implements a Deep Reinforcement Learning (DRL) approach for automated forex trading, specifically designed for USDJPY on the 5-minute timeframe. The system uses the Proximal Policy Optimization (PPO) algorithm to learn optimal trading strategies while maintaining strict risk management rules.

## System Architecture

### 1. Trading Environment (forex_env.py)
The trading environment is built using OpenAI Gymnasium framework and implements:

#### State Space
- Account balance
- Current position
- Market data (OHLCV)
- Technical indicators:
  - EMA (fast/slow)
  - RSI
  - OBV
  - ATR

#### Action Space
- 0: Hold
- 1: Buy
- 2: Sell

#### Reward System
- Based on realized profit/loss
- Includes transaction costs
- Penalties for exceeding risk limits
- Rewards scaled by position size

### 2. Risk Management

#### Position Sizing
- Base risk: 1% per trade
- Position size calculation: `(risk_amount / ATR) * leverage`
- Leverage: 1:1000

#### Stop Loss & Take Profit
- Stop Loss: 1.5 * ATR
- Take Profit: 2.25 * ATR (1:1.5 risk-reward ratio)

#### Daily Risk Control
- Maximum daily drawdown: 5%
- Position closure on drawdown limit breach
- Daily balance reset tracking

### 3. Technical Analysis (indicators.py)

#### Moving Averages
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Used for trend direction

#### Relative Strength Index (RSI)
- Period: 14
- Overbought level: 70
- Oversold level: 30
- Used for momentum analysis

#### On-Balance Volume (OBV)
- Cumulative volume indicator
- Used for volume trend confirmation

#### Average True Range (ATR)
- Period: 14
- Used for volatility measurement
- Position sizing and stop level calculation

### 4. Model Training (train_model.py)

#### Data Preparation
- Historical 5-minute USDJPY data
- Feature engineering with technical indicators
- Data normalization

#### Training Process
- PPO algorithm implementation
- Batch size: 64
- Learning rate: 0.0001
- Entropy coefficient: 0.01
- Total timesteps: 100,000

### 5. Backtesting (backtest.py)

#### Performance Metrics
- Total return
- Win rate
- Maximum drawdown
- Profit factor
- Average win/loss
- Trade duration statistics

#### Visualization
- Balance curve
- Trade distribution
- Profit/Loss histogram
- Technical indicator signals

## Trading Logic

### Entry Conditions
The DRL agent learns to recognize optimal entry conditions based on:
1. Trend direction (EMA crossovers)
2. Momentum (RSI)
3. Volume confirmation (OBV)
4. Volatility context (ATR)

### Exit Conditions
Positions are closed when:
1. Take profit level reached (2.25 * ATR)
2. Stop loss hit (1.5 * ATR)
3. Daily drawdown limit reached (5%)
4. Opposing signal from DRL agent

### Risk Management Rules
1. Maximum 1% risk per trade
2. Position sizing based on ATR
3. 5% maximum daily drawdown
4. No overnight positions
5. Leverage limit at 1:1000

## Implementation Steps

1. Data Preparation
   ```python
   python src/scripts/prepare_data.py
   ```

2. Model Training
   ```python
   python src/scripts/train_model.py
   ```

3. Backtesting
   ```python
   python src/scripts/backtest.py
   ```

## Performance Analysis

### Trading Reports
- Detailed trade log (CSV format)
- Daily performance summary
- Risk metrics analysis
- Equity curve visualization

### Optimization
The system continuously searches for optimal parameters:
1. Technical indicator periods
2. Risk-reward ratios
3. Position sizing rules
4. Entry/exit conditions

## Future Improvements

1. Multi-timeframe analysis
2. Additional technical indicators
3. Advanced position sizing strategies
4. Market regime detection
5. Real-time execution capabilities
6. Portfolio management extension
