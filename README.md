# AI DRL Trading Lab

## Description
An automated trading system using Deep Reinforcement Learning (DRL) for forex trading, specifically designed for USDJPY on the 5-minute timeframe. The system implements strict risk management rules and uses technical indicators for enhanced decision making.

## Features
- Deep Reinforcement Learning with PPO algorithm
- Technical indicators (EMA, RSI, OBV, ATR)
- Risk management with ATR-based position sizing
- Daily drawdown control
- Automated backtesting and analysis

## Project Structure
```
ai-drl-trading-lab/
├── docs/
│   └── concept.md           # Detailed system concept documentation
├── src/
│   ├── data/
│   │   └── raw/            # Price data files
│   ├── rl_env/
│   │   └── forex_env.py    # Trading environment
│   ├── rl_model/
│   │   └── ...            # Trained models
│   ├── scripts/
│   │   ├── train_model.py  # Model training script
│   │   └── backtest.py     # Backtesting script
│   └── utils/
│       └── indicators.py   # Technical indicators
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ai-drl-trading-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training the model:
```bash
python src/scripts/train_model.py
```

2. Running backtest:
```bash
python src/scripts/backtest.py
```

3. View results:
- Trading results will be saved in `src/backtest_report/trades.csv`
- Performance charts will be saved in `src/backtest_report/backtest_results.png`

## Configuration
Key parameters can be modified in the respective scripts:

- `src/rl_env/forex_env.py`:
  - Initial balance
  - Leverage
  - Maximum daily drawdown
  - Risk per trade

- `src/scripts/train_model.py`:
  - Training timesteps
  - Learning rate
  - Batch size
  - Other PPO parameters

## Risk Management
- Maximum 1% risk per trade
- 5% maximum daily drawdown
- ATR-based position sizing
- 1:1.5 minimum risk-reward ratio
- No overnight positions

## Results Analysis
The backtest results include:
- Total return
- Win rate
- Maximum drawdown
- Trade statistics
- Equity curve
- Trade distribution

## Contributing
Feel free to submit issues and enhancement requests.

## License
[Your chosen license]
