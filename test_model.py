import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import gymnasium as gym

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
    console = Console()
    returns = []
    win_rate = 0
    trades = []
    
    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        refresh_per_second=10
    ) as progress:
        # Main task for backtesting progress
        task = progress.add_task(
            "[cyan]Backtesting Progress", 
            total=n_episodes,
            start=True
        )
        
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
            
            # Update progress
            progress.update(task, advance=1)
    
    win_rate = win_rate / n_episodes * 100
    
    # Create evaluation results table
    table = Table(title="[bold]Backtesting Results", show_header=False, 
                 title_style="yellow", border_style="bright_black")
    table.add_column("Metric", style="bright_white")
    table.add_column("Value", style="bright_cyan")
    
    table.add_row("Number of Episodes", str(n_episodes))
    table.add_row("Average Return", f"{np.mean(returns):.2f}")
    table.add_row("Win Rate", f"{win_rate:.2f}%")
    table.add_row("Standard Deviation", f"{np.std(returns):.2f}")
    
    console.print("\n")
    console.print(table)
    
    # Create trades DataFrame and analyze results
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Create trade statistics table
        trade_table = Table(title="[bold]Trade Statistics", show_header=False,
                          title_style="yellow", border_style="bright_black")
        trade_table.add_column("Metric", style="bright_white")
        trade_table.add_column("Value", style="bright_cyan")
        
        trade_table.add_row("Total Trades", str(len(trades_df)))
        trade_table.add_row("Win Rate", f"{(trades_df['pnl'] > 0).mean():.2%}")
        trade_table.add_row("Average Profit", f"${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}")
        trade_table.add_row("Average Loss", f"${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}")
        trade_table.add_row("Profit Factor", 
            f"{abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()):.2f}")
        trade_table.add_row("Max Drawdown", f"${trades_df['pnl'].cumsum().min():.2f}")
        
        console.print("\n")
        console.print(trade_table)
        
        # Save report
        report_path = f"backtest_report/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(report_path)
        console.print(f"\n[green]Detailed trade report saved to:[/green] {report_path}")

def main():
    # Create directories if they don't exist
    os.makedirs("backtest_report", exist_ok=True)

    # Load and prepare data
    data_path = "data/raw/USDJPY_5M_2024.csv"
    df = prepare_data(data_path)

    # Create test environment
    test_env = ForexTradingEnv(
        df=df,
        initial_balance=100.0,
        leverage=1000.0,
        max_daily_drawdown=0.05
    )
    
    # Load the model
    model_dir = "rl_models"
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.zip')])
    if not model_files:
        raise ValueError("No model files found in rl_models directory")
    
    latest_model = os.path.join(model_dir, model_files[-1])
    print(f"\nLoading model from: {latest_model}")
    model = PPO.load(latest_model)
    
    print("\nStarting backtesting...")
    evaluate_model(model, test_env)

if __name__ == "__main__":
    main()
