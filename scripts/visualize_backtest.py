import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

def load_trade_data():
    """Load the latest trade report from backtest_report directory"""
    report_dir = "backtest_report"
    trade_files = sorted([f for f in os.listdir(report_dir) if f.startswith('trades_') and f.endswith('.csv')])
    if not trade_files:
        raise ValueError("No trade report files found")
    
    latest_file = os.path.join(report_dir, trade_files[-1])
    print(f"Loading trade data from: {latest_file}")
    
    # Read CSV with explicit parsing of datetime columns
    trades_df = pd.read_csv(latest_file, parse_dates=['entry_time', 'exit_time'])
    return trades_df

def load_price_data():
    """Load price data"""
    data_path = "data/raw/USDJPY_5M_2024.csv"
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def create_candlestick_chart(price_df, trades_df, output_file):
    """Create candlestick chart with indicators and trade points"""
    # Convert time column to datetime
    price_df['time'] = pd.to_datetime(price_df['time'])
    
    # Calculate indicators
    price_df['ema7'] = price_df['close'].ewm(span=7, adjust=False).mean()
    price_df['ema21'] = price_df['close'].ewm(span=21, adjust=False).mean()
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.75, 0.25])

    # Add candlestick
    fig.add_trace(go.Candlestick(x=price_df['time'],
                                open=price_df['open'],
                                high=price_df['high'],
                                low=price_df['low'],
                                close=price_df['close'],
                                name='USDJPY'),
                  row=1, col=1)

    # Add EMAs
    fig.add_trace(go.Scatter(x=price_df['time'],
                            y=price_df['ema7'],
                            line=dict(color='blue', width=1),
                            name='EMA 7'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=price_df['time'],
                            y=price_df['ema21'],
                            line=dict(color='orange', width=1),
                            name='EMA 21'),
                  row=1, col=1)

 

    # Process trades
    completed_trades = trades_df.dropna(subset=['pnl']).copy()

    # Add trade entry points
    entry_types = {
        'long': {'color': 'green', 'symbol': 'triangle-up'},
        'short': {'color': 'red', 'symbol': 'triangle-down'}
    }
    
    for position_type in ['long', 'short']:
        position_trades = completed_trades[completed_trades['position_type'] == position_type]
        
        # Entry points
        fig.add_trace(go.Scatter(
            x=position_trades['entry_time'],
            y=position_trades['entry_price'],
            mode='markers',
            marker=dict(
                symbol=entry_types[position_type]['symbol'],
                size=10,
                color=entry_types[position_type]['color']
            ),
            name=f"{position_type.capitalize()} Entry",
            legendgroup=position_type
        ), row=1, col=1)

        # Exit points
        fig.add_trace(go.Scatter(
            x=position_trades['exit_time'],
            y=position_trades['exit_price'],
            mode='markers',
            marker=dict(
                symbol='x',
                size=10,
                color=entry_types[position_type]['color']
            ),
            name=f"{position_type.capitalize()} Exit",
            legendgroup=position_type
        ), row=1, col=1)

        # Connect entry and exit with lines
        for _, trade in position_trades.iterrows():
            fig.add_trace(go.Scatter(
                x=[trade['entry_time'], trade['exit_time']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines',
                line=dict(
                    color=entry_types[position_type]['color'],
                    width=1,
                    dash='dot'
                ),
                showlegend=False,
                legendgroup=position_type
            ), row=1, col=1)

    # Update layout
    fig.update_layout(
        title='USDJPY Price Action with Trades',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=1200,
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d %H:%M',
            showgrid=True,
            gridcolor='LightGray',
            tickangle=45,
            dtick='H1',  # Show tick every hour
            tickmode='auto',
            nticks=20
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            groupclick="toggleitem"
        ),
        margin=dict(r=150)
    )

    # Save the figure
    fig.write_html(output_file)
    print(f"Saved candlestick chart to: {output_file}")

def analyze_trading_performance(trades_df):
    """Analyze trading performance metrics"""
    # Filter completed trades only
    completed_trades = trades_df.dropna(subset=['pnl'])
    
    # Basic statistics
    total_trades = len(completed_trades)
    winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
    losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Profit analysis
    total_profit = completed_trades['pnl'].sum()
    avg_profit = completed_trades['pnl'].mean()
    max_profit = completed_trades['pnl'].max()
    max_loss = completed_trades['pnl'].min()
    
    # Risk metrics
    profit_std = completed_trades['pnl'].std()
    sharpe_ratio = avg_profit / profit_std if profit_std != 0 else 0
    
    # Drawdown analysis
    cumulative_returns = completed_trades['pnl'].cumsum()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns - rolling_max
    max_drawdown = drawdowns.min()
    
    # Time analysis
    completed_trades['exit_time'] = pd.to_datetime(completed_trades['exit_time'])
    completed_trades['entry_time'] = pd.to_datetime(completed_trades['entry_time'])
    avg_trade_duration = (completed_trades['exit_time'] - completed_trades['entry_time']).mean()
    
    # Position analysis
    long_trades = len(completed_trades[completed_trades['position_type'] == 'long'])
    short_trades = len(completed_trades[completed_trades['position_type'] == 'short'])
    
    print("\n=== Trading Performance Analysis ===")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${avg_profit:.2f}")
    print(f"Maximum Profit: ${max_profit:.2f}")
    print(f"Maximum Loss: ${max_loss:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: ${abs(max_drawdown):.2f}")
    print(f"Average Trade Duration: {avg_trade_duration}")
    print(f"\nPosition Distribution:")
    print(f"Long Trades: {long_trades} ({long_trades/total_trades*100:.1f}%)")
    print(f"Short Trades: {short_trades} ({short_trades/total_trades*100:.1f}%)")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def analyze_monthly_performance(trades_df):
    """Analyze trading performance by month"""
    # Filter completed trades only and convert times
    completed_trades = trades_df.dropna(subset=['pnl']).copy()
    completed_trades['exit_time'] = pd.to_datetime(completed_trades['exit_time'])
    
    # Add month column
    completed_trades['month'] = completed_trades['exit_time'].dt.strftime('%Y-%m')
    
    # Initialize monthly stats
    monthly_stats = []
    
    for month in sorted(completed_trades['month'].unique()):
        month_trades = completed_trades[completed_trades['month'] == month]
        
        # Buy orders analysis
        buy_trades = month_trades[month_trades['position_type'] == 'long']
        buy_total = len(buy_trades)
        buy_wins = len(buy_trades[buy_trades['pnl'] > 0])
        buy_win_rate = (buy_wins / buy_total * 100) if buy_total > 0 else 0
        
        # Sell orders analysis
        sell_trades = month_trades[month_trades['position_type'] == 'short']
        sell_total = len(sell_trades)
        sell_wins = len(sell_trades[sell_trades['pnl'] > 0])
        sell_win_rate = (sell_wins / sell_total * 100) if sell_total > 0 else 0
        
        # Total analysis
        total_trades = len(month_trades)
        total_wins = len(month_trades[month_trades['pnl'] > 0])
        total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        total_profit = month_trades['pnl'].sum()
        
        monthly_stats.append({
            'Month': month,
            'Total Trades': total_trades,
            'Total Win Rate': f"{total_win_rate:.1f}%",
            'Total Profit': f"${total_profit:.2f}",
            'Buy Orders': buy_total,
            'Buy Wins': buy_wins,
            'Buy Win Rate': f"{buy_win_rate:.1f}%",
            'Sell Orders': sell_total,
            'Sell Wins': sell_wins,
            'Sell Win Rate': f"{sell_win_rate:.1f}%"
        })
    
    # Convert to DataFrame for better display
    stats_df = pd.DataFrame(monthly_stats)
    
    # Print table
    print("\n=== Monthly Trading Performance ===")
    print(stats_df.to_string(index=False))
    
    return stats_df

def main():
    # Get the latest trades CSV file from backtest_report directory
    report_dir = "backtest_report"
    trades_files = sorted([f for f in os.listdir(report_dir) if f.startswith('trades_') and f.endswith('.csv')])
    if not trades_files:
        raise ValueError("No trades CSV files found in backtest_report directory")
    
    latest_trades = os.path.join(report_dir, trades_files[-1])
    print(f"Loading trades from: {latest_trades}")
    
    # Load the data
    trades_df = pd.read_csv(latest_trades)
    data_path = "data/raw/USDJPY_5M_2024.csv"
    price_df = pd.read_csv(data_path)
    price_df['time'] = pd.to_datetime(price_df['time'])
    
    # Analyze trading performance
    performance_metrics = analyze_trading_performance(trades_df)
    monthly_stats = analyze_monthly_performance(trades_df)
    
    # Create candlestick chart
    output_file = latest_trades.replace('.csv', '.html')
    create_candlestick_chart(price_df, trades_df, output_file)

if __name__ == "__main__":
    main()
