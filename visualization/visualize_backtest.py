import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

def load_trade_data():
    """Load the latest trade report from backtest_report directory"""
    report_dir = "backtest_report"
    trade_files = sorted([f for f in os.listdir(report_dir) if f.startswith('trades_')])
    if not trade_files:
        raise ValueError("No trade report files found")
    
    latest_file = os.path.join(report_dir, trade_files[-1])
    print(f"Loading trade data from: {latest_file}")
    return pd.read_csv(latest_file)

def load_price_data():
    """Load price data"""
    data_path = "data/raw/USDJPY_5M_2023.csv"
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def create_candlestick_chart(price_df, trades_df, output_file):
    """Create candlestick chart with indicators and trade points"""
    # Calculate indicators
    price_df['ema7'] = price_df['close'].ewm(span=7, adjust=False).mean()
    price_df['ema21'] = price_df['close'].ewm(span=21, adjust=False).mean()
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

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

    # Add volume
    colors = ['red' if row['open'] > row['close'] else 'green' for index, row in price_df.iterrows()]
    fig.add_trace(go.Bar(x=price_df['time'],
                        y=price_df['tick_volume'],
                        marker_color=colors,
                        name='Volume'),
                  row=2, col=1)

    # Process trades
    completed_trades = trades_df[trades_df['pnl'].notna()].copy()
    completed_trades['entry_time'] = pd.to_datetime(completed_trades['entry_time'])
    completed_trades['exit_time'] = pd.to_datetime(completed_trades['exit_time'])

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
        height=1000,
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

def main():
    # Load data
    trades_df = load_trade_data()
    price_df = load_price_data()

    # Create output directory if it doesn't exist
    trades_filename = os.path.basename(os.path.dirname(trades_df['entry_time'].iloc[0]))
    
    # Save to the same directory as trades file
    output_file = f"backtest_report/trades_20241205_224350.html"
    create_candlestick_chart(price_df, trades_df, output_file)

if __name__ == "__main__":
    main()