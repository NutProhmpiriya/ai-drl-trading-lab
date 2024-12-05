import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_data(file_path):
    # Read and prepare data
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Calculate indicators
    # EMA 5 and 13
    df['ema5'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['ema13'] = EMAIndicator(close=df['close'], window=13).ema_indicator()
    
    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # ATR
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # OBV with tick_volume
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['tick_volume']).on_balance_volume()
    
    return df

def plot_data(df, start_date=None, end_date=None):
    if start_date and end_date:
        df = df[start_date:end_date]
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=('USDJPY 5M with Indicators', 'Tick Volume', 'RSI', 'ATR')
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='USDJPY'
        ),
        row=1, col=1
    )

    # Add EMA traces
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ema5'],
            name='EMA 5',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ema13'],
            name='EMA 13',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )

    # Add Tick Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['tick_volume'],
            name='Tick Volume',
            marker_color='rgb(158,202,225)',
            opacity=0.6
        ),
        row=2, col=1
    )

    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=3, col=1
    )

    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)

    # Add ATR
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['atr'],
            name='ATR',
            line=dict(color='green', width=1)
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        title_text="USDJPY 5M Market Analysis",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Save as HTML file for interactive viewing
    fig.write_html('src/visualization/market_analysis.html')
    
    # Also save as static image
    fig.write_image('src/visualization/market_analysis.png')

if __name__ == "__main__":
    # Read data
    file_path = "src/data/raw/USDJPY_5M_2023.csv"
    df = prepare_data(file_path)
    
    # Plot entire year
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    plot_data(df, start_date, end_date)
