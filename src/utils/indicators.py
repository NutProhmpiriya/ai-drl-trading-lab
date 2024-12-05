import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    
    # EMA
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_trend'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow'] * 100
    
    # RSI
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = calculate_rsi(df['close'])
    df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
    df['rsi_trend'] = df['rsi'] - df['rsi_ma']
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    df['obv_trend'] = (df['obv'] - df['obv_ma']) / df['obv_ma'] * 100
    
    # ATR
    def calculate_atr(high, low, close, period=14):
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['atr_ratio'] = df['atr'] / df['close'] * 100
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'] * 100
    
    return df.dropna()
