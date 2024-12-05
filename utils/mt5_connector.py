import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

class MT5Connector:
    def __init__(self):
        self.connected = False
        
    def connect(self, username=None, password=None, server=None, path=None):
        """Connect to MT5 terminal"""
        if not mt5.initialize(path=path):
            print(f"initialize() failed, error code = {mt5.last_error()}")
            return False
            
        if username and password and server:
            authorized = mt5.login(username, password, server)
            if not authorized:
                print(f"Failed to connect to account, error code: {mt5.last_error()}")
                return False
                
        self.connected = True
        return True
        
    def disconnect(self):
        """Disconnect from MT5 terminal"""
        mt5.shutdown()
        self.connected = False
        
    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            return None
            
        account_info = mt5.account_info()
        if account_info is None:
            return None
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_level': account_info.margin_level,
            'leverage': account_info.leverage
        }
        
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        if not self.connected:
            return None
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
            
        return {
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'spread': symbol_info.spread,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max
        }
        
    def get_historical_data(self, symbol, timeframe, start_date, end_date=None):
        """Get historical price data"""
        if not self.connected:
            return None
            
        timezone = pytz.timezone("Etc/UTC")
        start_date = pd.to_datetime(start_date).tz_localize(timezone)
        if end_date:
            end_date = pd.to_datetime(end_date).tz_localize(timezone)
        else:
            end_date = datetime.now(timezone)
            
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    def place_market_order(self, symbol, order_type, volume, stop_loss=None, take_profit=None):
        """Place a market order"""
        if not self.connected:
            return None
            
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit
            
        result = mt5.order_send(request)
        return result
        
    def get_positions(self):
        """Get open positions"""
        if not self.connected:
            return None
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [{
            'ticket': position.ticket,
            'symbol': position.symbol,
            'volume': position.volume,
            'type': position.type,
            'price_open': position.price_open,
            'sl': position.sl,
            'tp': position.tp,
            'profit': position.profit
        } for position in positions]
        
    def close_position(self, ticket):
        """Close a specific position"""
        if not self.connected:
            return None
            
        position = mt5.positions_get(ticket=ticket)
        if position is None:
            return None
            
        position = position[0]
        symbol = position.symbol
        lot = position.volume
        
        price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        position_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": position_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result
