#!/usr/bin/env python3
"""
Shared MT5 connector and data loading utilities.
Eliminates duplicate code across the project.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger
import pytz

class MT5Connector:
    """Shared MT5 connector for data loading and trading operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MT5 connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.connected = False
        self.timezone = pytz.timezone('UTC')
        
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        # Login if credentials provided
        if 'mt5' in self.config and 'login' in self.config['mt5']:
            login = self.config['mt5']['login']
            password = self.config['mt5']['password']
            server = self.config['mt5']['server']
            
            if not mt5.login(login=login, password=password, server=server):
                logger.error("MT5 login failed")
                return False
        
        self.connected = True
        logger.info("✅ MT5 connected successfully")
        return True
    
    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("🔌 MT5 disconnected")
    
    def load_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = 'M5') -> pd.DataFrame:
        """
        Load market data from MT5.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            if not self.connect():
                return pd.DataFrame()
        
        # Convert timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Convert to UTC
        start_dt = self.timezone.localize(start_dt)
        end_dt = self.timezone.localize(end_dt)
        
        # Load data
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_dt, end_dt)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        logger.info(f"📊 Loaded {len(df)} bars for {symbol} ({timeframe})")
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if error
        """
        if not self.connected:
            if not self.connect():
                return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return (tick.bid + tick.ask) / 2
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol info dictionary or None if error
        """
        if not self.connected:
            if not self.connect():
                return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            'name': info.name,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_mode': info.trade_mode,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None) -> Optional[int]:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            volume: Order volume
            price: Order price (None for market orders)
            sl: Stop loss
            tp: Take profit
            
        Returns:
            Order ticket or None if error
        """
        if not self.connected:
            if not self.connect():
                return None
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"✅ Order placed: {order_type} {volume} {symbol} at {result.price}")
        return result.order
    
    def get_positions(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get open positions.
        
        Args:
            symbol: Filter by symbol (None for all)
            
        Returns:
            DataFrame with positions
        """
        if not self.connected:
            if not self.connect():
                return pd.DataFrame()
        
        positions = mt5.positions_get(symbol=symbol)
        
        if positions is None:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information.
        
        Returns:
            Account info dictionary or None if error
        """
        if not self.connected:
            if not self.connect():
                return None
        
        account = mt5.account_info()
        if account is None:
            return None
        
        return {
            'login': account.login,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'profit': account.profit,
            'currency': account.currency
        }

# Global MT5 connector instance
_mt5_connector = None

def get_mt5_connector(config: Dict[str, Any]) -> MT5Connector:
    """
    Get or create global MT5 connector instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MT5Connector instance
    """
    global _mt5_connector
    
    if _mt5_connector is None:
        _mt5_connector = MT5Connector(config)
    
    return _mt5_connector

def load_mt5_data(symbol: str, start_date: str, end_date: str, timeframe: str = 'M5', config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load MT5 data using shared connector.
    
    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Timeframe
        config: Configuration dictionary
        
    Returns:
        DataFrame with OHLCV data
    """
    if config is None:
        from utils.config.config_loader import load_config
        config = load_config()
    
    connector = get_mt5_connector(config)
    return connector.load_data(symbol, start_date, end_date, timeframe) 