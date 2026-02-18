"""
Abstract Data Provider Interface.
Allows swapping between yfinance, Alpaca, or other data sources.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Callable
import pandas as pd
from datetime import datetime


class DataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    def get_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start: Start datetime
            end: End datetime
            timeframe: '1m', '5m', '15m', '1h', '1d'
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        pass
    
    @abstractmethod
    def get_bars_batch(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1d'
    ) -> dict:
        """
        Fetch historical bars for multiple tickers.
        
        Returns:
            Dict mapping ticker -> DataFrame
        """
        pass
    
    @abstractmethod
    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent bar for a ticker.
        
        Returns:
            Dict with keys: open, high, low, close, volume, timestamp
        """
        pass
    
    @abstractmethod
    def get_latest_quote(self, ticker: str) -> Optional[dict]:
        """
        Get the latest bid/ask quote.
        
        Returns:
            Dict with keys: bid, ask, bid_size, ask_size, timestamp
        """
        pass
    
    def supports_streaming(self) -> bool:
        """Whether this provider supports real-time streaming."""
        return False
    
    def stream_bars(
        self,
        tickers: List[str],
        callback: Callable[[str, dict], None]
    ) -> None:
        """
        Stream real-time bars (optional, not all providers support this).
        
        Args:
            tickers: List of symbols to stream
            callback: Function called with (ticker, bar_data) on each update
        """
        raise NotImplementedError("Streaming not supported by this provider")
