"""
Alpaca Markets Data Provider.
Uses alpaca-py for market data with streaming support.
"""
import os
import logging
from typing import Optional, List, Callable
from datetime import datetime, timedelta
import pandas as pd

from .base import DataProvider

logger = logging.getLogger(__name__)

# Try to import alpaca-py
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.live import StockDataStream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Run: pip install alpaca-py")


class AlpacaProvider(DataProvider):
    """
    Alpaca Markets data provider.
    
    Benefits:
    - Real-time data (with subscription)
    - WebSocket streaming
    - 5+ years of intraday history
    - Better rate limits than yfinance
    
    Requires:
    - ALPACA_API_KEY environment variable
    - ALPACA_SECRET_KEY environment variable
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py")
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not found. Using free data (15min delayed).")
            # Free tier works without keys for delayed data
            self.client = StockHistoricalDataClient()
        else:
            self.client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
        
        self._stream = None
    
    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert string timeframe to Alpaca TimeFrame object."""
        mapping = {
            '1m': TimeFrame(1, TimeFrameUnit.Minute),
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '30m': TimeFrame(30, TimeFrameUnit.Minute),
            '1h': TimeFrame(1, TimeFrameUnit.Hour),
            '4h': TimeFrame(4, TimeFrameUnit.Hour),
            '1d': TimeFrame(1, TimeFrameUnit.Day),
            '1w': TimeFrame(1, TimeFrameUnit.Week),
        }
        return mapping.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))
    
    def get_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV bars for a single ticker."""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=self._parse_timeframe(timeframe),
                start=start,
                end=end
            )
            
            bars = self.client.get_stock_bars(request)
            
            if not bars or ticker not in bars:
                return None
            
            # Convert to DataFrame
            df = bars[ticker].df.reset_index()
            # Explicit column renaming for robustness
            rename_map = {
                'timestamp': 'Timestamp',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            }
            
            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Alpaca data missing columns. Found: {df.columns}")
                return None
                
            df = df.rename(columns=rename_map)
            df = df.set_index('Timestamp')
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Alpaca get_bars failed for {ticker}: {e}")
            return None
    
    def get_bars_batch(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1d'
    ) -> dict:
        """Fetch historical bars for multiple tickers."""
        result = {}
        
        try:
            # Alpaca supports batch requests
            request = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=self._parse_timeframe(timeframe),
                start=start,
                end=end
            )
            
            bars = self.client.get_stock_bars(request)
            
            for ticker in tickers:
                if ticker in bars:
                    df = bars[ticker].df.reset_index()
                    rename_map = {
                        'timestamp': 'Timestamp',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }
                    df = df.rename(columns=rename_map)
                    df = df.set_index('Timestamp')
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    result[ticker] = df
                    
        except Exception as e:
            logger.error(f"Alpaca batch request failed: {e}")
            # Fall back to individual requests
            for ticker in tickers:
                df = self.get_bars(ticker, start, end, timeframe)
                if df is not None:
                    result[ticker] = df
        
        return result
    
    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """Get the most recent bar."""
        try:
            request = StockLatestBarRequest(symbol_or_symbols=ticker)
            bar = self.client.get_stock_latest_bar(request)
            
            if ticker not in bar:
                return None
            
            b = bar[ticker]
            return {
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume,
                'timestamp': b.timestamp
            }
            
        except Exception as e:
            logger.error(f"Alpaca get_latest_bar failed: {e}")
            return None
    
    def get_latest_quote(self, ticker: str) -> Optional[dict]:
        """Get the latest bid/ask quote."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = self.client.get_stock_latest_quote(request)
            
            if ticker not in quote:
                return None
            
            q = quote[ticker]
            return {
                'bid': q.bid_price,
                'ask': q.ask_price,
                'bid_size': q.bid_size,
                'ask_size': q.ask_size,
                'timestamp': q.timestamp
            }
            
        except Exception as e:
            logger.error(f"Alpaca get_latest_quote failed: {e}")
            return None
    
    def supports_streaming(self) -> bool:
        """Alpaca supports WebSocket streaming."""
        return bool(self.api_key and self.secret_key)
    
    def stream_bars(
        self,
        tickers: List[str],
        callback: Callable[[str, dict], None]
    ) -> None:
        """Stream real-time bars via WebSocket."""
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key required for streaming")
        
        self._stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        async def bar_handler(bar):
            callback(bar.symbol, {
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'timestamp': bar.timestamp
            })
        
        self._stream.subscribe_bars(bar_handler, *tickers)
        self._stream.run()
    
    def stop_streaming(self):
        """Stop the WebSocket stream."""
        if self._stream:
            self._stream.stop()
            self._stream = None
