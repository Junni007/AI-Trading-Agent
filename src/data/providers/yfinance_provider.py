"""
yfinance Data Provider (Fallback).
Wraps existing yfinance logic into the DataProvider interface.
"""
import logging
from typing import Optional, List, Callable
from datetime import datetime
import pandas as pd

from .base import DataProvider

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed")


class YFinanceProvider(DataProvider):
    """
    yfinance data provider.
    
    Good for:
    - Free historical data
    - Quick prototyping
    
    Limitations:
    - 15 minute delay on quotes
    - Rate limits
    - 60-day max for intraday data
    """
    
    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is not installed")
    
    def _parse_interval(self, timeframe: str) -> str:
        """Convert standard timeframe to yfinance interval."""
        if timeframe == '4h':
            raise ValueError("yfinance does not support '4h'. Please aggregate '1h' data manually.")

        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1wk',
        }
        
        if timeframe not in mapping:
            supported = list(mapping.keys())
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")
            
        return mapping[timeframe]

    def get_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV bars."""
        try:
            df = yf.download(
                tickers=ticker,
                start=start.isoformat(timespec='seconds'),
                end=end.isoformat(timespec='seconds'),
                interval=self._parse_interval(timeframe),
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                return None
            
            # Handle MultiIndex columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"yfinance get_bars failed for {ticker}: {e}")
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
        logged_unexpected_format = False
        
        try:
            df = yf.download(
                tickers=tickers,
                start=start.isoformat(timespec='seconds'),
                end=end.isoformat(timespec='seconds'),
                interval=self._parse_interval(timeframe),
                progress=False,
                auto_adjust=True,
                group_by='ticker',
                threads=True
            )
            
            if df.empty:
                return result
            
            is_multi = isinstance(df.columns, pd.MultiIndex)
            
            for ticker in tickers:
                try:
                    if is_multi and ticker in df.columns.get_level_values(0):
                        ticker_df = df[ticker].copy()
                    elif not is_multi and len(tickers) == 1:
                        ticker_df = df.copy()
                    else:
                        if not logged_unexpected_format and len(tickers) > 1:
                            logger.warning(f"Unexpected yfinance batch format for {len(tickers)} tickers. Columns: {df.columns}")
                            logged_unexpected_format = True
                        continue
                    
                    ticker_df = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    ticker_df.dropna(inplace=True)
                    
                    if not ticker_df.empty:
                        result[ticker] = ticker_df
                        
                except Exception as e:
                    logger.warning(f"Failed to process {ticker}: {e}")
                    
        except Exception as e:
            logger.error(f"yfinance batch download failed: {e}")
        
        return result
    
    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """Get the most recent bar."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            
            if hist.empty:
                return None
            
            row = hist.iloc[-1]
            return {
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume'],
                'timestamp': hist.index[-1]
            }
            
        except Exception as e:
            logger.error(f"yfinance get_latest_bar failed: {e}")
            return None
    
    def get_latest_quote(self, ticker: str) -> Optional[dict]:
        """Get latest quote (limited in yfinance)."""
        from datetime import timezone
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'bid_size': info.get('bidSize'),
                'ask_size': info.get('askSize'),
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"yfinance get_latest_quote failed: {e}")
            return None
    
    def supports_streaming(self) -> bool:
        """yfinance does not support streaming."""
        return False
