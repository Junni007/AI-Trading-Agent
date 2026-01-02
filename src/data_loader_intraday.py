import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Dict

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IntradayLoader')

class IntradayDataLoader:
    """
    Robust Data Loader for Intraday (15m, 5m, 1m) data.
    Handles yfinance limitations (60-day max for <1d intervals).
    """
    
    def __init__(self):
        self.cache = {}

    def fetch_data(self, ticker: str, interval: str = '15m', period: str = '59d') -> Optional[pd.DataFrame]:
        """
        Fetches intraday data for a single ticker.
        
        Args:
            ticker: Symbol (e.g., 'AAPL', 'NVDA')
            interval: '15m', '5m', '1h' (Default: '15m')
            period: '59d' (Max for 15m is 60d, use 59 for safety)
            
        Returns:
            pd.DataFrame or None if failed/empty.
        """
        try:
            # Respect yfinance constraints
            # 1m = max 7 days
            if interval == '1m' and int(period[:-1]) > 7:
                logger.warning(f"Period {period} too long for 1m data. truncating to 7d.")
                period = '7d'
                
            logger.info(f"Fetching {interval} data for {ticker} (Period: {period})...")
            
            # Download
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False # Single thread to avoid rate limits on loop
            )
            
            if df is None or df.empty:
                logger.warning(f"No data found for {ticker}")
                return None
                
            # Formatting
            # Ensure index is Datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Flatten MultiIndex columns if present (common in new yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Standardize Columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"Missing columns {missing} for {ticker}")
                return None
            
            # Clean Data
            df = df[required_cols].copy()
            df.dropna(inplace=True)
            
            if len(df) < 50:
                logger.warning(f"Insufficient data points ({len(df)}) for {ticker}")
                return None
                
            logger.info(f"Successfully loaded {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'Sniper' features: VWAP, RSI, ATR.
        """
        if df is None or df.empty: return df
        
        df = df.copy()
        
        # 1. VWAP (Intraday Volume Weighted Average Price)
        # simplistic calculation (rolling window approx for continuity across days or reset daily)
        # For sniper momentum, checking vs a Rolling VWAP is often sufficient.
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
        
        # 2. RSI (14 period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. ATR (14 period) - For Volatility Normalization
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # 4. Volume Z-Score (vs 20 period mean)
        vol_mean = df['Volume'].rolling(window=20).mean()
        vol_std = df['Volume'].rolling(window=20).std()
        df['Vol_Z'] = (df['Volume'] - vol_mean) / vol_std
        
        # Cleanup
        df.dropna(inplace=True)
        return df

if __name__ == "__main__":
    # Sanity Check
    loader = IntradayDataLoader()
    df_test = loader.fetch_data("NVDA", interval="15m")
    if df_test is not None:
        df_test = loader.add_technical_indicators(df_test)
        print(df_test.tail())
        print("Sanity Check Passed.")
    else:
        print("Sanity Check Failed.")
