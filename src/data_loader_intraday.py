import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from src.config import settings

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IntradayLoader')

class IntradayDataLoader:
    """
    Robust Data Loader for Intraday (15m, 5m, 1m) data.
    Primary: Alpaca API (Faster, Reliable).
    Fallback: yfinance (Backup).
    """
    
    def __init__(self):
        self.cache = {}
        self.alpaca_client = None
        
        if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
            try:
                self.alpaca_client = StockHistoricalDataClient(
                    settings.ALPACA_API_KEY,
                    settings.ALPACA_SECRET_KEY
                )
                logger.info("✅ Alpaca Client Initialized")
            except Exception as e:
                logger.error(f"❌ Failed to init Alpaca Client: {e}")
        else:
            logger.warning("⚠️ Alpaca Credentials missing. Using yfinance fallback.")

    def fetch_data(self, ticker: str, interval: str = '15m', period: str = '59d') -> Optional[pd.DataFrame]:
        """
        Fetches intraday data for a single ticker.
        """
        # Try Alpaca First
        # Skip Alpaca for Indian stocks (.NS, .BO) and indices (^NSEI, ^NSEBANK) as Alpaca only supports US equities
        is_indian_stock = ticker.endswith('.NS') or ticker.endswith('.BO') or ticker.startswith('^NSE')
        
        if not is_indian_stock and self.alpaca_client and settings.DATA_PROVIDER == "alpaca":
            df = self._fetch_alpaca(ticker, interval)
            if df is not None and not df.empty:
                return df
            logger.warning(f"Alpaca fetch failed for {ticker}. Falling back to yfinance.")
            
        # Fallback to yfinance
        return self._fetch_yfinance(ticker, interval, period)

    def _fetch_alpaca(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        try:
            # Map interval string to TimeFrame
            tf_map = {'1m': TimeFrame.Minute, '15m': TimeFrame.Minute, '1h': TimeFrame.Hour, '1d': TimeFrame.Day}
            # Note: Alpaca '15m' request needs custom handling or just fetch Minutes and resample?
            # Alpaca SDK supports TimeFrame(15, TimeFrameUnit.Minute) but simple mapping here:
            # For simplicity in this MVP, we might need to fetch 1Min and resample if 15m not directly supported by enum?
            # Actually Alpaca allows arbitrary Multi-minute.
            
            # Let's use 15Min specific constructor if available or just fallback to 1Min and resample.
            # SDK `TimeFrame.Minute` is 1Min.
            # We will fetch 1Min bars and resample to 15Min to be robust, OR check SDK docs.
            # Standard SDK usage: TimeFrame(15, TimeFrameUnit.Minute)
            
            # Simplified for now: Fetch days of data
            start_date = datetime.now() - timedelta(days=5 if interval=='1m' else 60)
            
            # Correct TimeFrame construction for 15m
            if interval == '15m':
                tf = TimeFrame(15, TimeFrameUnit.Minute)
            elif interval == '5m':
                tf = TimeFrame(5, TimeFrameUnit.Minute)
            elif interval == '1h':
                tf = TimeFrame.Hour
            else:
                tf = TimeFrame.Minute # Default 1m

            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start_date,
                limit=10000, 
                adjustment='all',
                feed='iex'  # 'sip' (pro) or 'iex' (free-ish)
            )

            bars = self.alpaca_client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            
            if df.empty: return None
            
            # Reset index to get 'timestamp' as column, or handle MultiIndex (symbol, timestamp)
            # Alpaca returns MultiIndex [symbol, timestamp]
            df = df.reset_index(level=0, drop=True) # Drop symbol level
            
            # Rename columns to standard specific case (Open, High, Low, Close, Volume)
            # Alpaca columns are lowercase: open, high, low, close, volume
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Ensure proper timezone (remove tz for compatibility or keep?)
            # yfinance is usually tz-naive or local. Alpaca is UTC.
            # Let's allow UTC but ensure datetime index.
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            logger.error(f"Alpaca Error: {e}")
            return None

    def _fetch_yfinance(self, ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        """Legacy yfinance fetcher (Backup via yfinance)"""
        try:
            # Respect yfinance constraints
            if interval == '1m' and int(period[:-1]) > 7:
                period = '7d'
            
            # Download
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False
            )
            
            if df is None or df.empty: return None
            
            # Format
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    df.columns = df.columns.get_level_values(0)
            
            df = df.loc[:, ~df.columns.duplicated()]
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in required:
                if c not in df.columns: return None
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
            df.dropna(inplace=True)
            return df[required]
            
        except Exception as e:
            logger.error(f"YF Error: {e}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'Sniper' and 'RL' features: VWAP, RSI, ATR, MACD, Log_Return.
        """
        if df is None or df.empty: return df
        df = df.copy()
        
        # 1. VWAP
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
        
        # 2. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. ATR (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # 4. Volume Z-Score
        vol_mean = df['Volume'].rolling(window=20).mean()
        vol_std = df['Volume'].rolling(window=20).std()
        df['Vol_Z'] = (df['Volume'] - vol_mean) / vol_std
        
        # 5. MACD (12, 26, 9) - Required for RL
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 6. Log Returns - Required for RL
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        df.dropna(inplace=True)
        return df

if __name__ == "__main__":
    loader = IntradayDataLoader()
    # Test fallback if no keys, or alpaca if keys present
    print("Testing Fetch...")
    df = loader.fetch_data("AAPL", interval="15m")
    if df is not None:
        df = loader.add_technical_indicators(df)
        print(df.tail())
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("Fetch Failed.")
