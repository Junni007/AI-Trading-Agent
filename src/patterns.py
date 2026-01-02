import pandas as pd
import numpy as np

class CandlestickDetector:
    """
    Vectorized Candlestick Pattern Detector.
    Input: DataFrame with Open, High, Low, Close.
    Output: DataFrame with boolean columns for patterns.
    """
    
    @staticmethod
    def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
        """
        Bullish Engulfing:
        1. Previous candle is Red.
        2. Current candle is Green.
        3. Current Body engulfs Previous Body.
        """
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        prev_body_size = np.abs(prev_open - prev_close)
        
        curr_open = df['Open']
        curr_close = df['Close']
        curr_body_size = np.abs(curr_open - curr_close)
        
        # Logic
        is_prev_red = prev_close < prev_open
        is_curr_green = curr_close > curr_open
        engulfing = (curr_open <= prev_close) & (curr_close >= prev_open)
        
        # Strict version: Current body > Previous body
        strict_body = curr_body_size > prev_body_size
        
        return is_prev_red & is_curr_green & engulfing & strict_body

    @staticmethod
    def detect_hammer(df: pd.DataFrame) -> pd.Series:
        """
        Hammer:
        1. Small Body (upper 30% of range).
        2. Long Lower Shadow (> 2x Body).
        3. Small/No Upper Shadow.
        """
        body_size = np.abs(df['Close'] - df['Open'])
        full_range = df['High'] - df['Low']
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        
        # Logic
        # Avoid division by zero
        full_range = full_range.replace(0, 0.0001)
        
        is_small_body = body_size < (0.3 * full_range)
        long_lower_shadow = lower_shadow > (2.0 * body_size)
        small_upper_shadow = upper_shadow < (1.0 * body_size) # Allow small shadow
        
        return is_small_body & long_lower_shadow & small_upper_shadow

    @staticmethod
    def detect_doji(df: pd.DataFrame) -> pd.Series:
        """
        Doji:
        1. Body is extremely small (< 5% of range).
        """
        body_size = np.abs(df['Close'] - df['Open'])
        full_range = df['High'] - df['Low']
        
        full_range = full_range.replace(0, 0.0001)
        return body_size <= (0.05 * full_range)

    @staticmethod
    def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Adds pattern columns to the DataFrame."""
        df = df.copy()
        df['Pattern_Engulfing'] = CandlestickDetector.detect_bullish_engulfing(df)
        df['Pattern_Hammer'] = CandlestickDetector.detect_hammer(df)
        df['Pattern_Doji'] = CandlestickDetector.detect_doji(df)
        return df
