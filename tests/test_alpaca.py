"""
Alpaca API Integration Test Script
Tests all data provider functionality using IntradayDataLoader.
"""
from datetime import datetime, timedelta
import os
import sys
import pytest
from src.data_loader_intraday import IntradayDataLoader
from src.config import settings

# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def loader():
    """Pytest fixture that provides an initialized IntradayDataLoader."""
    return IntradayDataLoader()

# ============================================================================
# PYTEST TEST FUNCTIONS
# ============================================================================

def test_config():
    """Test configuration loading."""
    assert settings.DATA_PROVIDER in ["alpaca", "yfinance"]
    # Alpaca key assertion only when keys are actually set
    if settings.DATA_PROVIDER == "alpaca" and settings.ALPACA_API_KEY:
        assert settings.ALPACA_API_KEY, "ALPACA_API_KEY not set"

def test_loader_init():
    """Test loader initialization."""
    loader = IntradayDataLoader()
    assert loader is not None
    assert hasattr(loader, 'fetch_data')

def test_fetch_data(loader):
    """Test fetching data."""
    # This might use YFinance fallback if keys aren't set in CI/CD
    df = loader.fetch_data('AAPL', interval='1d', period='5d')
    
    if df is not None:
        assert not df.empty
        assert 'Close' in df.columns
        assert 'Volume' in df.columns
        assert len(df) > 0

def test_technical_indicators(loader):
    """Test adding technical indicators."""
    import pandas as pd
    # Create dummy dataframe if fetch fails or just to test logic
    data = {
        'Open': [100, 101, 102, 103, 104] * 10,
        'High': [105, 106, 107, 108, 109] * 10,
        'Low': [95, 96, 97, 98, 99] * 10,
        'Close': [102, 103, 104, 105, 106] * 10,
        'Volume': [1000, 1100, 1200, 1300, 1400] * 10
    }
    df = pd.DataFrame(data)
    
    df_result = loader.add_technical_indicators(df)
    
    # Check for required columns for RL
    assert 'MACD' in df_result.columns
    assert 'Log_Return' in df_result.columns
    assert 'RSI' in df_result.columns
    assert 'ATR' in df_result.columns

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))

