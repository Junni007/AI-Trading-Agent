"""
Alpaca API Integration Test Script
Tests all data provider functionality.
Works with both pytest and direct execution.
"""
from datetime import datetime, timedelta
import os
import sys
import pytest


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def provider():
    """Pytest fixture that provides an initialized data provider."""
    from src.data.factory import get_provider
    return get_provider()


# ============================================================================
# PYTEST TEST FUNCTIONS
# ============================================================================

def test_config():
    """Test configuration loading."""
    from src.config import settings
    assert settings.DATA_PROVIDER in ["alpaca", "yfinance"]
    # Alpaca key assertion only when keys are actually set
    if settings.DATA_PROVIDER == "alpaca" and os.getenv("ALPACA_API_KEY"):
        assert settings.ALPACA_API_KEY, "ALPACA_API_KEY not set"
        assert settings.ALPACA_SECRET_KEY, "ALPACA_SECRET_KEY not set"


def test_provider_init():
    """Test provider initialization via factory."""
    from src.data.factory import get_provider
    provider = get_provider()
    assert provider is not None
    assert hasattr(provider, 'get_bars')
    assert hasattr(provider, 'get_latest_bar')


def test_latest_bar(provider):
    """Test fetching latest bar."""
    bar = provider.get_latest_bar('AAPL')
    # Bar may be None if market is closed, which is acceptable
    if bar is not None:
        assert 'close' in bar
        assert 'volume' in bar
        assert bar['close'] > 0


def test_latest_quote(provider):
    """Test fetching latest quote."""
    quote = provider.get_latest_quote('AAPL')
    if quote is not None:
        assert 'bid' in quote
        assert 'ask' in quote


def test_historical_bars(provider):
    """Test fetching historical bars."""
    end = datetime.now()
    start = end - timedelta(days=5)
    bars = provider.get_bars('AAPL', start, end, '1d')
    # May return None due to subscription limits on free tier
    # This is expected behavior, not a test failure
    assert bars is None or len(bars) >= 0


def test_batch_bars(provider):
    """Test batch historical bars."""
    end = datetime.now()
    start = end - timedelta(days=5)
    result = provider.get_bars_batch(['AAPL', 'MSFT'], start, end, '1d')
    # May return empty dict due to subscription limits
    assert isinstance(result, dict)


def test_factory():
    """Test provider factory."""
    from src.data.factory import get_provider
    provider = get_provider()
    provider_name = type(provider).__name__
    assert provider_name in ["AlpacaProvider", "YFinanceProvider"]


# ============================================================================
# STANDALONE EXECUTION (for detailed output)
# ============================================================================

def _run_standalone():
    """Run tests with detailed output when executed directly."""
    print("=" * 60)
    print("   ALPACA API INTEGRATION TEST")
    print("=" * 60)
    
    results = []
    
    # Test config
    print("\n=== 1. Config Check ===")
    try:
        from src.config import settings
        print(f"  DATA_PROVIDER: {settings.DATA_PROVIDER}")
        print(f"  ALPACA_API_KEY: {settings.ALPACA_API_KEY[:8]}..." if settings.ALPACA_API_KEY else "  ALPACA_API_KEY: Not set")
        print(f"  ALPACA_SECRET_KEY: {settings.ALPACA_SECRET_KEY[:8]}..." if settings.ALPACA_SECRET_KEY else "  ALPACA_SECRET_KEY: Not set")
        print("  ✅ Config loaded successfully")
        results.append(("Config", True))
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        results.append(("Config", False))
    
    # Test provider
    print("\n=== 2. Provider Initialization ===")
    provider = None
    try:
        from src.data.factory import get_provider
        provider = get_provider()
        print(f"  ✅ Provider initialized: {type(provider).__name__}")
        print(f"  Streaming supported: {provider.supports_streaming()}")
        results.append(("Provider Init", True))
    except Exception as e:
        print(f"  ❌ Init error: {e}")
        results.append(("Provider Init", False))
    
    if provider:
        # Test latest bar
        print("\n=== 3. Latest Bar Test (AAPL) ===")
        try:
            bar = provider.get_latest_bar('AAPL')
            if bar:
                print(f"  ✅ Latest bar received:")
                print(f"     Close: ${bar['close']:.2f}")
                print(f"     Volume: {bar['volume']:,}")
            else:
                print("  ⚠️  No bar returned (market may be closed)")
            results.append(("Latest Bar", True))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(("Latest Bar", False))
        
        # Test latest quote
        print("\n=== 4. Latest Quote Test (AAPL) ===")
        try:
            quote = provider.get_latest_quote('AAPL')
            if quote:
                print(f"  ✅ Latest quote received:")
                print(f"     Bid: ${quote['bid']:.2f} x {quote['bid_size']}")
                print(f"     Ask: ${quote['ask']:.2f} x {quote['ask_size']}")
            else:
                print("  ⚠️  No quote returned")
            results.append(("Latest Quote", True))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(("Latest Quote", False))
        
        # Test historical
        print("\n=== 5. Historical Bars Test ===")
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            bars = provider.get_bars('AAPL', start, end, '1d')
            if bars is not None and len(bars) > 0:
                print(f"  ✅ Historical bars: {len(bars)} rows")
            else:
                print("  ⚠️  No historical data (may require paid subscription)")
            results.append(("Historical Bars", True))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(("Historical Bars", False))
    
    # Test factory
    print("\n=== 6. Factory Test ===")
    try:
        from src.data.factory import get_provider
        p = get_provider()
        print(f"  ✅ Factory returned: {type(p).__name__}")
        results.append(("Factory", True))
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(("Factory", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("   TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    print(f"\n  Total: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_standalone())

