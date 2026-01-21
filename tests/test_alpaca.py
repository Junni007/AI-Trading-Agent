"""
Alpaca API Integration Test Script
Tests all data provider functionality.
"""
from datetime import datetime, timedelta
import sys

def test_config():
    """Test configuration loading."""
    print("=== 1. Config Check ===")
    try:
        from src.config import settings
        print(f"  DATA_PROVIDER: {settings.DATA_PROVIDER}")
        print(f"  ALPACA_API_KEY: {settings.ALPACA_API_KEY[:8]}..." if settings.ALPACA_API_KEY else "  ALPACA_API_KEY: Not set")
        print(f"  ALPACA_SECRET_KEY: {settings.ALPACA_SECRET_KEY[:8]}..." if settings.ALPACA_SECRET_KEY else "  ALPACA_SECRET_KEY: Not set")
        print(f"  ALPACA_PAPER: {settings.ALPACA_PAPER}")
        print("  ✅ Config loaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def test_provider_init():
    """Test provider initialization via factory."""
    print("\n=== 2. Provider Initialization ===")
    try:
        from src.data.factory import get_provider
        provider = get_provider()
        print(f"  ✅ Provider initialized: {type(provider).__name__}")
        print(f"  Streaming supported: {provider.supports_streaming()}")
        return provider
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Init error: {e}")
        return None


def test_latest_bar(provider):
    """Test fetching latest bar."""
    print("\n=== 3. Latest Bar Test (AAPL) ===")
    try:
        bar = provider.get_latest_bar('AAPL')
        if bar:
            print(f"  ✅ Latest bar received:")
            print(f"     Close: ${bar['close']:.2f}")
            print(f"     Volume: {bar['volume']:,}")
            print(f"     Timestamp: {bar['timestamp']}")
            return True
        else:
            print("  ⚠️  No bar returned (market may be closed)")
            return True  # Not a failure
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_latest_quote(provider):
    """Test fetching latest quote."""
    print("\n=== 4. Latest Quote Test (AAPL) ===")
    try:
        quote = provider.get_latest_quote('AAPL')
        if quote:
            print(f"  ✅ Latest quote received:")
            print(f"     Bid: ${quote['bid']:.2f} x {quote['bid_size']}")
            print(f"     Ask: ${quote['ask']:.2f} x {quote['ask_size']}")
            return True
        else:
            print("  ⚠️  No quote returned")
            return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_historical_bars(provider):
    """Test fetching historical bars."""
    print("\n=== 5. Historical Bars Test (AAPL, 5 days) ===")
    try:
        end = datetime.now()
        start = end - timedelta(days=5)
        bars = provider.get_bars('AAPL', start, end, '1d')
        if bars is not None and len(bars) > 0:
            print(f"  ✅ Historical bars received: {len(bars)} rows")
            print(bars.tail(3).to_string(index=True))
            return True
        else:
            print("  ⚠️  No historical data returned")
            return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_batch_bars(provider):
    """Test batch historical bars."""
    print("\n=== 6. Batch Bars Test (AAPL, MSFT, GOOGL) ===")
    try:
        end = datetime.now()
        start = end - timedelta(days=5)
        result = provider.get_bars_batch(['AAPL', 'MSFT', 'GOOGL'], start, end, '1d')
        if result:
            print(f"  ✅ Batch bars received for: {list(result.keys())}")
            for ticker, df in result.items():
                print(f"     {ticker}: {len(df)} rows")
            return True
        else:
            print("  ⚠️  No batch data returned")
            return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_factory():
    """Test provider factory."""
    print("\n=== 7. Factory Test ===")
    try:
        from src.data.factory import get_provider
        provider = get_provider()
        print(f"  ✅ Factory returned: {type(provider).__name__}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("   ALPACA API INTEGRATION TEST")
    print("=" * 60)
    
    results = []
    
    # Test config
    results.append(("Config", test_config()))
    
    # Test provider
    provider = test_provider_init()
    results.append(("Provider Init", provider is not None))
    
    if provider:
        results.append(("Latest Bar", test_latest_bar(provider)))
        results.append(("Latest Quote", test_latest_quote(provider)))
        results.append(("Historical Bars", test_historical_bars(provider)))
        results.append(("Batch Bars", test_batch_bars(provider)))
    
    # Test factory
    results.append(("Factory", test_factory()))
    
    # Summary
    print("\n" + "=" * 60)
    print("   TEST SUMMARY")
    print("=" * 60)
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
