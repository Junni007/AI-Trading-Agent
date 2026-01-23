
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.ticker_utils import get_nifty500_tickers, get_extended_tickers

def verify_nifty_load():
    print("Testing get_nifty500_tickers()...")
    tickers = get_nifty500_tickers()
    
    print(f"Count: {len(tickers)}")
    
    if len(tickers) == 0:
        print("FAIL: No tickers loaded.")
        return

    # Check Suffix
    if not tickers[0].endswith(".NS"):
        print(f"FAIL: Ticker {tickers[0]} missing .NS suffix.")
    else:
        print(f"Success: Tickers have .NS suffix (e.g., {tickers[0]}).")

    if len(tickers) < 490:
        print(f"WARNING: Expected ~500 tickers, got {len(tickers)}. Check CSV.")
    else:
        print("PASS: Ticker count looks correct for Nifty 500.")

    # Test Extended
    print("\nTesting get_extended_tickers(limit=None)...")
    extended = get_extended_tickers(limit=None)
    print(f"Extended Count: {len(extended)}")
    
if __name__ == "__main__":
    verify_nifty_load()
