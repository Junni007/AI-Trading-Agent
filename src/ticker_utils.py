import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_sp500_tickers():
    """Scrapes S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Antibot Fix: Use User-Agent header
        import requests
        from io import StringIO
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Fix for Yahoo Finance (BF.B -> BF-B)
        tickers = [t.replace('.', '-') for t in tickers]
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500: {e}")
        return ["AAPL", "MSFT", "GOOGL"] # Fallback

def get_nifty50_tickers():
    """Returns top 50 Indian Stocks (NSE) for valid Yahoo Finance symbols."""
    # Nifty 50 Symbols as of 2024 (Suffix .NS for NSE)
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS",
        "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS", "ONGC.NS", "TATAMOTORS.NS",
        "NTPC.NS", "AXISBANK.NS", "ULTRACEMCO.NS", "ADANIPORTS.NS", "POWERGRID.NS",
        "M&M.NS", "WIPRO.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "ASIANPAINT.NS",
        "JSWSTEEL.NS", "NESTLEIND.NS", "TATASTEEL.NS", "GRASIM.NS", "TECHM.NS",
        "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS", "INDUSINDBANK.NS", "CIPLA.NS",
        "TATACONSUM.NS", "DRREDDY.NS", "EICHERMOT.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "HEROMOTOCO.NS", "BPCL.NS", "SHRIRAMFIN.NS", "HINDALCO.NS", "UPL.NS"
    ]
    logger.info(f"Loaded {len(tickers)} Nifty 50 tickers.")
    return tickers

def get_extended_tickers(limit=500):
    """
    Simulates a '5000' ticker request by combining lists and careful scaling.
    For Colab Free, 5000 is risky. We limit to 'limit' by default.
    """
    sp500 = get_sp500_tickers()
    nifty = get_nifty50_tickers()
    
    combined = sp500 + nifty
    # If user wants massive scale, we would need a bigger sources/file.
    # For now, this ~550 is a safe "Large" start.
    return combined[:limit] if limit else combined
