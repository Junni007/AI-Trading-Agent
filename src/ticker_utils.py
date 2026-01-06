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

def get_nifty_total_market():
    """
    Returns a broad universe of Liquid Indian Stocks (Nifty 100 + Key Midcaps).
    Expanded for 'Whole Market' feel while maintaining scan speed.
    """
    nifty_50 = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
        "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "LICI.NS",
        "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
        "ULTRACEMCO.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "M&M.NS", "ADANIENT.NS", "TATAMOTORS.NS", "COALINDIA.NS",
        "JSWSTEEL.NS", "TATASTEEL.NS", "NESTLEIND.NS", "WIPRO.NS",
        "HCLTECH.NS", "TECHM.NS", "INDUSINDBK.NS", "HDFCLIFE.NS",
        "SBILIFE.NS", "BAJAJFINSV.NS", "GRASIM.NS", "BRITANNIA.NS",
        "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS", "APOLLOHOSP.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "HINDALCO.NS", "UPL.NS",
        "BPCL.NS", "TATACONSUM.NS", "ADANIPORTS.NS", "SHRIRAMFIN.NS"
    ]

    nifty_next_50 = [
        "HAL.NS", "DLF.NS", "BEL.NS", "VBL.NS", "ZOMATO.NS",
        "SIEMENS.NS", "TRENT.NS", "PIDILITIND.NS", "IOC.NS",
        "RECLTD.NS", "PFC.NS", "GAIL.NS", "CHOLAFIN.NS",
        "BANKBARODA.NS", "CANBK.NS", "PNB.NS", "TVSMOTOR.NS",
        "HAVELLS.NS", "ABB.NS", "AMBUJACEM.NS", "INDIGO.NS",
        "NAUKRI.NS", "JIOFIN.NS", "GODREJCP.NS", "VEDL.NS",
        "DABUR.NS", "SHREECEM.NS", "BERGEPAINT.NS",
        "ICICIPRULI.NS", "BOSCHLTD.NS", "MOTHERSON.NS",
        "POLYCAB.NS", "SBICARD.NS", "MARICO.NS", "SRF.NS",
        "MUTHOOTFIN.NS", "ICICIGI.NS", "COLPAL.NS", "IRCTC.NS",
        "TORNTPOWER.NS", "LODHA.NS", "JSWENERGY.NS",
        "MANKIND.NS", "ADANIENSOL.NS", "ADANIGREEN.NS",
        "ATGL.NS", "DMART.NS", "LTIM.NS"
    ]
    
    fo_universe = list(set(nifty_50 + nifty_next_50))
    logger.info(f"Loaded {len(fo_universe)} Indian Market tickers.")
    return fo_universe

def get_nifty500_tickers():
    """
    Returns full Nifty 500 list from local CSV.
    FALLBACK: Returns core 100 list if CSV missing.
    """
    csv_path = "src/nifty500.csv"
    try:
        if not pd.io.common.file_exists(csv_path):
             # Try absolute path based on execution context if needed, or relative
             # But let's assume running from root.
             import os
             csv_path = os.path.join(os.path.dirname(__file__), "nifty500.csv")
             
        df = pd.read_csv(csv_path)
        tickers = df['Symbol'].tolist()
        # Add .NS and dedup
        tickers = [f"{t}.NS" for t in tickers]
        tickers = list(set(tickers))
        logger.info(f"Loaded {len(tickers)} Nifty 500 tickers from CSV.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to load Nifty 500 CSV: {e}. Falling back to Nifty Total Market.")
        return get_nifty_total_market()

def get_nifty50_tickers():
    """Returns top 50 Indian Stocks (NSE) for valid Yahoo Finance symbols."""
    return get_nifty_total_market() # Redirect to larger list

def get_extended_tickers(limit=500):
    """
    Simulates a '5000' ticker request by combining lists and careful scaling.
    For Colab Free, 5000 is risky. We limit to 'limit' by default.
    """
    # sp500 = get_sp500_tickers()
    # nifty = get_nifty50_tickers()
    
    # combined = sp500 + nifty
    
    # NEW LOGIC: Just get Nifty 500
    combined = get_nifty500_tickers()
    
    if limit and len(combined) > limit:
        return combined[:limit]
    return combined
