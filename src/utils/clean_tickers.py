import pandas as pd

# List of delisted/invalid tickers from logs
delisted_tickers = [
    'JUBILANT', 'ITDCEM', 'TCNSBRANDS', 'SYNDIBANK', 'CADILAHC', 'SREINFRA', 'HDFC', 'CORPBANK', 'MCDOWELL-N', 'MONSANTO',
    'ISEC', 'ANDHRABANK', 'LTI', 'MINDTREE', 'PEL', 'LAXMIMACH', 'ADANITRANS', 'PHILIPCARB', 'RNAM', 'HEXAWARE',
    'WABCOINDIA', 'TV18BRDCST', 'ESSELPACK', 'GMRINFRA', 'JSLHISAR', 'PVR', 'L&TFH', 'NBVENTURES', 'GUJFLUORO',
    'MOTHERSUMI', 'GEPIL', 'ALBK', 'GDL', 'AMARAJABAT', 'KALPATPOWR', 'GET&D', 'TATACOFFEE', 'MINDAIND', 'INFRATEL',
    'SRTRANSFIN', 'TATAMTRDVR', 'SWANENERGY',
    # Round 2
    'ORIENTBANK', 'AEGISCHEM', 'GSKCONS', 'LAKSHVILAS', 'WELSPUNIND', 'NIITTECH', 'SUNCLAYLTD', 'UJJIVAN', 'TATAMOTORS', 
    'MAHINDCIE', 'MAGMA', 'DHFL', 'IBULHSGFIN', 'IDFC', 'EQUITAS'
]

# Read CSV
# Relative path from src/utils/ to src/nifty500.csv
csv_path = '../nifty500.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Original Count: {len(df)}")
    
    # Filter
    # Ensure Symbol column matches (remove .NS if present in list but CSV might be raw symbols)
    # The logs showed X.NS errors, so the CSV likely has X (before .NS is added by ticker_utils)
    # or ticker_utils added it. The CSV downloaded usually has just Symbol.
    
    # Check if 'Symbol' column exists
    if 'Symbol' in df.columns:
        # Filter out rows where Symbol is in delisted_tickers
        df_clean = df[~df['Symbol'].isin(delisted_tickers)]
        print(f"Cleaned Count: {len(df_clean)}")
        
        df_clean.to_csv(csv_path, index=False)
        print("Successfully saved cleaned CSV.")
    else:
        print("Error: 'Symbol' column not found in CSV.")

except Exception as e:
    print(f"Error processing CSV: {e}")
