
import requests
import pandas as pd
from io import StringIO

def test_wiki_scrape():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print(f"Testing fetch from {url}...")
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Parsing table...")
            tables = pd.read_html(StringIO(response.text))
            print(f"Found {len(tables)} tables.")
            print(tables[0].head())
        else:
            print("Failed to fetch.")
    except Exception as e:
        print(f"Error: {e}")

    # Also test Nifty 500 URL
    nifty_url = "https://en.wikipedia.org/wiki/NIFTY_500"
    print(f"\nTesting fetch from {nifty_url}...")
    try:
        response = requests.get(nifty_url, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
             # Nifty 500 page often lists components in a table, or might be different. 
             # Let's check if we can read it.
             tables = pd.read_html(StringIO(response.text))
             if tables:
                 print(f"Found {len(tables)} tables on Nifty page.")
                 print(tables[0].head())
             else:
                 print("No tables found on Nifty page.")
    except Exception as e:
        print(f"Error on Nifty: {e}")

if __name__ == "__main__":
    test_wiki_scrape()
