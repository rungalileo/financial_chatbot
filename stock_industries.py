import pandas as pd
import yfinance as yf
import time
import random

CSV_FILE = "stock_industries.csv"

def get_sp500_companies():
    """Fetches S&P 500 company symbols and names from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return [{"symbol": row["Symbol"], "name": row["Security"]} for _, row in table.iterrows()]

def get_nasdaq_100_companies():
    """Fetches NASDAQ 100 company symbols and names from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    table = tables[4] if len(tables) > 4 else tables[0]  # Adjust table index if needed
    return [{"symbol": row["Ticker"], "name": row["Company"]} for _, row in table.iterrows()]

def fetch_industry_data(stocks):
    """Fetches industry information from yfinance and returns a list of (symbol, industry)."""
    industry_data = []
    
    for stock in stocks:
        symbol = stock["symbol"]
        try:
            sleep_time = random.uniform(1.5, 3.5)  # Random delay to avoid rate limits
            print(f"[INFO] Fetching industry info for {symbol}... (Sleeping {sleep_time:.2f}s)")
            time.sleep(sleep_time)

            stock_info = yf.Ticker(symbol).info
            industry = stock_info.get("industry", "Unknown")
            industry_data.append((symbol, industry))
        except Exception as e:
            print(f"[ERROR] Failed to fetch industry for {symbol}: {e}")
            industry_data.append((symbol, "Unknown"))  # Default to "Unknown" if fetching fails

    return industry_data

def save_to_csv(data, filename):
    """Saves symbol and industry data to a CSV file."""
    df = pd.DataFrame(data, columns=["symbol", "industry"])
    df.to_csv(filename, index=False)
    print(f"[INFO] Data saved to {filename}")

def main():
    print("[INFO] Fetching stock symbols from Wikipedia...")
    sp500_stocks = get_sp500_companies()
    nasdaq_stocks = get_nasdaq_100_companies()

    all_stocks = sp500_stocks + nasdaq_stocks
    print(f"[INFO] Total stocks found: {len(all_stocks)}")

    print("[INFO] Fetching industry data from Yahoo Finance...")
    industry_data = fetch_industry_data(all_stocks)

    print("[INFO] Saving data to CSV...")
    save_to_csv(industry_data, CSV_FILE)

if __name__ == "__main__":
    main()