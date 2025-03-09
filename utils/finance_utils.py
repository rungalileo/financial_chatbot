import time
import warnings
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from openai import OpenAI
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pytickersymbols import PyTickerSymbols
from IPython.display import display, HTML
from utils.llm_utils import ask_openai
from prompts import GET_STOCK_SYMBOL_PROMPT, RATING_BASED_ON_NEWS_PROMPT

import matplotlib.dates as mdates
import os

from dotenv import load_dotenv

load_dotenv()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

warnings.filterwarnings("ignore")


def get_stock_symbol_from_user_phrase(user_phrase: str, stock_symbol: str) -> str:
    if not stock_symbol:
        stock_symbol_prompt = GET_STOCK_SYMBOL_PROMPT.format(user_phrase=user_phrase)
        stock_symbol = ask_openai(user_content=stock_symbol_prompt)
    print(f"[DEBUG][Inside Action] STOCK SYMBOL: {stock_symbol}")
    return stock_symbol

def simplify_dollar_amount(amount_str: str) -> str:
    amount = float(amount_str.replace('$', '').replace(',', ''))
    if amount >= 1_000_000_000:
        amount /= 1_000_000_000
        return f"{amount:.0f}B"
    elif amount >= 1_000_000:
        amount /= 1_000_000
        return f"{amount:.0f}M"
    elif amount >= 1_000:
        amount /= 1_000
        return f"{amount:.0f}K"
    else:
        return amount_str

def get_industry_from_ticker(ticker):
    stock = yf.Ticker(ticker)
    return stock.info.get('industry', None)

def classify_valuation(ticker):
    stock = yf.Ticker(ticker)
    gross_margins, ebitda_margins, operating_margins, trailing_peg_ratio = stock.info.get('grossMargins', -1), stock.info.get('ebitdaMargins', -1), stock.info.get('operatingMargins', -1), stock.info.get('trailingPegRatio', -1)
    if trailing_peg_ratio is not None and trailing_peg_ratio > 2.0:
        return "Overvalued"
    elif trailing_peg_ratio is not None and 1.0 <= trailing_peg_ratio <= 2.0:
        if gross_margins > 0.50 and ebitda_margins > 0.30 and operating_margins > 0.20:
            return "Correctly Valued"
        else:
            return "Overvalued"
    else:
        if gross_margins > 0.40 and ebitda_margins > 0.20 and operating_margins > 0.15:
            return "Undervalued"
        else:
            return "Correctly Valued"

def buy_or_sell(content, company_name="the company"):
    try:
        time.sleep(1)
        rating_based_on_news_content = RATING_BASED_ON_NEWS_PROMPT.format(content=content, company_name=company_name)
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock market expert with accurate predictions."},
                {"role": "user", "content": rating_based_on_news_content},
            ]
        )
        output = response.choices[0].message.content.strip("```json").strip("```").strip()
        return json.loads(output)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_stock_chart(ticker, time_window):
    stock = yf.Ticker(ticker)
    df = stock.history(period=time_window)
    df["Date"] = df.index
    df["Percent Diff"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
    df['Future_Price'] = df['Close'].shift(-1)
    df = df.dropna()
    df['Target'] = (df['Future_Price'] - df['Close']) / df['Close']
    df.sort_index(inplace=True)
    ts = df['Close']

    if ts.empty:
        print(f"[DEBUG] Stock {ticker} has no data for the time period {time_window}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))


    ax.plot(ts.index, ts, color="#FF4500", linewidth=2, label='Stock Price')

    df['MA50'] = df['Close'].rolling(window=50).mean()
    ax.plot(df.index, df['MA50'], color="#1f77b4", linestyle="--", linewidth=2, label="50-day MA")

    ax.axvline(x=ts.index[-1], color='gray', linestyle='--', linewidth=1.5)

    ax.set_title(f"{ticker} Stock Price Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price (USD)", fontsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    ax.legend(frameon=True, fontsize=12)

    return fig

def training_data():
    np.random.seed(42)
    n_samples = 2000
    data = []
    for _ in range(n_samples):
        grossMargins = np.random.uniform(0.5, 0.7)
        ebitdaMargins = np.random.uniform(0.4, 0.6)
        operatingMargins = np.random.uniform(0.3, 0.5)
        trailingPegRatio = np.random.uniform(0.5, 3.0)
        priceToEarnings = np.random.uniform(10, 40)
        returnOnAssets = np.random.uniform(0.05, 0.2)
        returnOnEquity = np.random.uniform(0.1, 0.4)
        debtToEquity = np.random.uniform(20, 50)
        currentRatio = np.random.uniform(1.0, 2.0)
        priceToSales = np.random.uniform(5, 15)
        
        if trailingPegRatio < 1 and returnOnEquity > 0.3 and priceToEarnings < 20:
            label = "STRONG BUY (Trailing PEG Ratio=LOW and Return on Equity=HIGH and Price to Earnings=LOW)"
        elif trailingPegRatio < 2 and returnOnEquity > 0.25 and priceToEarnings < 25:
            label = "BUY (Trailing PEG Ratio=LOW and Return on Equity=GOOD and Price to Earnings=LOW)"
        elif trailingPegRatio > 2.5 or debtToEquity > 40:
            label = "SELL (Trailing PEG Ratio=HIGH or Debt to Equity=HIGH)"
        else:
            label = "NEUTRAL"
        
        data.append([grossMargins, ebitdaMargins, operatingMargins, trailingPegRatio,
                     priceToEarnings, returnOnAssets, returnOnEquity, debtToEquity, 
                     currentRatio, priceToSales, label])
    
    return pd.DataFrame(data, columns=[
        'grossMargins', 'ebitdaMargins', 'operatingMargins', 'trailingPegRatio',
        'priceToEarnings', 'returnOnAssets', 'returnOnEquity', 'debtToEquity', 
        'currentRatio', 'priceToSales', 'label'])

def worth_buying(stock):
    df = training_data()
    stock = yf.Ticker(stock)
    financial_data = { 
        key: stock.info[key] 
        for key in ['grossMargins', 'ebitdaMargins', 'operatingMargins', 'trailingPegRatio', 
                'trailingPE', 'returnOnAssets', 'returnOnEquity', 'debtToEquity', 
                'currentRatio', 'priceToSalesTrailing12Months'] 
        if key in stock.info 
    }

    financial_data['priceToEarnings'] = financial_data.pop('trailingPE', None)
    financial_data['priceToSales'] = financial_data.pop('priceToSalesTrailing12Months', None)

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    try:
        financial_input = np.array([list(financial_data.values())]).reshape(1, -1)
        prediction = clf.predict(financial_input)
    except Exception as e:
        return "Unable to predict"
    return prediction[0]
