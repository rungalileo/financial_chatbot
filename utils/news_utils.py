import requests
import yfinance as yf
import time
from datetime import datetime, timedelta
from utils.finance_utils import buy_or_sell
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def get_market_news() -> tuple[list[str], list[str]]:
    url = "https://newsapi.org/v2/everything"
    api_key = os.getenv("NEWSAPI_API_KEY")
    if api_key is None:
        api_key = st.secrets["NEWSAPI_API_KEY"]

    current_date = datetime.today()

    params = {
        'apikey': api_key,
        'q': '"stock market" OR "S&P 500" OR "NASDAQ" OR "Dow Jones" OR "market update"',
        'language': 'en',
        'sortBy': "publishedAt",
    }
    response = requests.get(url, params=params)
    authors = []
    titles = []
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        for i, article in enumerate(articles):
            if i >= 10:
                break
            authors.append(article.get('author', ''))
            titles.append(article.get('title', ''))
    else:
        print(f"Error: {response.status_code}")
        return None

    return list(set(authors)), list(set(titles))

def get_news_for_stock(ticker: str) -> tuple[str, list[str]]:
    url = "https://newsapi.org/v2/everything"

    # first load from os.getenv, if not found, then load from streamlit secrets
    api_key = os.getenv("NEWSAPI_API_KEY")

    if api_key is None:
        api_key = st.secrets["NEWSAPI_API_KEY"]

    stock = yf.Ticker(ticker)
    all_news_article_content = ""
    current_date = datetime.today()

    authors = []
    for _ in range(4):
        params = {
            'apikey': api_key,
            'q': ticker + " (" + str(stock.info["shortName"]) + ") stock",
            'from': current_date.strftime("%Y-%m-%d"),
            'sortBy': "publishedAt",
        }
        response = requests.get(url, params=params)
        time.sleep(1)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            for i, article in enumerate(articles):
                if i >= 10:
                    break
                all_news_article_content += article.get('content', '')
                authors.append(article.get('author', ''))
        else:
            print(f"Error: {response.status_code}")
            return None
        current_date -= timedelta(days=6)
    recommendation_with_reason = buy_or_sell(all_news_article_content, str(stock.info["shortName"]))
    return recommendation_with_reason, list(set(authors))
