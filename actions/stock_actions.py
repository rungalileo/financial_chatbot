from utils.actions_utils import StockAction, StockActionResult, StockActionCompoundResult
from utils.finance_utils import simplify_dollar_amount, worth_buying, plot_stock_chart
from utils.llm_utils import ask_openai, extract_time_period_from_query, extract_top_n_from_query, is_not_none
from utils.news_utils import get_news_newsapi_org
from pytickersymbols import PyTickerSymbols
from actions.prompts import EXTRACT_FROM_STOCK_INFO_PROMPT
from actions.error_dialogs import CAN_ONLY_DO_PERF_INFO_ON_SPECIFIC_STOCKS, PLEASE_SPECIFY_A_STOCK
from utils.finance_utils import get_stock_symbol_from_user_phrase

from tqdm import tqdm
import yfinance as yf
import streamlit as st
import pandas as pd
import random
import datetime
import spacy
import faiss
import pickle
import numpy as np


class WhatCanYouDo(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        from tools import STOCK_ACTIONS
        action_names = list(STOCK_ACTIONS.keys())
        action_names = [name for name in action_names if name not in ["fallback_response", "what_can_you_do"]]
        action_names = [name.replace("_", " ").capitalize() for name in action_names]
        action_names_string = ", ".join(action_names)
        action_names_string = "<ul>" + "\n".join([f"<li>{name}</li>" for name in action_names]) + "</ul>"
        return StockActionResult(f"I can help you with the following: {action_names_string}", "html")

class FallbackResponse(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        return StockActionResult(
            "I'm not programmed to answer this. I'm a financial guru. Ask me about stock prices, stock history, top performers, etc.", "html")

class GeneralInvestmentQuery(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        intermediate_response = st.markdown(f"Answering your query...")
        prompt = f"Answer in brief, the following question: {user_phrase}, Stock symbol: {stock_symbol}"
        response = ask_openai(user_content=prompt)
        intermediate_response.markdown("")
        return StockActionResult(response, "html")

class CompanyFinanceQuestionAndAnswer(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return GeneralInvestmentQuery().execute(user_phrase, stock_symbol)

        stock = yf.Ticker(stock_symbol)
        info = stock.info

        extract_from_stock_info_prompt = f"""
        {EXTRACT_FROM_STOCK_INFO_PROMPT}
        User question: {user_phrase}
        Company: {stock_symbol}
        Information:
        ```json
        {info}
        ```
        """
        response = ask_openai(user_content=extract_from_stock_info_prompt)
        return StockActionResult(response, "html")

class GetTopPerformers(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        st.markdown("Let's see what the top performers are...")
        N = extract_top_n_from_query(user_phrase)
        if is_not_none(N):
            st.markdown(f"Let's see the top {N} performers...")
            return WhichStocksToBuy(top_n=int(N)).execute(user_phrase, stock_symbol)
        else:
            return WhichStocksToBuy().execute(user_phrase, stock_symbol)

class GetNewsAndSentiment(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return StockActionResult(PLEASE_SPECIFY_A_STOCK, "html")

        st.markdown(f"Let's see what the market is saying about {stock_symbol}...")

        print(f"STOCK SYMBOL: {stock_symbol}")

        recommendation, authors = get_news_newsapi_org(stock_symbol)

        authors_list = "".join(f"<li>{author}</li>" for author in authors if author)
        headlines_list = "".join(f"<li>{headline}</li>" for headline in recommendation['headlines'])

        html_table = f"""
        <table>
            <tr><th style='text-align: left;'>Authors</th><td>{authors_list}</td></tr>
            <tr><th style='text-align: left;'>Headlines</th><td>{headlines_list}</td></tr>
            <tr><th style='text-align: left;'>Reasoning</th><td>{recommendation['reason']}</td></tr>
            <tr><th style='text-align: left;'>Sentiment</th><td>{recommendation['rating']}</td></tr>
        </table>
        """
        prediction = plot_stock_chart(stock_symbol, '1y')

        return StockActionCompoundResult([
            html_table, 
            f"Here is how {stock_symbol} has done in the last year", 
            prediction], 
            ["html", "html", "chart"])

class IsStockWorthBuying(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return StockActionResult(CAN_ONLY_DO_PERF_INFO_ON_SPECIFIC_STOCKS, "html")

        intermediate_response = st.markdown(f"Let's see if {stock_symbol} is worth buying...")
        recommendation, authors = get_news_newsapi_org(stock_symbol)

        res = worth_buying(stock_symbol)
        

        stock_chart = plot_stock_chart(stock_symbol, '2y')

        main_response = f"""
            <p>Pure stats says that {stock_symbol} is a {res}. 
            And according to recent news, {stock_symbol} is a {recommendation['rating']}.</p>
        """

        stock_news_data, authors = get_news_newsapi_org(stock_symbol)
        stock_news_data_html = f"""
            <p>Summary of the latest news about {stock_symbol}:</p>
            <ul>
                {stock_news_data['reason']}
            </ul>
        """
        intermediate_response.markdown("")
        return StockActionCompoundResult([
            main_response,
            stock_chart,
            stock_news_data_html,
            "In the end, it's up to you. So do your own research and make your own decision!"
        ],
        ["html", "chart", "html", "html"])

class TellMeAboutThisCompany(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return StockActionResult(CAN_ONLY_DO_PERF_INFO_ON_SPECIFIC_STOCKS, "html")

        st.markdown(f"Let's see what we can find about {stock_symbol}...")

        try:
            stock = yf.Ticker(stock_symbol)
            df = stock.history(period='2y').dropna()
        except Exception as e:
            return StockActionResult(f"Error in stock agent: API call failed.", "html")

        if df.empty:
            return StockActionResult("No stock data available.", "html")

        df["Percent Diff"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
        stock_value_today = df['Close'].iloc[-1]

        stock_info = stock.info
        market_cap = simplify_dollar_amount(f"${stock_info.get('marketCap', 0):,.2f}")
        arr = simplify_dollar_amount(f"${stock_info.get('totalRevenue', 0):,.2f}")
        total_cash = simplify_dollar_amount(f"${stock_info.get('totalCash', 0):,.2f}")
        industry = stock_info.get("industry", "Unknown Industry")
        city = stock_info.get("city", "Unknown City")

        week_change = stock_info.get("52WeekChange", 0) * 100

        html_content = f"""
            <p>{stock_symbol} is a {city}-based company, specializing in {industry}.</p>
            <p>It has a market cap of <span style="color: green;">{market_cap}</span>. 
            Total ARR is <span style="color: green;">{arr}</span> and it holds 
            <span style="color: blue;">{total_cash}</span> in cash.</p>
            <p>Its 52-week change is <span style="color: red;">{week_change:.2f}%</span>, 
            and its stock value today is <span style="color: red;">${stock_value_today:,.2f}</span>.</p>
            <p>Hope that helps!</p>
        """

        return StockActionResult(html_content, "html")

class GetStockPrice(StockAction):

    def __init__(self):
        self.stock_price = None

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return StockActionResult(PLEASE_SPECIFY_A_STOCK, "html")

        st.markdown(f"Fetching current price of {stock_symbol}...")

        if not stock_symbol:
            return StockActionResult("Error: Unable to determine stock symbol.", "html")

        try:
            stock = yf.Ticker(stock_symbol)
            info = stock.info

            current_price = info.get('currentPrice', 'N/A')
            target_low = info.get('targetLowPrice', 'N/A')
            target_high = info.get('targetHighPrice', 'N/A')
            recommendation = info.get('recommendationKey', '')

            sentiment_map = {
                "strong_buy": "GREAT",
                "buy": "fairly well",
                "sell": "not so great",
                "strong_sell": "poorly"
            }
            sentiment = sentiment_map.get(recommendation, "neutral")
            dialog = (
                f"<p>{stock_symbol} is doing <b>{random.choice([sentiment])}</b>!</p>"
                f"<p>The current price is <b>${current_price}</b>, "
                f"the target LOW is <span style='color: red;'>${target_low}</span>, "
                f"and the target HIGH is <span style='color: green;'>${target_high}</span>.</p>"
            )

            corporate_actions = info.get('corporateActions', [])
            if corporate_actions:
                dialog += "<p>The company recently announced the following:</p><ul>"
                for action in corporate_actions:
                    header, message = action.get('header'), action.get('message')
                    if header and message:
                        dialog += f"<li>{header}: {message}</li>"
                dialog += "</ul>"

        except Exception as e:
            return StockActionResult(f"<p>Error fetching stock data: {e}</p>", "html")

        plot_chart = plot_stock_chart(stock_symbol, '2y')

        return StockActionCompoundResult([dialog, plot_chart], ["html", "chart"])


## Get stock performance in the last X days|weeks|months|years
class GetStockPerformance(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        return self.execute(user_phrase, stock_symbol, '2w')

    def execute(self, user_phrase: str, stock_symbol: str, time_period: str = '2w') -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        time_period = extract_time_period_from_query(user_phrase)
        print(f"[DEBUG] TIME PERIOD: {time_period}")

        if stock_symbol == "None":
            return StockActionResult(PLEASE_SPECIFY_A_STOCK, "html")

        intermediate_response = st.markdown(f"Fetching history of {stock_symbol}...")
        stock = yf.Ticker(stock_symbol)
        history = stock.history(period='5d')
        history = history.iloc[::-1]
        history = history[['Open', 'High', 'Low', 'Close']]
        intermediate_response.markdown("")

        prediction = plot_stock_chart(stock_symbol, time_window=time_period)

        return StockActionCompoundResult([
            f"Here's <b>{stock_symbol}'s</b> performance over the last <b>{time_period}</b>", 
            prediction, 
            f"Here's more recent data from the past week", 
            history], 
            ["html", "chart", "html", "dataframe"])

class WhichStocksToBuy(StockAction):

    def __init__(self, top_n: int = 50):
        self.nlp = spacy.load("en_core_web_lg")
        self.top_n = top_n
        self.index, self.company_vectors, self.company_names = self._load_or_build_faiss_index()
        x = PyTickerSymbols().get_stocks_by_index("S&P 500")
        stocks_list = list(x)
        industries = []
        for i in range(len(stocks_list)):
            if "industries" in stocks_list[i]:
                industries.extend(stocks_list[i]['industries'])
        self.possible_sectors = list(set(industries))


    def _load_or_build_faiss_index(self):
        try:
            index = faiss.read_index("faiss_company_index.bin")
            with open("company_vectors.pkl", "rb") as f:
                company_vectors = pickle.load(f)
            with open("company_names.pkl", "rb") as f:
                company_names = pickle.load(f)
            print("FAISS index load successful")
            return index, company_vectors, company_names
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return self._build_faiss_index()

    def _build_faiss_index(self):
        sp500_stocks = list(PyTickerSymbols().get_stocks_by_index("S&P 500"))
        nasdaq_stocks = list(PyTickerSymbols().get_stocks_by_index("NASDAQ 100"))
        all_stocks = sp500_stocks + nasdaq_stocks
        print(f"[DEBUG] Found {len(all_stocks)} companies to index")

        company_vectors = []
        company_names = []

        for stock in all_stocks:
            industries = stock.get("industries", [])
            if not industries:
                continue
            industry_vectors = np.array([self.nlp(industry).vector for industry in industries])
            avg_vector = np.mean(industry_vectors, axis=0)
            avg_vector = avg_vector.astype("float32")

            company_vectors.append(avg_vector)
            company_names.append(stock["name"])

        company_vectors = np.array(company_vectors).astype("float32")
        faiss.normalize_L2(company_vectors)
        index = faiss.IndexFlatIP(company_vectors.shape[1])
        index.add(company_vectors)
        faiss.write_index(index, "faiss_company_index.bin")
        with open("company_vectors.pkl", "wb") as f:
            pickle.dump(company_vectors, f)
        with open("company_names.pkl", "wb") as f:
            pickle.dump(company_names, f)

        print("FAISS index built and saved successfully")
        return index, company_vectors, company_names

    def _find_best_matching_companies(self, user_sector: str, top_n=10):
        """
        Finds the companies most similar to the given user input sector.
        """
        print(f"[DEBUG] Searching for companies related to: {user_sector}")
        sector_vector = self.nlp(user_sector).vector.astype("float32").reshape(1, -1)

        faiss.normalize_L2(sector_vector)

        distances, indices = self.index.search(sector_vector, top_n)
        results = [(self.company_names[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
        for company, score in results:
            print(f"[DEBUG][MATCH] {company}: {score:.4f}")

        return results

    def _extract_sector_from_query(self, user_phrase: str) -> str:

        sector_extraction_prompt = f"""
            The user asked: "{user_phrase}".
            Identify the industry sector mentioned in this query.
            It can be amongst the following sectors:
            {self.possible_sectors}
            ONLY return the exact sector name (closest match), not any surrounding text.
            If you cannot find a relevant sector, return the exact string "None".
        """
        sector = ask_openai(user_content=sector_extraction_prompt).strip().lower()
        return sector

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        sector = self._extract_sector_from_query(user_phrase)
        top_n = extract_top_n_from_query(user_phrase)
        print(f"[DEBUG] SECTOR: {sector}")
        print(f"[DEBUG] TOP N: {top_n}")
        sp500_stocks = list(PyTickerSymbols().get_stocks_by_index("S&P 500"))
        nasdaq_stocks = list(PyTickerSymbols().get_stocks_by_index("NASDAQ 100"))

        if sector != "None" and sector != "none":
            matched_companies = self._find_best_matching_companies(sector, top_n=50)
            sp500_stocks = [stock for stock in sp500_stocks if stock["name"] in {name for name, _ in matched_companies}]
            nasdaq_stocks = [stock for stock in nasdaq_stocks if stock["name"] in {name for name, _ in matched_companies}]
            print(f"[DEBUG] Found {len(sp500_stocks + nasdaq_stocks)} stocks in sector:{sector}")

        combined_stocks = sp500_stocks + nasdaq_stocks
        df_stocks = pd.DataFrame(combined_stocks)
        df_stocks = df_stocks.drop_duplicates(subset=['symbol'])

        stock_performance = []
        progress_bar = st.progress(0)
        analyze_image_placeholder = st.empty()

        status_msg = st.markdown("Crunching some numbers...")
        analyze_image_placeholder.image("analyze.gif", caption="Analyzing markets... ")

        for i, (name, symbol) in enumerate(tqdm(zip(df_stocks['name'], df_stocks['symbol']), total=len(df_stocks))):
            progress_bar.text(f"Found {len(df_stocks)} {sector} stocks for you. Analyzing {i + 1} out of {len(df_stocks)} stocks...")

            try:
                stock = yf.Ticker(symbol)
                history = stock.history(period='1y')
                if history.empty:
                    continue

                start_price = history['Close'].iloc[0]
                end_price = history['Close'].iloc[-1]
                percent_change = ((end_price - start_price) / start_price) * 100
                if percent_change < 0:
                    continue
                if (name, symbol) in stock_performance:
                    continue
                stock_performance.append((name, symbol, percent_change))
            except Exception as e:
                st.error(f"Error processing {name} ({symbol}): {e}")
                continue
        
        top_stocks = sorted(stock_performance, key=lambda x: x[2], reverse=True)
        if top_n != -1:
            top_stocks = top_stocks[:top_n]

        status_msg.markdown(f""" 
            <p>Did some filtering, here's {len(top_stocks)} {sector} (and related) stocks that have performed well over the past couple of years. 
            NOTE: Past indicators don't always predict future performance. 
            Consider this as a suggestion, not financial advice. I'm just a helpful assistant!</p>
        """, unsafe_allow_html=True)
        df_results = pd.DataFrame(top_stocks, columns=['Company', 'Ticker', '1-Year Gain (%)'])

        analyze_image_placeholder.empty()
        return StockActionResult(df_results, "dataframe")


class GetMarketOrSectorTrends(StockAction):

    def __init__(self):
        self.INDICES = {
            "S&P 500": "^GSPC",
            "NASDAQ 100": "^NDX",
            "Dow Jones": "^DJI"
        }

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:

        st.markdown("📈 🤓 Analyzing the market's 2 year trends...")
        
        predictions = []
        
        for name, symbol in self.INDICES.items():
            try:
                stock = yf.Ticker(symbol)
                history = stock.history(period='2y')

                if history.empty:
                    continue

                history.index = history.index.tz_localize(None)
                
                last_year = history.loc[history.index >= (datetime.datetime.today() - datetime.timedelta(days=365))]
                start_price = last_year['Close'].iloc[0]
                end_price = last_year['Close'].iloc[-1]
                percent_change = ((end_price - start_price) / start_price) * 100

                predictions.append((name, symbol, percent_change))
            except Exception as e:
                st.error(f"Error processing {name} ({symbol}): {e}")
                continue
        
        st.markdown("Here's the market sentiment from recent news...")
        sentiment_data, _ = get_news_newsapi_org("^GSPC")

        sentiment = sentiment_data['rating'] if sentiment_data else "neutral"
        reasoning = sentiment_data['reason'] if sentiment_data else "No strong bias detected."
        
        market_analysis = """
            <p>Based on historical performance and recent news sentiment, here's an outlook:</p>
            <ul>
        """
        
        for name, symbol, change in predictions:
            market_analysis += f"<li>{name} ({symbol}): Last year's growth was <b>{change:.2f}%</b>.</li>"
        
        market_analysis += "</ul>"
        market_analysis += f"<p>Recent market sentiment suggests: <b>{sentiment}</b>.</p>"
        market_analysis += f"<p>Key takeaways from financial news: {reasoning}</p>"
        
        return StockActionResult(market_analysis, "html")
