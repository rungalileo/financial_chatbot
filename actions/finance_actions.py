from utils.stock_action_types import StockActionWithState, StockActionResult, StockActionCompoundResult
from utils.agent_state import StockAgentState
from utils.finance_utils import simplify_dollar_amount, plot_stock_chart
from utils.finance_utils import get_stock_symbol_from_user_phrase
from actions.error_dialogs import PLEASE_SPECIFY_A_STOCK
from utils.llm_utils import extract_time_period_from_query
from utils.stock_action_types import StockAction, StockActionResult, StockActionCompoundResult
from actions.general_llm_actions import GeneralInvestmentQuery
from prompts import GET_FIELD_FROM_STOCK_INFO_PROMPT
from utils.news_utils import get_news_newsapi_org
from utils.llm_utils import ask_openai, extract_time_period_from_query, extract_top_n_from_query, is_not_none
from utils.finance_utils import plot_stock_chart
from utils.llm_utils import is_not_none


import spacy
import faiss
import pickle
import numpy as np
from tqdm import tqdm
import datetime
import yfinance as yf
import pandas as pd
import random
import streamlit as st


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
        history = history[['Close']]
        intermediate_response.markdown("")

        prediction = plot_stock_chart(stock_symbol, time_window=time_period)

        return StockActionCompoundResult([
            f"Here's <b>{stock_symbol}'s</b> performance over the last <b>{time_period}</b>", 
            prediction, 
            f"Here's more recent data from the past week", 
            history], 
            ["html", "chart", "html", "dataframe"])


class CompareStocks(StockActionWithState):
    def execute(self, user_phrase: str, stock_symbols: str, state: StockAgentState) -> StockActionResult:
        stock_list = stock_symbols.split(",")
        if len(stock_list) != 2:
            return StockActionResult("Please provide exactly two stock symbols for comparison.", "html")

        stock1, stock2 = stock_list
        ticker1, ticker2 = yf.Ticker(stock1), yf.Ticker(stock2)

        try:
            info1, info2 = ticker1.info, ticker2.info
        except Exception as e:
            return StockActionResult(f"Error fetching stock data: {e}", "html")

        # Select key financial metrics for comparison
        keys = {
            "currentPrice": "Current Price",
            "marketCap": "Market Cap",
            "totalRevenue": "Total Revenue",
            "netIncomeToCommon": "Net Income",
            "trailingPE": "Trailing P/E",
            "forwardPE": "Forward P/E",
            "profitMargins": "Profit Margins",
            "returnOnAssets": "Return on Assets",
            "returnOnEquity": "Return on Equity",
            "debtToEquity": "Debt to Equity",
            "dividendRate": "Dividend Rate",
            "dividendYield": "Dividend Yield"
        }

        data = []
        for key, metric_name in keys.items():
            value1 = info1.get(key, "N/A")
            value2 = info2.get(key, "N/A")

            # Format values
            if isinstance(value1, (int, float)):
                if key in ["marketCap", "totalRevenue", "netIncomeToCommon"]:
                    value1 = f"${simplify_dollar_amount(str(value1))}"
                else:
                    value1 = f"{round(value1, 1)}"

            if isinstance(value2, (int, float)):
                if key in ["marketCap", "totalRevenue", "netIncomeToCommon"]:
                    value2 = f"${simplify_dollar_amount(str(value2))}"
                else:
                    value2 = f"{round(value2, 1)}"

            # Ensure all values are strings to prevent serialization issues
            data.append([metric_name, str(value1), str(value2)])

        # Create a pandas DataFrame
        df = pd.DataFrame(data, columns=["Metric", stock1, stock2])

        state["last_stock_symbol"] = None
        state["last_query"] = None
        state["last_action"] = None

        return StockActionCompoundResult(
            [
                f"Alright, here is a comparison of {stock1} and {stock2}:",
                df
            ],
            ["html", "dataframe"]
        )

class CompanyFinanceQuestionAndAnswer(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)
        print(f"[DEBUG] STOCK SYMBOL(S): {stock_symbol}")

        stock_list = stock_symbol.split(",")

        if len(stock_list) == 2:
            stock1, stock2 = stock_list
            ticker1, ticker2 = yf.Ticker(stock1), yf.Ticker(stock2)
        else:
            stock1 = stock_list[0]
            ticker1 = yf.Ticker(stock1)

        if stock_symbol == "None":
            return GeneralInvestmentQuery().execute(user_phrase, stock_symbol)

        stock = yf.Ticker(stock_symbol)
        info = stock.info

        extract_from_stock_info_prompt = f"""
        {GET_FIELD_FROM_STOCK_INFO_PROMPT}
        User question: {user_phrase}
        Company: {stock_symbol}
        Information:
        ```json
        {info}
        ```
        """
        response = ask_openai(user_content=extract_from_stock_info_prompt)
        return StockActionResult(response, "html")



class WhichStocksToBuy(StockAction):

    def __init__(self, top_n: int = 50):
        self.nlp = spacy.load("en_core_web_lg")
        self.top_n = top_n
        self.index, self.company_vectors, self.company_names = self._load_or_build_faiss_index()

        with open("sectors.csv", "r") as f:
            sectors = f.readlines()
        self.possible_sectors = [sector.strip() for sector in sectors]


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

        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sp_500_stocks = table['Symbol'].to_list()
        # nasdaq_stocks = list(PyTickerSymbols().get_stocks_by_index("NASDAQ 100"))
        all_stocks = sp_500_stocks
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

        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sp500_stocks = table['Symbol'].to_list()
        # nasdaq_stocks = list(PyTickerSymbols().get_stocks_by_index("NASDAQ 100"))

        if sector != "None" and sector != "none":
            matched_companies = self._find_best_matching_companies(sector, top_n=50)
            sp500_stocks = [stock for stock in sp500_stocks if stock["name"] in {name for name, _ in matched_companies}]
            # nasdaq_stocks = [stock for stock in nasdaq_stocks if stock["name"] in {name for name, _ in matched_companies}]
            print(f"[DEBUG] Found {len(sp500_stocks)} stocks in sector:{sector}")

        df_stocks = pd.DataFrame(sp500_stocks)
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

class GetTopPerformers(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        st.markdown("Let's see what the top performers are...")
        N = extract_top_n_from_query(user_phrase)
        if is_not_none(N):
            st.markdown(f"Let's see the top {N} performers...")
            return WhichStocksToBuy(top_n=int(N)).execute(user_phrase, stock_symbol)
        else:
            return WhichStocksToBuy().execute(user_phrase, stock_symbol)

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
