from utils.stock_action_types import StockActionWithState, StockActionResult, StockActionCompoundResult
from utils.agent_state import StockAgentState
from utils.finance_utils import simplify_dollar_amount, plot_stock_chart
from utils.finance_utils import get_stock_symbol_from_user_phrase
from actions.error_dialogs import PLEASE_SPECIFY_A_STOCK
from utils.llm_utils import extract_time_period_from_query
from utils.stock_action_types import StockAction, StockActionResult, StockActionCompoundResult
from actions.general_llm_actions import GeneralInvestmentQuery
from prompts import GET_FIELD_FROM_STOCK_INFO_PROMPT, SECTOR_EXTRACT_PROMPT
from utils.news_utils import get_news_for_stock
from utils.llm_utils import ask_openai, extract_time_period_from_query, extract_top_n_from_query, is_not_none
from utils.finance_utils import plot_stock_chart
from utils.llm_utils import is_not_none
from actions.stock_buying_actions import WhichStocksToBuy
from utils.finance_utils import merge_results

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


class GetETFPrice(StockAction):

    def get_description(self) -> str:
        return """
            Get the current price of an ETF.
            Example queries:
            - What's the price of SPY?
            - How is the price of QQQ doing?
            - What's the price of the S&P 500?
        """
    
    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        # stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)
        return StockActionResult("ETF Price demo", "html")
        

class GetStockPrice(StockAction):

    def __init__(self):
        self.stock_price = None

    def get_description(self) -> str:
        return """
            Get the current price of a stock.
            You can also ask for the price of a stock in the last X days|weeks|months|years.
            Example queries:
            - What's the price of AAPL?
            - How has the price of TSLA been in the last month?
            - What about NVDA in the last year?
        """

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


class GetStockPerformance(StockAction):

    def get_description(self) -> str:
        return """
            Ask for the performance of a particular stock or stocks, optionally in the last X days|weeks|months|years.
            Example queries:
            - How has the price of AAPL been in the last 2 weeks?
            - How is Apple doing?
            - How has NVDA done in the last month?
        """

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

        # prediction = plot_stock_chart(stock_symbol, time_window=time_period)
        get_stock_price_action_result = GetStockPrice().execute(user_phrase, stock_symbol)

        stock_performance_action_result = StockActionCompoundResult([
            "Here's more recent data from the past week", 
            history
        ], 
        ["html", "dataframe"])

        return merge_results([get_stock_price_action_result, stock_performance_action_result])


class CompareStocks(StockAction):

    def get_description(self) -> str:
        return """
            Compare the performance of two stocks.
            Example queries:
            - Compare AAPL and TSLA
            - How is apple doing compared to tesla?
            - Do a comparison of NVDA and MSFT
        """

    def execute(self, user_phrase: str, stock_symbols: str) -> StockActionResult:
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

        return StockActionCompoundResult(
            [
                f"Alright, here is a comparison of {stock1} and {stock2}:",
                df
            ],
            ["html", "dataframe"]
        )


class CompanyFinanceQuestionAndAnswer(StockAction):

    def get_description(self) -> str:
        return """
            Ask a question about a company's financials.
            Choose from the following topics:
            'sector', 'fullTimeEmployees', 'companyOfficers', 'auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk', 'overallRisk', 'governanceEpochDate', 'compensationAsOfEpochDate', 'irWebsite', 'executiveTeam', 'maxAge', 'priceHint', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'exDividendDate', 'payoutRatio', 'fiveYearAvgDividendYield', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield', 'currency', 'tradeable', 'enterpriseValue', 'profitMargins', 'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth', 'sharesShortPreviousMonthDate', 'dateShortInterest', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'impliedSharesOutstanding', 'bookValue', 'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon', 'trailingEps', 'forwardEps', 'lastSplitFactor', 'lastSplitDate', 'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'lastDividendDate', 'quoteType', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'grossProfits', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins', 'financialCurrency', 'symbol', 'language', 'region', 'typeDisp', 'quoteSourceName', 'triggerable', 'customPriceAlertConfidence', 'shortName', 'longName', 'regularMarketChangePercent', 'regularMarketPrice', 'corporateActions', 'postMarketTime', 'regularMarketTime', 'exchange', 'messageBoardId', 'exchangeTimezoneName', 'exchangeTimezoneShortName', 'gmtOffSetMilliseconds', 'market', 'esgPopulated', 'marketState', 'sourceInterval', 'exchangeDataDelayedBy', 'averageAnalystRating', 'cryptoTradeable', 'hasPrePostMarketData', 'firstTradeDateMilliseconds', 'postMarketChangePercent', 'postMarketPrice', 'postMarketChange', 'regularMarketChange', 'regularMarketDayRange', 'fullExchangeName', 'averageDailyVolume3Month', 'fiftyTwoWeekLowChange', 'fiftyTwoWeekLowChangePercent', 'fiftyTwoWeekRange', 'fiftyTwoWeekHighChange', 'fiftyTwoWeekHighChangePercent', 'fiftyTwoWeekChangePercent', 'dividendDate', 'earningsTimestamp', 'earningsTimestampStart', 'earningsTimestampEnd', 'earningsCallTimestampStart', 'earningsCallTimestampEnd', 'isEarningsDateEstimate', 'epsTrailingTwelveMonths', 'epsForward', 'epsCurrentYear', 'priceEpsCurrentYear', 'fiftyDayAverageChange', 'fiftyDayAverageChangePercent', 'twoHundredDayAverageChange', 'twoHundredDayAverageChangePercent', 'displayName', 'trailingPegRatio'
            Example queries:
            - How much revenue does Apple make?
            - What's the net income of Tesla?
            - How is Apple doing financially?
            - What is Google's P/E ratio?
            - How many employees does Apple have?
            - What is the overall risk of Apple?
        """

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


class GetTopPerformers(StockAction):

    def get_description(self) -> str:
        return """
            Get the top performing stocks , either generally or in a particular sector.
            Assume that the user is hinting at asking for buying recommendations.
            Example queries:
            - What are the top performing stocks?
            - What are the top performing stocks in the S&P 500?
            - What are the top performing stocks in the healthcare sector?
        """
    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        st.markdown("Let's see what the top performers are...")
        N = extract_top_n_from_query(user_phrase)

        if is_not_none(N):
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

    def get_description(self) -> str:
        return """
            Get the trends of the market or a particular sector.
            Example queries:
            - What's the trend of the S&P 500?
            - How is the construction industry doing this year?
            - How is the S&P 500 doing?
            - How is the NASDAQ doing?
            - What about the Dow Jones?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:

        st.markdown("📈 🤓 Analyzing market trends...")
        
        predictions = []

        # sp500, nasdaq, dow
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
        sentiment_data, _ = get_news_for_stock("^GSPC")

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
