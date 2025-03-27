from utils.stock_action_types import StockAction
from utils.finance_utils import get_stock_symbol_from_user_phrase, merge_results
from utils.news_utils import get_news_for_stock, get_market_news
from utils.llm_utils import is_not_none
from utils.finance_utils import plot_stock_chart
from utils.stock_action_types import StockActionResult, StockActionCompoundResult
from utils.finance_utils import simplify_dollar_amount, worth_buying, plot_stock_chart
from actions.error_dialogs import CAN_ONLY_DO_PERF_INFO_ON_SPECIFIC_STOCKS, PLEASE_SPECIFY_A_STOCK
import streamlit as st
import yfinance as yf


class GetMarketNews(StockAction):

    def get_description(self) -> str:
        return """
            Get the latest news on the market.
            Example queries:
            - How's the market doing today?
            - What's going on in the markets?
            - What's the latest news on the market?
            - What's the latest news on the S&P 500?
            - What's the latest news on the NASDAQ?
            - What's the latest news on the Dow Jones?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        authors, headlines = get_market_news()

        html_table = f"""
        <table>
            <tr><th style='text-align: left;'>Authors</th><td>{authors}</td></tr>
            <tr><th style='text-align: left;'>Headlines</th><td>{headlines}</td></tr>
        </table>
        """

        return StockActionCompoundResult([
            html_table,
        ],
        ["html"])
        

class GetNewsAndSentiment(StockAction):

    def get_description(self) -> str:
        return """
            Get the latest news and/or sentiment about a particular stock.
            Example queries:
            - What's the latest news about AAPL?
            - What is the sentiment of Tesla?
            - What is the latest on Google?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return StockActionResult(PLEASE_SPECIFY_A_STOCK, "html")

        st.markdown(f"Let's see what the market is saying about {stock_symbol}...")

        print(f"STOCK SYMBOL: {stock_symbol}")

        recommendation, authors = get_news_for_stock(stock_symbol)

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


class TellMeAboutThisCompany(StockAction):

    def get_description(self) -> str:
        return """
            Get a brief financial and business summary about a particular company.
            The user can either mention a company, or a stock symbol.
            Example queries:
            - Tell me about Apple
            - What does Apple do?
            - Tell me about GOOG
        """

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
            <h5>Financial Summary</h5>
            <p>{stock_symbol} is a {city}-based company, specializing in {industry}.</p>
            <p>It has a market cap of <span style="color: green;">{market_cap}</span>. 
            Total ARR is <span style="color: green;">{arr}</span> and it holds 
            <span style="color: blue;">{total_cash}</span> in cash.</p>
            <p>Its 52-week change is <span style="color: red;">{week_change:.2f}%</span>, 
            and its stock value today is <span style="color: red;">${stock_value_today:,.2f}</span>.</p>
        """
        long_business_summary = stock_info.get("longBusinessSummary", "Unknown History")

        additional_info = f"""
            <h5>Business Summary</h5>
            <p>{long_business_summary}</p>
        """

        news_result = GetNewsAndSentiment().execute(user_phrase, stock_symbol)

        financial_and_business_summary = StockActionCompoundResult([
            html_content,
            additional_info,
        ],
        ["html", "html"])

        return merge_results(stock_action_compound_results=[
            financial_and_business_summary, 
            StockActionCompoundResult(
                ["<h5>News</h5>"],
                ["html"]
            ),
            news_result,
        ])


class IsStockWorthBuying(StockAction):

    def get_description(self) -> str:
        return """
            Determine if a particular stock is worth buying.
            Example queries:
            - Is Tesla worth buying?
            - Is Google worth buying?
            - Should I buy AAPL?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        stock_symbol = get_stock_symbol_from_user_phrase(stock_symbol=stock_symbol, user_phrase=user_phrase)

        if stock_symbol == "None":
            return StockActionResult(CAN_ONLY_DO_PERF_INFO_ON_SPECIFIC_STOCKS, "html")

        intermediate_response = st.markdown(f"Let's see if {stock_symbol} is worth buying...")
        recommendation, authors = get_news_for_stock(stock_symbol)

        res = worth_buying(stock_symbol)
        

        stock_chart = plot_stock_chart(stock_symbol, '2y')

        main_response = f"""
            <p>Pure stats says that {stock_symbol} is a {res}. 
            And according to recent news, {stock_symbol} is a {recommendation['rating']}.</p>
        """

        stock_news_data, authors = get_news_for_stock(stock_symbol)
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
