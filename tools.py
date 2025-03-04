# from langchain.tools import Tool

from actions.stock_actions import (
    GetStockPrice, 
    GetStockPerformance, 
    TellMeAboutThisCompany, 
    GetNewsAndSentiment, 
    IsStockWorthBuying, 
    GetTopPerformers, 
    WhichStocksToBuy, 
    GetMarketOrSectorTrends, 
    GeneralInvestmentQuery, 
    CompanyFinanceQuestionAndAnswer, 
    FallbackResponse,
    WhatCanYouDo
)

STOCK_ACTIONS = {
    "get_stock_price": GetStockPrice(), # Finance API
    "get_stock_performance": GetStockPerformance(), # Finance API
    "financial_question_about_stock_or_company": CompanyFinanceQuestionAndAnswer(), # Finance API
    "which_stocks_to_buy_or_invest_in": WhichStocksToBuy(), # Finance API
    "get_stock_news_and_sentiment": GetNewsAndSentiment(), # Finance & News API
    "tell_me_about_company_or_stock": TellMeAboutThisCompany(), # Finance & News API
    "company_finance_question_and_answer": CompanyFinanceQuestionAndAnswer(), # Finance API
    "is_stock_worth_buying": IsStockWorthBuying(), # Finance & News API
    "get_top_stocks_performing_in_market": GetTopPerformers(), # Finance API
    "get_market_or_sector_trends": GetMarketOrSectorTrends(), # Finance API
    "general_investment_query": GeneralInvestmentQuery(), # General LLM
    "what_can_you_do": WhatCanYouDo(), # General LLM
    "fallback_response": FallbackResponse() # General LLM
}

# def create_tool(action_class, name, description):

#     def tool_function(stock_symbol: str):
#         action = action_class()
#         result = action.execute(stock_symbol)
#         output = result.to_dict()
#         print(f"[DEBUG] Tool output for {stock_symbol}: {output}")
#         return output

#     return Tool(
#         name=name,
#         func=tool_function,
#         description=description
#     )


# get_news_tool = create_tool(
#     GetNewsAndSentiment,
#     "get_news_and_sentiment",
#     """
#     Get the current news and market sentiment about the stock. 
#     Queries can be about the stock or the company.
#     E.g. 
#     "What is the news of Apple?"
#     "What is the news of AAPL?"
#     "What's the latest news on Google?"
#     "What's the market sentiment on Tesla?"
#     """
# )

# stock_price_tool = create_tool(
#     GetStockPrice,
#     "get_stock_price",
#     """
#     Get the price of a stock. Queries should include the name of a company or a stock symbol. 
#     E.g. "What is the price of Apple?" or "What is the price of AAPL?"
#     This tool must always be used for stock prices.
#     """
# )

# stock_history_tool = create_tool(
#     GetStockHistory,
#     "get_stock_history",
#     """
#     Get the history of a stock. Queries should include the name of a company or a stock symbol. 
#     E.g. "What is the history of Apple?" or "What is the history of AAPL?"
#     This tool must be used to get past data about a stock (could be last 5 days, or last 5 years)
#     The output is always a pandas dataframe.
#     """
# )

# tell_me_about_company_tool = create_tool(
#     TellMeAboutThisCompany,
#     "tell_me_about_company",
#     """
#     Getting information or status or generally asking about a company. 
#     Queries should include the name of a company or a stock symbol. 
#     E.g. Tell me about Apple? Or How is Apple doing? or How well is Apple doing these days?
#     This tool must be used if a user asks for inforamtion about the company.
#     """
# )

# is_worth_buying_tool = create_tool(
#     IsStockWorthBuying,
#     "is_worth_buying",
#     """
#     User is asking if a stock is worth buying or selling.
#     Queries should include the name of a company or a stock symbol. 
#     E.g. Is Apple worth buying? Or Is AAPL worth buying?
#     Or Should I be buying Tesla? etc.
#     This tool must be used if a user asks if a stock is worth buying or selling.
#     """
# )

# get_top_performers_tool = create_tool(
#     GetTopPerformers,
#     "get_top_performers",
#     """
#     Get the top performing stocks in the market.
#     The tool should extract the name of the index and the time range from the user's query.
#     E.g. "What are the top performing stocks in the S&P 500 in the last 1 year?"
#     Here, S&P 500 is the index and last '1y' is the time range.
#     """
# )
