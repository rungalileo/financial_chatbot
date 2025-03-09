from actions.finance_actions import (
    GetStockPrice, 
    GetStockPerformance, 
    GetTopPerformers, 
    GetMarketOrSectorTrends, 
    GeneralInvestmentQuery, 
    CompanyFinanceQuestionAndAnswer, 
    CompareStocks
)
from actions.finance_news_actions import (
    GetNewsAndSentiment,
    TellMeAboutThisCompany,
    IsStockWorthBuying
)
from actions.general_llm_actions import (
    WhatCanYouDo,
    FallbackResponse
)
from actions.stock_buying_actions import (
    WhichStocksToBuy
)

STOCK_ACTIONS = {
    "get_stock_price": GetStockPrice(), # Finance API
    "get_stock_performance": GetStockPerformance(), # Finance API
    "compare_two_different_stocks": CompareStocks(), # Finance API
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
