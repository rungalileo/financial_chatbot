from typing import TypedDict

class StockAgentState(TypedDict):
    input: str
    action: str
    result: str
    last_stock_symbol: str
    last_query: str
    last_action: str

