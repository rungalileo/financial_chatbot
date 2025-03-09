from utils.stock_action_types import StockAction, StockActionResult
from utils.llm_utils import ask_openai
import streamlit as st

class GeneralInvestmentQuery(StockAction):

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        intermediate_response = st.markdown(f"Answering your query...")
        prompt = f"Answer in brief, the following question: {user_phrase}, Stock symbol: {stock_symbol}"
        response = ask_openai(user_content=prompt)
        intermediate_response.markdown("")
        return StockActionResult(response, "html")


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
