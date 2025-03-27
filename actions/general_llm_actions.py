from utils.stock_action_types import StockAction, StockActionResult
from utils.llm_utils import ask_openai
import streamlit as st

class GeneralInvestmentQuery(StockAction):

    def get_description(self) -> str:
        return """
            Ask a general question about the stock market, investing, or finance.
            Example queries:
            - What is the best way to diversify my investment portfolio?
            - How should I invest in the stock market?
            - What are the key differences between ETFs and mutual funds?
            - What are some common mistakes new investors make?
            - How do stock buybacks impact share prices?
            - How can I generate passive income through investments?
            - What factors drive stock market crashes?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        intermediate_response = st.markdown(f"Answering your query...")
        prompt = f"Answer in brief, the following question: {user_phrase}, Stock symbol: {stock_symbol}"
        response = ask_openai(user_content=prompt)
        intermediate_response.markdown("")
        return StockActionResult(response, "html")


class WhatCanYouDo(StockAction):

    def get_description(self) -> str:
        return """
            Ask what I can do.
            Example queries:
            - What can you do?
            - What are your capabilities?
            - What more can you do?
            - What are you capable of?
            - Hi, what can you do?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        from tools import STOCK_ACTIONS
        action_names = list(STOCK_ACTIONS.keys())
        action_names = [name for name in action_names if name not in ["fallback_response", "what_can_you_do"]]
        action_names = [name.replace("_", " ").capitalize() for name in action_names]
        action_names_string = ", ".join(action_names)
        action_names_string = "<ul>" + "\n".join([f"<li>{name}</li>" for name in action_names]) + "</ul>"
        return StockActionResult(f"I can help you with the following: {action_names_string}", "html")


class FallbackResponse(StockAction):

    def get_description(self) -> str:
        return """
            Fallback response for when the user asks a question that is not related to stocks, investing, or finance.
            Or any garbage non-sense query would fall under this category.
            This includes all canned dialogs.
            Example queries:
            - What is the weather in New York?
            - What is the capital of France?
            - What is the meaning of life?
        """

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        return StockActionResult(
            "I'm not programmed to answer this. Ask me about stocks, investing, or just ask me what I can do.", "html")
