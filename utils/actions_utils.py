from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import streamlit as st
import pandas as pd
from functools import singledispatchmethod
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from utils.stock_action_types import StockAction, StockActionResult, StockActionCompoundResult

load_dotenv()


class StockActionExecutor:

    def set_action(self, action: StockAction):
        self.action = action

    def run(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        return self.action.execute(self, user_phrase=user_phrase, stock_symbol=stock_symbol)

class ResultRenderer:

    @singledispatchmethod
    @staticmethod
    def render(result):
        raise TypeError(f"Unsupported result type: {type(result)}")

    @render.register
    @staticmethod
    def _(result: StockActionCompoundResult):
        for ds, output_type in zip(result.data_structures, result.output_types):
            ResultRenderer.render(StockActionResult(data=ds, output_type=output_type))

    @render.register
    @staticmethod
    def _(result: StockActionResult):
        if isinstance(result, dict) and 'output' in result:
            st.markdown(result['output'], unsafe_allow_html=True)
        elif result.output_type == "html":
            st.markdown(result.data, unsafe_allow_html=True)
        elif result.output_type == "dataframe":
            st.table(result.data)
        elif result.output_type == "chart":
            if isinstance(result.data, plt.Figure):
                st.pyplot(result.data)
            else:
                print(f"[DEBUG] Could not display chart. Output type: {type(result.data)}")
        else:
            st.error("Unknown output type")