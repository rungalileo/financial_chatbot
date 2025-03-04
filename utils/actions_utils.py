from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import streamlit as st
import pandas as pd
from functools import singledispatchmethod
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

class StockActionCompoundResult:
    def __init__(self, data_structures: list[Any], output_types: list[str]):
        self.data_structures = data_structures
        self.output_types = output_types

    def to_dict(self):
        return {
            "data_structures": [structure.to_dict() for structure in self.data_structures],
            "output_types": self.output_types
        }

    @staticmethod
    def from_dict(result_dict):
        data_structures = [StockActionResult.from_dict(structure) for structure in result_dict["data_structures"]]
        return StockActionCompoundResult(data_structures, result_dict["output_types"])

class StockActionResult:
    def __init__(self, data: Any, output_type: str):
        self.data = data
        self.output_type = output_type

    def to_dict(self):
        if isinstance(self.data, pd.DataFrame):
            return {
                "data": self.data.to_dict(orient="records"),
                "output_type": self.output_type
            }
        return {"data": self.data, "output_type": self.output_type}

    @staticmethod
    def from_dict(result_dict):
        if not isinstance(result_dict, dict):
            raise ValueError("Invalid dictionary format for StockActionResult")

        data = result_dict["data"]
        if result_dict["output_type"] == "dataframe":
            data = pd.DataFrame(data)
        return StockActionResult(data, result_dict["output_type"])

class StockAction(ABC):
    @abstractmethod
    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        pass

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