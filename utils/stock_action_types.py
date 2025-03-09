from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from utils.agent_state import StockAgentState
from dotenv import load_dotenv
import os

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

class StockActionWithState(ABC):
    @abstractmethod
    def execute(self, user_phrase: str, stock_symbol: str, state: StockAgentState) -> StockActionResult:
        pass
