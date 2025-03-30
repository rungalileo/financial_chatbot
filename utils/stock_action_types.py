from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from utils.agent_state import StockAgentState
from dotenv import load_dotenv
import os
import json
from dataclasses import dataclass

load_dotenv()

@dataclass
class StockActionCompoundResult:
    data_structures: list[Any]
    output_types: list[str]

    def __init__(self, data_structures: list[Any], output_types: list[str]):
        self.data_structures = data_structures
        self.output_types = output_types

    def to_dict(self):
        # Bogdan: not used anywhere so I'm repurposing it to stringify the data structures for logging to Galileo
        data_structures_str = []
        for structure in self.data_structures:
            try:
                if isinstance(structure, str):
                    structure_str = structure
                elif isinstance(structure, pd.core.frame.DataFrame):
                    structure_str = json.dumps(structure.to_dict(orient="records"))
                else:
                    structure_str = json.dumps(structure)
            except Exception as e:
                structure_str = f"Couldn't stringify structure of type {type(structure)}"
            data_structures_str.append(structure_str)
            
        return {
            "data_structures": data_structures_str,
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

    @abstractmethod
    def get_description(self) -> str:
        pass

class StockActionWithState(ABC):
    @abstractmethod
    def execute(self, user_phrase: str, stock_symbol: str, state: StockAgentState) -> StockActionResult:
        pass
