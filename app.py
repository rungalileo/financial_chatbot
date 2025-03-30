from datetime import datetime
from zoneinfo import ZoneInfo
import json
from json import tool
import streamlit as st
from utils.actions_utils import StockActionExecutor, ResultRenderer, StockActionResult, StockActionCompoundResult
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain.schema import AIMessage
from dotenv import load_dotenv
from tools import STOCK_ACTIONS
from prompts import TOOL_SELECTION_PROMPT, GET_STOCK_SYMBOL_PROMPT, FINANCE_QUERY_CLASSIFICATION_PROMPT
from utils.agent_state import StockAgentState
import os
from galileo_core.schemas.logging.llm import Message, ToolCall, ToolCallFunction
from langchain_core.runnables import RunnableConfig
from galileo import GalileoLogger
import inspect


load_dotenv()

galileo_logger = GalileoLogger(project=os.getenv("GALILEO_PROJECT"), log_stream=os.getenv("GALILEO_LOG_STREAM"))

action = StockActionExecutor()

main_llm = os.getenv("LLM")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

########################################################
# ROUTER
########################################################

def is_finance_query(query: str, stock_to_use: str, allowed_topics: list[str], last_query: str, last_action: str) -> bool:

    classification_prompt = FINANCE_QUERY_CLASSIFICATION_PROMPT.format(
        query=query,
        stock_to_use=stock_to_use,
        last_query=last_query,
        last_action=last_action,
        allowed_topics=allowed_topics
    )
    response = llm.invoke(classification_prompt).content.strip().lower()
    return response == "yes"

def extract_stock_symbol_llm(user_input: str) -> str:
    llm_prompt = GET_STOCK_SYMBOL_PROMPT.format(user_phrase=user_input)
    response = llm.invoke(llm_prompt).content.strip().upper()
    return None if response == "NONE" else response

def route_stock_action(state: StockAgentState) -> StockAgentState:
    """ Router function made of multiple LLM calls + some logic.
    Goal: select the appropriate tool to call based on the user input
    Steps:
        1. Extract stock symbol from user input (llm call, extract_stock_symbol_llm)
        2. Check if the user input is a finance query (llm call, is_finance_query)
        3. If not, return fallback response
        4. If yes, select the appropriate tool (llm call) -> only return tool name
    """

    user_input = state["input"]
    last_stock = state.get("last_stock_symbol", None)
    last_query = state.get("last_query", None)
    last_action = state.get("last_action", None)

    extracted_stock = extract_stock_symbol_llm(user_input)
    stock_to_use = extracted_stock if extracted_stock else last_stock

    # loop through all the keys, get the Stock Action and concatenate the outputs of get_description
    stock_action_descriptions = []
    for action_name in STOCK_ACTIONS.keys():
        stock_action_descriptions.append(action_name)

    if not is_finance_query(user_input, stock_to_use, stock_action_descriptions, last_query, last_action):
        print(f"[DEBUG][ROUTER] Not finance related! Returning fallback tool.")
        return {**state, "action": "fallback_response"}

    tool_selection_prompt = TOOL_SELECTION_PROMPT.format(
        user_phrase=user_input,
        stock_to_use=stock_to_use,
        last_query=last_query,
        last_action=last_action,
        last_stock=last_stock,
        stock_actions=list(STOCK_ACTIONS.keys())
    )
 
    llm_response = llm.invoke(tool_selection_prompt)

    tool_response = llm_response.content.strip().lower() if isinstance(llm_response, AIMessage) else str(llm_response).strip().lower()
    print(f"[DEBUG][ROUTER] TOOL SELECTED: {tool_response}")

    new_state = {
        **state,
        "action": tool_response,
        "last_stock_symbol": stock_to_use,
        "last_query": user_input,
        "last_action": state.get("action", None),
        "result": state.get("result", None),
    }
    
    # Log manually here since we need to assemble the LLM's output as tool name + selected args (even if they come from different calls)
    tool_call_args = json.dumps( # TODO: some hardcoding seems necessary since tool input args != state keys
        {
            "user_phrase": new_state.get("input", None),
            "stock_to_use": new_state.get("last_stock_symbol", None)
        }
    )
    tool_response_message = Message(
        role="assistant", 
        content="", 
        tool_call_id= "fake_id", 
        tool_calls=[ToolCall(id=tool_response, function=ToolCallFunction(name=tool_response, arguments=tool_call_args))])
    
    galileo_logger.add_llm_span(
        input=tool_selection_prompt,
        output=tool_response_message,
        model=llm.model_name,
        name=f"Router",
        tools=[{k:v.get_description()} for k,v in STOCK_ACTIONS.items()]
    )

    return new_state

########################################################
# EXECUTOR
########################################################

def execute_stock_action(state: StockAgentState) -> StockAgentState:
    """ Execute the selected tool.
    Steps:
        1. Extract stock symbol from user input (llm call, extract_stock_symbol_llm)
        2. Execute the selected tool (STOCK_ACTIONS[action_name].execute)
        3. Return the result
    """
    action_name = state.get("action", None)
    user_phrase = state.get("input", None)
    stock_to_use = state.get("last_stock_symbol", None)

    result = None

    if action_name in STOCK_ACTIONS:
        result = STOCK_ACTIONS[action_name].execute(user_phrase, stock_to_use)
        
        galileo_logger.add_tool_span(
            input=json.dumps({"user_phrase": user_phrase, "stock_to_use": str(stock_to_use)}),
            output=json.dumps(result.to_dict()),
            name=action_name,
        )

        extracted_stock = extract_stock_symbol_llm(user_phrase)
        if extracted_stock:
            stock_to_use = extracted_stock
    else:
        result = "No valid action found."

    new_state = {
        **state,
        "result": result,
        "last_stock_symbol": stock_to_use,
        "last_query": user_phrase,
        "last_action": action_name,
    }

    return new_state

########################################################

def main():
    
    with st.container() as ST:

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if type(message["content"]) == StockActionResult:
                    ResultRenderer.render(message["content"])
                elif type(message["content"]) == StockActionCompoundResult:
                    ResultRenderer.render(message["content"])
                else:
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask me about stocks 🚀, maybe start with 'What can you do?' 📈"):
            n_trace = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
            galileo_logger.start_trace(
                input=prompt,
                name=f"User Interaction {n_trace}",
                created_at=datetime.now(ZoneInfo("America/Detroit"))
            )            
            
                
            st.chat_message("user").markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})

            if "graph_state" not in st.session_state:
                st.session_state.graph_state = {"last_stock_symbol": None, "last_query": None, "last_action": None}

            workflow = StateGraph(StockAgentState)
            workflow.add_node("router", route_stock_action)
            workflow.add_node("executor", execute_stock_action)
            workflow.add_edge("router", "executor")
            workflow.set_entry_point("router")
            graph = workflow.compile()

            print("INVOKING DAG...")

            cfg = RunnableConfig()

            response = graph.invoke(
                {
                    "input": prompt, 
                    "last_stock_symbol": st.session_state.graph_state["last_stock_symbol"],
                    "last_query": st.session_state.graph_state["last_query"],
                    "last_action": st.session_state.graph_state["last_action"]
                },
                config=cfg
            )

            stock_action_result = response.get("result")
            
            st.session_state.graph_state["last_stock_symbol"] = response.get("last_stock_symbol", None)
            st.session_state.graph_state["last_query"] = response.get("last_query", None)
            st.session_state.graph_state["last_action"] = response.get("last_action", None)

            with st.chat_message("assistant"):
                if stock_action_result:
                    ResultRenderer.render(stock_action_result)
                    st.session_state.messages.append({"role": "assistant", "content": stock_action_result})
                else:
                    st.error(f"Agent did not return a tool response: {response}")

            galileo_logger.conclude(
                output=json.dumps(stock_action_result.to_dict()),
                # conclude_all=False  # Whether to conclude all open spans
            )
            galileo_logger.flush()

if __name__ == "__main__":
    main()