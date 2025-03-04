import streamlit as st
from utils.actions_utils import StockActionExecutor, ResultRenderer, StockActionResult, StockActionCompoundResult
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain.schema import AIMessage
from dotenv import load_dotenv
from tools import STOCK_ACTIONS

# from langchain.agents import initialize_agent, AgentType
# from tools import stock_price_tool, stock_history_tool, tell_me_about_company_tool, get_news_tool, is_worth_buying_tool, get_top_performers_tool

load_dotenv()

action = StockActionExecutor()


## TOOL SETUP
# tools = [
#     stock_price_tool, 
#     stock_history_tool, 
#     tell_me_about_company_tool, 
#     get_news_tool, 
#     is_worth_buying_tool, 
#     get_top_performers_tool
# ]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=1,
#     return_intermediate_steps=True
# )

class StockAgentState(TypedDict):
    input: str
    action: str
    result: str
    last_stock_symbol: str
    last_query: str
    last_action: str

########################################################
# ROUTER
########################################################

def is_finance_query(query: str, stock_to_use: str, allowed_topics: list[str], last_query: str, last_action: str) -> bool:

    classification_prompt = (
        f"User Query: '{query}'. \n\n"
        f"Stock being discussed: {stock_to_use}. \n\n"
        f"Last query: '{last_query}'. \n\n"
        f"Last action taken: '{last_action}'. \n\n"
        f"Based on the query, stock (if any), last query, and last action, determine if the following user question and stock combination is related to the following  topics: {allowed_topics}\n"
        f"Reply with only 'yes' or 'no'."
    )
    print(f"[DEBUG][ROUTER] IS FINANCE PROMPT: {classification_prompt}")
    response = llm.invoke(classification_prompt).content.strip().lower()
    return response == "yes"

def extract_stock_symbol_llm(user_input: str) -> str:
    llm_prompt = f"""
    You are a financial expert specializing in stock market data.
	- The user query is: {user_input}
	- The user might be asking about a stock symbol or referring to a company name.
	- Your task is to extract the relevant stock symbol (e.g., AAPL for Apple, TSLA for Tesla).

    Instructions:
	1.	If the query directly mentions a stock symbol, return the exact symbol (e.g., “AAPL” if the user input is “What is AAPL trading at?”).
	2.	If the query refers to a company name, return the corresponding stock symbol (e.g., “TSLA” for “Tesla”).
	3.	If the query does not clearly reference a specific stock or company, return “None” (without quotes).

    Output format:
	- Return ONLY the stock symbol in uppercase (e.g., “AAPL”).
	- If no valid stock symbol is found, return “None”.
	- Do not include any explanations, extra text, or formatting.
    """
    response = llm.invoke(llm_prompt).content.strip().upper()
    return None if response == "NONE" else response


def route_stock_action(state: StockAgentState) -> StockAgentState:

    user_input = state["input"]
    last_stock = state.get("last_stock_symbol", None)
    last_query = state.get("last_query", None)
    last_action = state.get("last_action", None)

    extracted_stock = extract_stock_symbol_llm(user_input)
    stock_to_use = extracted_stock if extracted_stock else last_stock

    if not is_finance_query(user_input, stock_to_use, list(STOCK_ACTIONS.keys()), last_query, last_action):
        print(f"[DEBUG][ROUTER] Not finance related! Returning fallback tool.")
        return {**state, "action": "fallback_response"}

    tool_selection_prompt = f"""
        User asked the question: '{user_input}'.
        The stock being discussed is: {stock_to_use}.

        Conversation history:
        - Previous query: {last_query}
        - Previous action: {last_action}
        - Last stock symbol: {last_stock}

        Based on the user query, select the BEST tool possible to handle this query from this list of tools:
        {list(STOCK_ACTIONS.keys())}.

        INSTRUCTIONS:
        1. Use previous query and previous action as a guideline, but not a strict rule.
        2. Use best judgment to choose the most relevant tool.
        3. If the last stock symbol and the current stock symbol are different, give preference to the current stock symbol.
        4. If both are none or empty, don't choose any tool about a particular stock.
        5. ONLY choose 'fallback_response' when you are unsure.
        6. ONLY return the tool name.
    """

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

    return new_state

########################################################
# EXECUTOR
########################################################

def execute_stock_action(state: StockAgentState) -> StockAgentState:
    action_name = state.get("action", None)
    user_phrase = state.get("input", None)
    stock_to_use = state.get("last_stock_symbol", None)

    result = None

    if action_name in STOCK_ACTIONS:
        result = STOCK_ACTIONS[action_name].execute(user_phrase, stock_to_use)

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

if prompt := st.chat_input("Ask me about stocks 🚀, maybe start with 'How is Apple stock doing?' 📈"):

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
 
    response = graph.invoke(
        {
            "input": prompt, 
            "last_stock_symbol": st.session_state.graph_state["last_stock_symbol"],
            "last_query": st.session_state.graph_state["last_query"],
            "last_action": st.session_state.graph_state["last_action"]
        }
    )

    # response = graph.invoke({"input": prompt})
    # response = workflow.invoke({"input": prompt})
    if "result" in response:
        stock_action_result = response["result"]
    else:
        stock_action_result = None

    st.session_state.graph_state["last_stock_symbol"] = response.get("last_stock_symbol", None)
    st.session_state.graph_state["last_query"] = response.get("last_query", None)
    st.session_state.graph_state["last_action"] = response.get("last_action", None)

    with st.chat_message("assistant"):
        if stock_action_result:
            ResultRenderer.render(stock_action_result)
            st.session_state.messages.append({"role": "assistant", "content": stock_action_result})
        else:
            st.error(f"Agent did not return a tool response: {response}")

        # if "intermediate_steps" in response and response["intermediate_steps"]:
        #     tool_output = response["intermediate_steps"][0][1]  # Get first tool response
        #     if isinstance(tool_output, dict) and "output_type" in tool_output:
        #         result = StockActionResult.from_dict(tool_output)
        #         ResultRenderer.render(result)
        #         st.session_state.messages.append({"role": "assistant", "content": result})
        #     else:
        #         st.error(f"Unexpected tool output format: {tool_output}")
        # else:
        #     st.error(f"Agent did not return a tool response: {response}")
