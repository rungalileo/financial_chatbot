import os
from openai import OpenAI
from prompts import TIME_PERIOD_EXTRACT_PROMPT
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def ask_openai(
    user_content,
    system_content="You are a smart assistant", 
    model="gpt-4o-mini"
):

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        api_key = st.secrets["OPENAI_API_KEY"]

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    )
    output = response.choices[0].message.content.replace("```markdown", "").replace("```code", "").replace("```html", "").replace("```", "")
    return output

def extract_time_period_from_query(user_phrase: str) -> str:
    time_period_extraction_prompt = TIME_PERIOD_EXTRACT_PROMPT.format(user_phrase=user_phrase)
    time_period = ask_openai(user_content=time_period_extraction_prompt).strip().lower()
    return time_period

def is_not_none(string: str) -> bool:
    return string != "None" and string != "none"

def extract_top_n_from_query(user_phrase: str) -> int:
    prompt = f"""
        Extract the number N from a "top N" user query.
        Input: "{user_phrase}"
        Instructions:
        - Identify the number associated with phrases like "top N," "best N," "first N," or similar patterns.
        - If a relevant number is found, return it as an integer.
        - If no valid number is found, return the the number -1.
    """
    response = ask_openai(user_content=prompt)
    # check if the response is a valid integer
    if response.isdigit():
        return int(response)
    else:
        return -1
