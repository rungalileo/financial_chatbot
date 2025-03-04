import os
from openai import OpenAI


def ask_openai(
    user_content,
    system_content="You are a smart assistant", 
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-4o-mini"
):
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
    time_period_extraction_prompt = f"""
        The user asked: "{user_phrase}".
        Identify the time period mentioned in this query.
        It can be ONLY amongst the following time periods: [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max].
        There is no "week", only days (d), months (mo), years (y), ytd (year to date) and max.
        Approximate the time period to the nearest time period in the list above.
        Return ONLY the time period name e.g. 1d, 5d, 1mo, 3mo etc.
    """
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
