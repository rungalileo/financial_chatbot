import galileo
from galileo.openai import openai
from galileo import log
from dotenv import load_dotenv
import os


load_dotenv()

logstream_name = f"testy_aqua_lynx"
llm = os.getenv("LLM")

def make_openai_call(input_text):
    response = openai.chat.completions.create(
        model=llm,
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f"Answer the question: {input_text}"}
        ],
    )
    return response.choices[0].message.content

@log(log_stream=f"{logstream_name}_{llm}")
def evaluate_response(input_text, response):
    print("Hello world")
    x = make_openai_call(input_text)
    print(f"Response: {x}")
    return True

def main():
    input_text = "What is the capital of France?"
    response = make_openai_call(input_text)
    evaluate_response(input_text, response)

if __name__ == "__main__":
    main()
