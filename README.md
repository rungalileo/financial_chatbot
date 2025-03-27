
Create a .env file and add the following lines
```
OPENAI_API_KEY=your_openai_key
NEWSAPI_API_KEY=your_newsapi_key
```

Get the news API key from: https://newsapi.org/

Create a virtual environment

Run the following:
```
1/ pip install -r requirements.txt
2/ add a .streamlit folder in the project
3/ add a secrets.toml file in that folder (empty)
4/ add a secrets.toml file in your home streamlit i.e. ~/.streamlit/secrets.toml (create folder if not exists)
5/ stremalit run app.py
```
