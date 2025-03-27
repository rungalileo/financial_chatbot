import spacy
import faiss
import pickle
import numpy as np
import yfinance as yf
import pandas as pd
import streamlit as st
import time
import random
from utils.llm_utils import ask_openai, extract_top_n_from_query
from prompts import SECTOR_EXTRACT_PROMPT
from utils.stock_action_types import StockAction, StockActionResult

CSV_FILE = "stock_industries.csv"
FAISS_INDEX_FILE = "faiss_company_index.bin"
PICKLE_VECTORS_FILE = "company_vectors.pkl"
PICKLE_NAMES_FILE = "company_names.pkl"

class WhichStocksToBuy(StockAction):

    def __init__(self, top_n: int = 50):
        self.nlp = spacy.load("en_core_web_lg")
        self.top_n = top_n
        self.stock_data = self._load_stock_data()
        self.possible_sectors = list(set(self.stock_data["industry"].dropna()))
        self.index, self.company_vectors, self.company_names = self._load_or_build_faiss_index()
        print(f"[DEBUG] Possible sectors: {len(self.possible_sectors)}")

    def get_description(self) -> str:
        return """
            Queries about whic stocks (generally or within a particular sector) should a user buy or invest in.
            Example queries:
            - What healthcare stocks should I buy?
            - Top 10 tech stocks to invest in
            - Top 100 energy stocks to buy
        """

    def _load_stock_data(self):
        """Loads stock symbols and industries from a pre-fetched CSV file."""
        try:
            df = pd.read_csv("stock_industries.csv")
            print(f"[INFO] Loaded {len(df)} stocks from CSV")
            return df
        except FileNotFoundError:
            print("[ERROR] stock_industries.csv not found! Please run fetch_stock_data.py first.")
            return pd.DataFrame(columns=["symbol", "industry"])

    def _load_or_build_faiss_index(self):
        """Loads the FAISS index if available, otherwise builds a new one."""
        try:
            print("[INFO] Loading FAISS index from disk...")
            index = faiss.read_index("faiss_company_index.bin")

            with open("company_vectors.pkl", "rb") as f:
                company_vectors = pickle.load(f)
            with open("company_names.pkl", "rb") as f:
                company_names = pickle.load(f)

            if index.ntotal == 0 or not company_names:
                raise ValueError("FAISS index is empty. Rebuilding...")

            print("[SUCCESS] FAISS index loaded successfully")
            return index, company_vectors, company_names

        except Exception as e:
            print(f"[ERROR] Failed to load FAISS index: {e}. Rebuilding FAISS index...")
            return self._build_faiss_index()  # Ensure this function always returns values

    def _build_faiss_index(self):
        """Builds the FAISS index from industry embeddings and saves it."""
        print("[INFO] Loading stock data from CSV...")
        try:
            df = pd.read_csv("stock_industries.csv")
        except FileNotFoundError:
            print("[ERROR] stock_industries.csv not found! Run fetch_stock_data_and_build_faiss.py first.")
            return None, None, None  # Prevent returning an invalid index

        print(f"[INFO] Found {len(df)} stocks in CSV.")

        # Load Spacy NLP model for text embeddings
        print("[INFO] Loading NLP model...")
        self.nlp = spacy.load("en_core_web_lg")

        company_vectors = []
        company_names = []

        for _, row in df.iterrows():
            symbol, industry = row["symbol"], row["industry"]

            if industry == "Unknown":
                continue  # Skip if no valid industry information

            industry_vector = self.nlp(industry).vector.astype("float32")

            company_vectors.append(industry_vector)
            company_names.append(symbol)  # Store symbols instead of names

        if not company_vectors:
            print("[ERROR] No valid data for FAISS index. Exiting.")
            return None, None, None  # Prevent returning invalid values

        # Convert to FAISS format
        print("[INFO] Building FAISS index...")
        company_vectors = np.array(company_vectors).astype("float32")
        faiss.normalize_L2(company_vectors)
        index = faiss.IndexFlatIP(company_vectors.shape[1])
        index.add(company_vectors)

        # Save FAISS index & vectors
        faiss.write_index(index, "faiss_company_index.bin")
        with open("company_vectors.pkl", "wb") as f:
            pickle.dump(company_vectors, f)
        with open("company_names.pkl", "wb") as f:
            pickle.dump(company_names, f)

        print("[SUCCESS] FAISS index built and saved successfully.")
        return index, company_vectors, company_names

    def _extract_sector_from_query(self, user_phrase: str) -> str:

        sector_extraction_prompt = SECTOR_EXTRACT_PROMPT.format(
            user_phrase=user_phrase, 
            possible_sectors=self.possible_sectors
        )
        sector = ask_openai(user_content=sector_extraction_prompt).strip().lower()
        return sector

    def execute(self, user_phrase: str, stock_symbol: str) -> StockActionResult:
        """Extracts sector from user query, finds top stocks using FAISS, and fetches performance data from yfinance."""

        sector = self._extract_sector_from_query(user_phrase)

        top_n = extract_top_n_from_query(user_phrase)

        print(f"[DEBUG] SECTOR: {sector}")
        print(f"[DEBUG] TOP N: {top_n}")

        df_stocks = self._load_stock_data()

        if sector.lower() != "none":
            print("[INFO] Searching FAISS index for best matching stocks...")

            matched_symbols = self._find_best_matching_stocks_faiss(sector, top_n=top_n)

            if not matched_symbols or len(matched_symbols) == 0:
                st.markdown(f"⚠️ Hmm, I couldn't find any stocks matching sector {sector}. It might be too generic. Maybe try something more specific, or describe the sector better?")
                return StockActionResult(pd.DataFrame(), "dataframe")

            df_stocks = df_stocks[df_stocks["symbol"].isin(matched_symbols)]
            print(f"[DEBUG] Found {len(df_stocks)} stocks in sector: {sector}")

        stock_performance = []
        progress_bar = st.progress(0)
        analyze_image_placeholder = st.empty()

        status_msg = st.markdown("")
        analyze_image_placeholder.image("analyze.gif", caption="Analyzing markets... ")

        few_mins_markdown = st.markdown("This might take a few minutes... Get yourself some coffee ☕️ and come back. ")
        for i, symbol in enumerate(df_stocks['symbol']):
            if sector.lower() != "none":
                progress_bar.text(f"Idenified {len(df_stocks)} stocks.\nFurther analyzing: {i + 1} out of {len(df_stocks)}...")
            else:
                progress_bar.text(f"Identified {len(df_stocks)} stocks. Analyzing performance: {i + 1} out of {len(df_stocks)}")
                few_mins_markdown.markdown("This might take a few minutes... Get yourself some coffee ☕️ and come back. Or hit Cmd/Ctrl+R and ask something more specific like technology stocks.")
            try:
                if len(df_stocks) > 100:
                    time.sleep(random.uniform(0.1, 0.5))

                try:
                    stock = yf.Ticker(symbol)
                except Exception as e:
                    error_msg = str(e)
                    print(f"[ERROR MESSAGE] {error_msg}")
                    if "Too Many Requests. Rate limited." in error_msg.lower():
                        few_mins_markdown.markdown("⚠️ Hitting rate limits. Waiting a bit...")
                        time.sleep(10)
                    else:
                        st.error(f"[ERROR] Failed to process {symbol}: {e}")
                    continue

                history = stock.history(period='1y')

                if history.empty:
                    print(f"[WARNING] No historical data for {symbol}, skipping...")
                    continue

                # Get company name
                company_name = stock.info.get("longName", symbol)  # Use symbol as fallback

                start_price = history['Close'].iloc[0]
                end_price = history['Close'].iloc[-1]
                percent_change = ((end_price - start_price) / start_price) * 100

                # or if percent_change is nana
                if percent_change < 0 or np.isnan(percent_change):
                    continue
                
                # if (symbol, company_name) in stock_performance:
                    # continue
                
                stock_performance.append((company_name, symbol, percent_change))

            except Exception as e:
                st.error(f"[ERROR] Failed to process {symbol}: {e}")
                continue
        few_mins_markdown.empty()
        # Sort by highest performance
        top_stocks = sorted(stock_performance, key=lambda x: x[2], reverse=True)

        if top_n != -1:
            top_stocks = top_stocks[:top_n]

        status_msg.markdown(f""" 
            <p>Here are {len(top_stocks)} {sector} stocks that have performed well over the past year. 
            NOTE: Past performance is not a guarantee of future results.</p>
        """, unsafe_allow_html=True)

        df_results = pd.DataFrame(top_stocks, columns=['Company Name', 'Ticker', '1-Year Gain (%)'])
        analyze_image_placeholder.empty()
        return StockActionResult(df_results, "dataframe")

    def _load_stock_data(self):
        """Loads stock symbols from the pre-fetched CSV file."""
        try:
            df = pd.read_csv("stock_industries.csv")
            print(f"[INFO] Loaded {len(df)} stocks from CSV")
            return df
        except FileNotFoundError:
            print("[ERROR] stock_industries.csv not found! Please run fetch_stock_data_and_build_faiss.py first.")
            return pd.DataFrame(columns=["symbol", "industry"])

    def _find_best_matching_stocks_faiss(self, sector, top_n=100):
        """Finds the best matching stocks for a given sector using FAISS."""
        if top_n == -1:
            top_n = 50
        try:
            # Load FAISS index
            index = faiss.read_index("faiss_company_index.bin")
            with open("company_vectors.pkl", "rb") as f:
                company_vectors = pickle.load(f)
            with open("company_names.pkl", "rb") as f:
                company_names = pickle.load(f)

            # Convert sector to a vector
            sector_vector = self.nlp(sector).vector.astype("float32").reshape(1, -1)
            faiss.normalize_L2(sector_vector)

            # Search FAISS index
            # choose a random number between 45 and 65
            num_similar = random.randint(45, 65)
            distances, indices = index.search(sector_vector, num_similar)

            # get similarity score 1 standard deviation away from the top score
            # top_score = distances[0][0]
            # std_dev = np.std(distances[0])
            # one_std_away_score = top_score - (2 * std_dev)

            # min_similarity_score = max(one_std_away_score, 0.6)

            filtered_matches = [
                (company_names[i], score) 
                for i, score in zip(indices[0], distances[0]) 
                if i < len(company_names) and score >= 0.75
            ]

            # print the matches and their scores
            for symbol, score in filtered_matches:
                print(f"[MATCH] Symbol: {symbol}, Score: {score}")
            return [symbol for symbol, _ in filtered_matches]
        except Exception as e:
            print(f"[ERROR] Failed to search FAISS index: {e}")
            return []
