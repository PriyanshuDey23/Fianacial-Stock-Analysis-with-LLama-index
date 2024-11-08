import os
from dotenv import load_dotenv
import openai
from llama_index import GPTVectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import streamlit as st
from llama_index import ServiceContext, LLMPredictor
from llama_index.llms import Gemini
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext


# Load environment variables
load_dotenv()

# Retrieve the Google API key from the environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Set up Gemini model
llm = Gemini(model="models/gemini-ultra")

# Set up open ai model
llm = OpenAI(model_name='gpt-3.5-turbo', max_tokens=6000)

# Load documents from the 'articles' directory
documents = SimpleDirectoryReader('data').load_data()

llm_predictor = LLMPredictor(llm=llm)

# Initialize the embedding model with Hugging Face model
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=800,
    chunk_overlap=20
)


# Convert documents to vector format and create the index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Persist the index to disk
index.storage_context.persist()  # specify directory if needed

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()


st.title('Financial Stock Analysis using LlamaIndex')

st.header("Reports:")

report_type = st.selectbox(
    'What type of report do you want?',
    ('Single Stock Outlook', 'Competitor Analysis'))


if report_type == 'Single Stock Outlook':
    symbol = st.text_input("Stock Symbol")

    if symbol:
        with st.spinner(f'Generating report for {symbol}...'):
            response = query_engine.query(f"Write a report on the outlook for {symbol} stock from the years 2023-2027. Be sure to include potential risks and headwinds.")
            print(type(response))

            st.write(str(response))
            

if report_type == 'Competitor Analysis':
    symbol1 = st.text_input("Stock Symbol 1")
    symbol2 = st.text_input("Stock Symbol 2")

    if symbol1 and symbol2:
        with st.spinner(f'Generating report for {symbol1} vs. {symbol2}...'):
            response = query_engine.query(f"Write a report on the competition between {symbol1} stock and {symbol2} stock.")

            st.write(str(response))




