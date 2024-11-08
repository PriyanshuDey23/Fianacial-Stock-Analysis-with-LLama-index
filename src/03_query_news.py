import os
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage
from IPython.display import Markdown, display
import openai

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Google API key from the environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if the Google API key is loaded correctly
if not google_api_key:
    raise ValueError("Google API key not found. Please check your .env file.")

# Set up the storage context and load the index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# Load the index from storage
index = load_index_from_storage(storage_context=storage_context)

# Set up the query engine
query_engine = index.as_query_engine()

# Perform a query
response = query_engine.query("What are some near-term risks to Nvidia?")
# print(response)

# Display the response
display(Markdown(f"<b>{response}</b>"))





