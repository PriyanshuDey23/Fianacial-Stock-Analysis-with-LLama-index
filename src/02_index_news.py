from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import Gemini
import os

# Load environment variables from .env file
load_dotenv()

# Set up Gemini model
llm = Gemini(model="models/gemini-ultra")

# Load documents from the 'articles' directory
documents = SimpleDirectoryReader('articles').load_data()

# Initialize the embedding model with Hugging Face model
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

# Set up service context with desired chunk size and overlap
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
