from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

documents =