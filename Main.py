from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


files = [
    "India_Crop_History_Profitability_Dataset.pdf",
    "India_Soil_Dataset PDF.pdf",
    "India_Soil_Weather_Crop_Dataset.pdf"
]

docs = []  #contains pages from all 3 pdf

for file in files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

# print(len(docs))


# print(docs[15].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="agriculture_rag",
    persist_directory="./chroma_db"
)
