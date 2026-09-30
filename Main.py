import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi

# LOAD ENVIRONMENT VARIABLES


load_dotenv()



#LOAD PDF DOCUMENTS

files = ['India_District_Agri_Master_RAG.pdf'
]

docs = []

for file in files:
    loader = PyPDFLoader(file)
    file_docs = loader.load()

    # Store source filename in metadata
    for doc in file_docs:
        doc.metadata["source_file"] = file

    docs.extend(file_docs)

print(f"Total pages loaded: {len(docs)}")


#SPLIT DOCUMENTS INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")



#CREATE EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



#STORE CHUNKS IN CHROMADB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="agriculture_rag",
    persist_directory="./chroma_db"
)





# CREATE BM25 INDEX
def tokenize(text):
    """
    Convert text into lowercase words.
    Handles punctuation better than simple .split()
    """
    return re.findall(r"\b\w+\b", text.lower())


tokenized_chunks = [
    tokenize(doc.page_content)
    for doc in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


#USER QUERY
query = "What is the soil type of Sehore?"

print("QUERY:", query)

#BM25 RETRIEVAL
tokenized_query = tokenize(query)

scores = bm25.get_scores(tokenized_query)

# Get indices of top 5 documents
ranked_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:2]



# DISPLAY BM25 RESULTS
print("\nBM25 RESULTS")

for rank, index in enumerate(ranked_indices, start=1):

    doc = chunks[index]

    print(f"\n  Result {rank}  ")
    print("BM25 Score:", scores[index])
    print("Source:", doc.metadata.get("source_file"))
    print("Page:", doc.metadata.get("page"))
    print("Content:")
    print(doc.page_content[:700])