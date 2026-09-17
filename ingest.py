import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load document
loader = TextLoader("knowledge_docs/agroecology_fao_ipcc.txt")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Free, local embedding model (runs completely offline)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in local ChromaDB
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Successfully stored {len(splits)} chunks in ChromaDB!")