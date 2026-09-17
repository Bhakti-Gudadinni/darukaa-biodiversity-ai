import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from schemas import SystemOutput

load_dotenv()

# Choose the embedding function you used in ingest.py
# If you used HuggingFace:
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except ImportError:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Using gpt-4o-mini for fast, cost-effective structured output
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm = llm.with_structured_output(SystemOutput)

def process_query(user_query: str, chat_history: str = "") -> SystemOutput:
    # 1. Retrieve knowledge from ChromaDB
    retrieved_docs = retriever.invoke(user_query)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

    # 2. Reasoning Prompt enforcing Hackathon Rubric
    prompt = f"""
    You are an expert AI Environmental Scientist at Darukaa.Earth.
    Evaluate the user's inquiry regarding ecosystem health and land restoration.

    CRITICAL RULES:
    1. Check for sufficient input variables: Does the user provide soil health (SOC % or pH), water/rainfall, and current land use?
    2. If ANY critical variables are absent or the input is vague (e.g., 'biodiversity is dropping' or 'restore my land'), set `needs_clarification = True` and output a polite, targeted `clarification_message` asking for those missing parameters.
    3. If sufficient variables exist, connect at least 3 distinct variables (e.g., Soil Organic Carbon + Infiltration/Precipitation + Pollinator/Biodiversity metrics).
    4. Base every recommendation on real ecological mechanisms. Quantified claims must match the scientific literature (FAO, IPCC) in the context.
    5. NEVER provide generic recommendations like 'use sustainable methods' or 'water regularly'.

    Retrieved Scientific Grounding:
    {retrieved_context}

    Conversation History:
    {chat_history}

    Current User Query:
    {user_query}
    """

    return structured_llm.invoke(prompt)