import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from schemas import SystemOutput

load_dotenv()

# 1. Secure API Key Resolution
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# 2. Embedding and Vector Database Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. LLM Setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=groq_api_key
)

# 4. Standard JSON Output Parser (Avoids the 404 from with_structured_output)
parser = JsonOutputParser(pydantic_object=SystemOutput)

prompt_template = PromptTemplate(
    template="""You are an expert AI Environmental Scientist at Darukaa.Earth.
Evaluate the user's inquiry regarding ecosystem health, soil parameters, and land restoration.

CRITICAL RULES:
1. Slot Verification: Check if the user has provided sufficient baseline context:
   - Soil metrics (e.g., Soil Organic Carbon %, pH, or moisture)
   - Hydrology / Climate (e.g., rainfall levels, temperature, drought frequency)
   - Land use / vegetation (e.g., monoculture crop, grazing, forest cover)
2. Clarification Loop: If any critical variables are missing or if the input is vague (e.g., "biodiversity is declining" or "plants are dying"), set `needs_clarification = true` and write a targeted `clarification_message` asking specifically for the missing variables.
3. Multi-Metric Synthesis: When context is complete, formulate recommendations that interconnect at least 3 distinct variables (e.g., Soil Organic Carbon + Infiltration/Precipitation + Pollinator/Species Richness). Single-variable answers are strictly prohibited.
4. Scientific Grounding: Ground all biological and geochemical mechanisms in the provided retrieved context (FAO, IPCC). Provide realistic percentage improvements and exact citations.
5. Zero Generic Outputs: Never advise generic practices like "adopt sustainable methods" or "water crops appropriately." Detail exact species, structures, or ecological techniques.

Retrieved Scientific Grounding:
{retrieved_context}

Conversation History:
{chat_history}

Current User Query:
{user_query}

Formatting Instructions:
{format_instructions}

Return ONLY valid JSON.
""",
    input_variables=["retrieved_context", "chat_history", "user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt_template | llm | parser

def process_query(user_query: str, chat_history: str = "") -> SystemOutput:
    retrieved_docs = retriever.invoke(user_query)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

    raw_dict = chain.invoke({
        "retrieved_context": retrieved_context,
        "chat_history": chat_history,
        "user_query": user_query
    })
    
    # Return as validated Pydantic object
    return SystemOutput.model_validate(raw_dict)