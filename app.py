import os
import streamlit as st
from dotenv import load_dotenv

# LangChain core imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


# Get API key and model
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Validate key
if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your .env or Streamlit Secrets.")
    st.stop()

# Streamlit App UI 
st.set_page_config(page_title="LangChain_Chatbot")
st.title("Chatbot")

# Input box
user_question = st.text_input("Ask your question here:")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond clearly to user queries."),
    ("user", "Question: {question}")
])

# Initialize Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(
    model=model,
    google_api_key=api_key
)

output_parser = StrOutputParser()

# Build simple chain for processing
chain = prompt | llm | output_parser

# Generate Response
if user_question:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_question})
    st.success("Response:")
    st.write(response)