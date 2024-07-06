from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from prompts.rag_prompt import summarize_chain_prompt

# model = ChatOpenAI(temperature=0, model="gpt-4o")
model = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=False)
summarize_chain = {"element": lambda x: x} | summarize_chain_prompt | model | StrOutputParser()
