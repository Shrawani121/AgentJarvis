# langchain_jarvis.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

# ---- LLM Setup ----
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}."),
    ("human", "{user_input}")
])

# ---- Chain ----
chain = prompt | llm

# ---- Test it ----
response = chain.invoke({
    "name": "Jarvis",
    "user_input": "Hey! Who are you?"
})

print(response.content)