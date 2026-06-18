# langchain_jarvis.py

import os
from urllib import response
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

# ---- LLM Setup ----
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


# ---- Output Parser ----
parser = StrOutputParser()

# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# ---- Chain ----
chain = prompt | llm | parser

# ---- Memory (our history list) ----
history = []


# ---- Chat Loop ----
def chat_loop():
    print("🤖 Jarvis (LangChain) is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Jarvis: Goodbye!")
            break

        if user_input.strip() == "":
            print("Jarvis: Please say something!\n")
            continue

        response = chain.invoke({
            "name": "Jarvis",
            "history": history, 
            "user_input": user_input
        })

         # save to memory AFTER getting response
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

        print(f"Jarvis: {response}\n")

if __name__ == "__main__":
    chat_loop()