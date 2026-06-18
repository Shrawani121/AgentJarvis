# rag_jarvis.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()


# ---- LLM Setup ----
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

print("✅ LLM ready!")



# ---- Load PDF ----
print("📄 Loading PDF...")

loader = PyPDFLoader("super-cheatsheet-machine-learning.pdf")
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages!")



# ---- Split into Chunks ----
print("✂️ Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk = 500 characters
    chunk_overlap=50     # 50 characters overlap between chunks
)

chunks = splitter.split_documents(documents)

print(f"✅ Split into {len(chunks)} chunks!")



# ---- Create Embeddings + Store in FAISS ----
print("🔢 Creating embeddings and storing in FAISS...")
print("⏳ First time will take 1-2 minutes, downloading model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

print("✅ Embeddings stored in FAISS!")
  

# ---- Create Retriever ----
print("🔍 Creating retriever...")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # return top 3 most relevant chunks
)

print("✅ Retriever ready!")

parser = StrOutputParser()

# ---- RAG Prompt ----
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know the answer, say "I don't know".

Context: {context}

Question: {question}
""")

# ---- RAG Chain ----
rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

print("✅ RAG chain ready!")

# ---- Chat Loop ----

def chat_loop():
    print("\n🤖 Jarvis (RAG) is ready! Ask anything about the PDF!\n")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ")

        if question.lower() == "quit":
            print("Jarvis: Goodbye!")
            break

        if question.strip() == "":
            print("Jarvis: Please ask something!\n")
            continue

        # runs question through complete RAG pipeline
        response = rag_chain.invoke(question)
        print(f"Jarvis: {response}\n")

        

if __name__ == "__main__":
    chat_loop()