# agent_jarvis.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from notion_client import Client

load_dotenv()

# ---- LLM Setup ----
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---- Web Search Tool ----
search = DuckDuckGoSearchRun()

class SearchInput(BaseModel):
    query: str

@tool(args_schema=SearchInput)
def web_search(query: str) -> str:
    """Search the internet for current real-time information and news."""
    return search.run(query)

# ---- Notion Setup ----
notion = Client(auth=os.getenv("NOTION_API_KEY"))
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID")

class NotionInput(BaseModel):
    text: str

@tool(args_schema=NotionInput)
def save_to_notion(text: str) -> str:
    """Save information or notes to Notion page."""
    notion.blocks.children.append(
        NOTION_PAGE_ID,
        children=[{
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": text}
                }]
            }
        }]
    )
    return "✅ Saved to Notion successfully!"

# ---- Tools List ---- ✅ defined AFTER both tools
tools = [web_search, save_to_notion]

# ---- Bind tools to LLM ----
llm_with_tools = llm.bind_tools(tools)

# ---- Agent Node ----
def agent_node(state: MessagesState):
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

# ---- Tool Node ----
tool_node = ToolNode(tools)

# ---- Router ----
def should_use_tool(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

# ---- Build Graph ----
graph = StateGraph(MessagesState)
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", should_use_tool)
graph.add_edge("tool_node", "agent_node")
agent = graph.compile()

print("✅ LangGraph 1.0 Agent ready!")

# ---- Chat Loop ----
def chat_loop():
    print("\n🤖 Jarvis (LangGraph 1.0) is ready! Type 'quit' to exit.\n")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "quit":
            print("Jarvis: Goodbye!")
            break
            
        if user_input.strip() == "":
            print("Jarvis: Please say something!\n")
            continue
        
        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": conversation_history})
        response = result["messages"][-1].content
        conversation_history.append(AIMessage(content=response))
        print(f"Jarvis: {response}\n")

if __name__ == "__main__":
    chat_loop()