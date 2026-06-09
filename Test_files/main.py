import os
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

# Gọi mô hình Gemini 2.5 Flash miễn phí và tối ưu tốc độ
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
) 

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("agent", chatbot)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

main_workflow = workflow.compile()

if __name__ == "__main__":
    # Lấy dữ liệu trực tiếp từ người dùng nhập vào
    user_prompt = input("Say something: ")

    # Đút biến user_prompt vào luồng xử lý của LangGraph
    state = main_workflow.invoke({"messages": [{ "role": "user", "content": user_prompt}]})
    ## print(state["messages"])
    print("\n[Gemini]:", state["messages"][-1].content)

