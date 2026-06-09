import os
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

load_dotenv()

# ===========================================================================
# TOOL
# ===========================================================================

_ddg = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """Tìm kiếm thông tin mới nhất trên web."""
    return _ddg.run(query)

tools = [search_web]

# ===========================================================================
# MODEL
# ===========================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
).bind_tools(tools)

# ===========================================================================
# STATE & NODES
# ===========================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = SystemMessage(
    content="Bạn là trợ lý AI có khả năng tìm kiếm web. "
            "Với mọi câu hỏi liên quan đến thông tin thực tế, tin tức, ngày tháng, "
            "hoặc bất kỳ thứ gì có thể thay đổi theo thời gian, hãy luôn dùng tool search_web."
)

def agent(state: State):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: State) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"


# ===========================================================================
# GRAPH
#
# Flow:
#   START → agent → (có tool_calls?) → tools → agent → ...
#                 → (không có)       → END
# ===========================================================================

workflow = StateGraph(State)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

main_workflow = workflow.compile()

# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    user_prompt = input("Say something: ")

    state = main_workflow.invoke({
        "messages": [{"role": "user", "content": user_prompt}]
    })

    content = state["messages"][-1].content
    if isinstance(content, list):
        text = "\n".join(block["text"] for block in content if block.get("type") == "text")
    else:
        text = content
    print("\n[Gemini]:", text)