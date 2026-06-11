"""
Weather Analyst sub-graph — Self-RAG flow.
==========================================

Flow:
  START → agent → tools → grade_documents → rewrite_question → agent (retry)
                       ↘ agent (nếu tool không phải retrieve)
  grade: relevant HOẶC hết quota rewrite → agent
  agent không có tool_calls → END
"""

import re
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prompts import WEATHER_ANALYST_PROMPT, GRADE_DOCUMENTS_PROMPT, REWRITE_QUESTION_PROMPT


class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    rewrite_count: int
    docs_relevant: bool


def build_analyst_graph(llm, weather_tools: list):
    """Build và compile Weather Analyst StateGraph."""
    llm_with_tools = llm.bind_tools(weather_tools)
    tool_node = ToolNode(weather_tools)

    # ── Helpers nội bộ ────────────────────────────────────────────────────

    def _get_question(state: AnalystState) -> str:
        for msg in state["messages"]:
            if type(msg).__name__ == "HumanMessage":
                c = msg.content if isinstance(msg.content, str) else str(msg.content)
                m = re.search(r"Câu hỏi đầy đủ của user:\s*(.+)", c, re.DOTALL)
                return m.group(1).strip() if m else c
        return ""

    def _get_last_retrieve_content(state: AnalystState) -> str:
        for msg in reversed(state["messages"]):
            if (type(msg).__name__ == "ToolMessage"
                    and getattr(msg, "name", "") == "retrieve_weather_knowledge"):
                content = msg.content
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                return str(content or "")
        return ""

    # ── Nodes ─────────────────────────────────────────────────────────────

    async def agent(state: AnalystState) -> dict:
        msgs = [SystemMessage(content=WEATHER_ANALYST_PROMPT)] + list(state["messages"])
        response = await llm_with_tools.ainvoke(msgs)
        for tc in getattr(response, "tool_calls", []):
            if tc["name"] == "retrieve_weather_knowledge":
                print(f"  [Analyst RAG] retrieve → query: {tc['args'].get('query', '')[:120]}")
        return {"messages": [response]}

    async def grade_documents(state: AnalystState) -> dict:
        question = _get_question(state)
        context  = _get_last_retrieve_content(state)

        print(f"  [Analyst RAG] grade — question: {question[:100]}")
        response = await llm.ainvoke([SystemMessage(
            content=GRADE_DOCUMENTS_PROMPT.format(
                question=question,
                context=context[:3500],
            )
        )])
        result = (response.content or "").strip()
        print(f"  [Analyst RAG] grade result → {result[:80]}")
        head = result.lower()[:20]
        if "yes" in head:
            docs_relevant = True
        elif "no" in head:
            docs_relevant = False
        else:
            docs_relevant = True
            print("  [Analyst RAG] grade unparseable → default relevant")
        return {"docs_relevant": docs_relevant}

    async def rewrite_question(state: AnalystState) -> dict:
        question = _get_question(state)
        context  = _get_last_retrieve_content(state)[:800]

        response = await llm.ainvoke([SystemMessage(
            content=REWRITE_QUESTION_PROMPT.format(
                question=question,
                context=context,
            )
        )])
        new_query = (response.content or "").strip()
        print(f"  [Analyst RAG] rewrite → new query: {new_query[:100]}")

        return {
            "messages": [SystemMessage(
                content=(
                    f"Tài liệu truy xuất chưa phù hợp. "
                    f"Gọi lại retrieve_weather_knowledge với query: {new_query}. "
                    f"Sau đó tiếp tục nhiệm vụ."
                )
            )],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    # ── Edge conditions ───────────────────────────────────────────────────

    def route_after_agent(state: AnalystState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def route_after_tools(state: AnalystState) -> str:
        has_retrieve = False
        for msg in reversed(state["messages"]):
            if type(msg).__name__ == "AIMessage":
                break
            if type(msg).__name__ == "ToolMessage":
                if getattr(msg, "name", "") == "retrieve_weather_knowledge":
                    has_retrieve = True
                    content = msg.content
                    if isinstance(content, list):
                        content = "".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    print(f"  [Analyst RAG] retrieve result (200c): {str(content)[:200]}")
                    sources = re.findall(r"\[Nguồn:\s*([^\]]+)\]", str(content))
                    if sources:
                        print(f"  [Analyst RAG] sources: {', '.join(sources)}")
        return "grade_documents" if has_retrieve else "agent"

    def route_after_grade(state: AnalystState) -> str:
        if state.get("docs_relevant", True) or state.get("rewrite_count", 0) >= 1:
            return "agent"
        return "rewrite_question"

    # ── Graph ─────────────────────────────────────────────────────────────

    graph = StateGraph(AnalystState)
    graph.add_node("agent",            agent)
    graph.add_node("tools",            tool_node)
    graph.add_node("grade_documents",  grade_documents)
    graph.add_node("rewrite_question", rewrite_question)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "tools", route_after_tools,
        {"grade_documents": "grade_documents", "agent": "agent"},
    )
    graph.add_conditional_edges(
        "grade_documents", route_after_grade,
        {"agent": "agent", "rewrite_question": "rewrite_question"},
    )
    graph.add_edge("rewrite_question", "agent")

    return graph.compile()
