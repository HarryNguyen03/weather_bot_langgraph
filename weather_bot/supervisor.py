"""
Supervisor — StateGraph orchestrator + MCP client setup.
=========================================================

Kiến trúc:
  run_agent(user_message, chat_id)
      │
      └── _supervisor_graph (compiled một lần lúc startup)
              │
              ├── weather_tools  → build_weather_analyst
              └── MCP tools (chat_id từ InjectedState, không bind)
                      ├── call_weather_reporter (LLM workflow — soạn rồi Python split & gửi)
                      └── send_plain_message (stateless counter guard)

FIX RACE CONDITION:
  chat_id đọc từ state["chat_id"] qua InjectedState trong mỗi tool — safe với concurrent users.
  Không còn closure dict (forecast_cache, feedback_store, rag_cache) — dữ liệu per-request
  chảy qua LangGraph State (forecast_json, rag_docs, validator_feedback).

StateGraph supervisor flow (cyclic — validator reflection loop):
  START → agent → tools → validator → agent (retry) ─┐
                    ↑                                  │
                    └──────────────────────────────────┘
  validator PASS / max retries → agent → reporter → END
"""

import asyncio
import os
import re
import sys
import logging
from typing import Annotated

from pathlib import Path

from dotenv import load_dotenv
from uuid import uuid4
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from state import State
from prompts import SUPERVISOR_PROMPT, VALIDATOR_PROMPT, WEATHER_REPORTER_PROMPT
from analyst import build_analyst_graph
from chart_utils import send_chart_to_telegram
from rag import build_vectorstore, build_retriever_tool

# .env nằm ở Dependencies/ — một cấp trên weather_bot/
load_dotenv(Path(__file__).parent.parent / "Dependencies" / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "gemma4"

def build_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# MCP server config — dùng absolute path để chạy từ bất kỳ thư mục nào
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))

MCP_CONFIG = {
    "weather": {
        "command": sys.executable,
        "args":    [os.path.join(_DIR, "weather_mcp_server.py")],
        "transport": "stdio",
    },
    "telegram": {
        "command": sys.executable,
        "args":    [os.path.join(_DIR, "telegram_mcp_server.py")],
        "transport": "stdio",
    },
}


# ── Global resources — init một lần lúc startup ──
_mcp_client    = None
_all_tools     = None
_vectorstore   = None
_llm           = None
_llm_with_tools = None
_supervisor_graph = None
_analyst_graph = None
_weather_tools = None   # weather MCP tools + retriever tool
_checkpointer  = None   # short-term memory (InMemorySaver, RAM-backed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_forecast_json_block(text: str) -> tuple:
    """
    Tìm và tách block [FORECAST_JSON]...[/FORECAST_JSON] khỏi text.

    Returns:
        (forecast_json_str, cleaned_text)  — nếu block tồn tại
        (None, text)                        — nếu không có block
    """
    pattern = r"\[FORECAST_JSON\](.*?)\[/FORECAST_JSON\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        forecast_json = match.group(1).strip()
        cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return forecast_json, cleaned
    return None, text


def _extract_final_response(result: dict) -> str:
    """Lấy AIMessage cuối cùng không có tool_calls từ kết quả agent."""
    final = next(
        msg for msg in reversed(result["messages"])
        if hasattr(msg, "content")
        and msg.content
        and not getattr(msg, "tool_calls", None)
    )
    return final.content


def _get_tool_call_id(state: State, tool_name: str) -> str:
    """Lấy tool_call_id cho lần gọi hiện tại từ AIMessage cuối cùng trong state."""
    for msg in reversed(state["messages"]):
        if type(msg).__name__ == "AIMessage":
            for tc in getattr(msg, "tool_calls", []):
                if tc["name"] == tool_name:
                    return tc["id"]
    return ""


# ---------------------------------------------------------------------------
# Tools (module level — dùng InjectedState để đọc state, Command để ghi state)
# ---------------------------------------------------------------------------

@tool
async def call_weather_analyst(
    task: str,
    state: Annotated[State, InjectedState],
) -> Command:
    """
    Giao nhiệm vụ cho Weather Analyst Agent để lấy và phân tích thời tiết.
    Dùng tool này khi cần thông tin thời tiết của một địa điểm —
    hiện tại, dự báo 5 ngày, hoặc cả hai.
    task: Task string theo format "[INTENT:current|forecast|both] tên địa điểm".
    """
    print("\n  [Weather Analyst] Đang xử lý...")

    messages = [{"role": "user", "content": task}]

    # [VALIDATOR] Inject feedback từ state nếu đây là retry
    feedback = state.get("validator_feedback")
    if feedback:
        messages.insert(0, {
            "role": "system",
            "content": f"[VALIDATOR FEEDBACK] {feedback}",
        })
        print(f"\n  [Analyst] ↩ Retry với feedback: {feedback[:80]}")

    result = await _analyst_graph.ainvoke({
        "messages":      messages,
        "rewrite_count": 0,
        "docs_relevant": True,
    })

    # Extract forecast_json từ ToolMessage của get_5day_forecast
    forecast_json = None
    for msg in result["messages"]:
        if getattr(msg, "name", None) == "get_5day_forecast" and msg.content:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            if content:
                forecast_json = content
            break

    # Extract rag_docs từ tất cả ToolMessage của retrieve_weather_knowledge
    rag_docs_to_store: list = []
    _no_result_msg = "Không tìm thấy thông tin liên quan trong knowledge base."
    for msg in result["messages"]:
        if (type(msg).__name__ == "ToolMessage"
                and getattr(msg, "name", "") == "retrieve_weather_knowledge"
                and msg.content):
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content or "")
            if content and content != _no_result_msg:
                rag_docs_to_store.append({"page_content": content, "source": "knowledge base"})

    analysis = _extract_final_response(result)

    if forecast_json:
        print(f"\n  [Analyst] ✅ forecast_json cached ({len(forecast_json)} chars)")
    else:
        print("\n  [Analyst] ℹ️ Không có forecast_json (intent=current hoặc tool chưa gọi)")
    if rag_docs_to_store:
        print(f"\n  [Analyst] ✅ rag_docs: {len(rag_docs_to_store)} retrieve result(s) captured")

    tool_call_id = _get_tool_call_id(state, "call_weather_analyst")

    return Command(update={
        "forecast_json": forecast_json,
        "rag_docs":      rag_docs_to_store,
        "messages": [ToolMessage(
            content=analysis,
            tool_call_id=tool_call_id,
            name="call_weather_analyst",
        )],
    })


@tool
async def call_weather_reporter(
    analysis: str,
    state: Annotated[State, InjectedState],
) -> str:
    """
    Giao nhiệm vụ cho Weather Reporter để soạn và gửi báo cáo thời tiết qua Telegram.
    Chỉ dùng sau khi đã có kết quả từ call_weather_analyst.
    chat_id được inject tự động — KHÔNG cần truyền vào args.
    """
    print("\n  [Weather Reporter] Đang soạn và gửi báo cáo thời tiết...")

    chat_id = state["chat_id"]
    cached_forecast_json = state.get("forecast_json")

    # BƯỚC 1 — Gửi chart từ forecast_json trong state
    print(f"\n    [Reporter] forecast_json hit: {cached_forecast_json is not None}")
    if cached_forecast_json:
        print("\n    [Reporter] Đang gửi biểu đồ dự báo (main process)...")
        try:
            loop = asyncio.get_running_loop()
            chart_result = await loop.run_in_executor(
                None, send_chart_to_telegram, cached_forecast_json, chat_id
            )
            print(f"\n    [Reporter] Chart result: {chart_result}")
        except Exception as chart_err:
            import traceback as _tb
            print(f"\n    [Reporter] ⚠️ Lỗi gửi biểu đồ: {chart_err}")
            print(_tb.format_exc())

    # BƯỚC 2 — MỘT lần llm.ainvoke duy nhất: soạn nội dung (không ReAct)
    response = await _llm.ainvoke([
        SystemMessage(content=WEATHER_REPORTER_PROMPT),
        {"role": "user", "content": analysis},
    ])
    raw = response.content or ""
    if isinstance(raw, list):
        raw = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw
        )

    # BƯỚC 3 — Python tự split và gửi qua raw MCP tool (không qua bound wrapper)
    parts = [p.strip() for p in str(raw).split("===MSG===")]
    parts = [p for p in parts if p][:3]     # bỏ rỗng, hard limit 3 tin
    parts = [p[:4096] for p in parts]       # truncate theo giới hạn Telegram
    print(f"\n    [Reporter] Soạn được {len(parts)} tin nhắn từ LLM output")

    raw_send_tool = next(t for t in _all_tools if t.name == "send_telegram_message")

    sent = 0
    for i, part in enumerate(parts, 1):
        ok = False
        for attempt in range(2):             # gửi + retry tối đa 1 lần
            res = await raw_send_tool.ainvoke({"message": part, "chat_id": chat_id})
            if "✅" in str(res):
                ok = True
                break
            print(f"    [Reporter] Lỗi gửi tin {i}: {str(res)[:200]}")
            if attempt == 0:
                print(f"\n    [Reporter] ⚠️ Tin {i} gửi lỗi, retry sau 1s...")
                await asyncio.sleep(1)
        if ok:
            sent += 1
            print(f"\n    [Reporter] ✅ Đã gửi tin {i}/{len(parts)}")
        else:
            print(f"\n    [Reporter] ❌ Tin {i}/{len(parts)} thất bại sau 2 lần thử")

    return f"Đã gửi {sent}/{len(parts)} tin nhắn báo cáo thời tiết."


@tool
async def send_plain_message(
    message: str,
    state: Annotated[State, InjectedState],
) -> str:
    """Gửi câu trả lời hội thoại thông thường qua Telegram. Dùng cho tin nhắn không liên quan thời tiết."""
    chat_id = state["chat_id"]

    # Stateless counter guard: đếm số ToolMessage đã gửi trong request này.
    # Không dùng closure dict — an toàn với concurrent requests.
    call_count = sum(
        1 for msg in state["messages"]
        if type(msg).__name__ == "ToolMessage"
        and getattr(msg, "name", "") == "send_plain_message"
    )
    if call_count >= 3:
        return ("⛔ ĐÃ GỬI ĐỦ SỐ TIN NHẮN. KHÔNG gọi tool này nữa. "
                "Trả về kết luận cuối cùng ngay.")

    raw_tool = next(t for t in _all_tools if t.name == "send_plain_message")
    return await raw_tool.ainvoke({"message": message, "chat_id": chat_id})


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def validate_analyst_output(state: State) -> dict:
    """
    [VALIDATOR] Kiểm tra output của analyst trước khi chuyển sang reporter.

    Tầng 1 — rule-based (không tốn LLM):
      - Response rỗng hoặc quá ngắn
      - Chứa error response từ API
      - Intent mismatch (forecast nhưng thiếu dữ liệu đa ngày)

    Tầng 1.5 — hallucination check (chỉ khi RAG đã inject, đọc rag_docs từ state).

    Tầng 2 — LLM judge.
    """
    # Lấy analyst output từ ToolMessage cuối cùng của call_weather_analyst
    analysis = ""
    for msg in reversed(state["messages"]):
        if type(msg).__name__ == "ToolMessage" and getattr(msg, "name", "") == "call_weather_analyst":
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            analysis = str(content or "")
            break

    # Trích intent từ tool call args trong AIMessage gần nhất
    intent = "current"
    for msg in reversed(state["messages"]):
        if type(msg).__name__ == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "call_weather_analyst":
                    task = tc.get("args", {}).get("task", "")
                    if "[INTENT:forecast]" in task:
                        intent = "forecast"
                    elif "[INTENT:both]" in task:
                        intent = "both"
                    break
            break

    current_retry = state.get("retry_count", 0)

    def _fail(reason: str) -> dict:
        print(f"\n  [VALIDATOR] ❌ {reason} (retry_count → {current_retry + 1})")
        return {"validator_feedback": reason, "retry_count": current_retry + 1}

    def _pass() -> dict:
        print(f"\n  [VALIDATOR] ✅ PASS")
        return {"validator_feedback": None}

    # ── Tầng 1: Rule-based ────────────────────────────────────────────

    if not analysis.strip() or len(analysis.strip()) < 50:
        return _fail("FAIL: Output của analyst rỗng hoặc quá ngắn.")

    if '"error"' in analysis or ('"detail"' in analysis and "lỗi" in analysis.lower()):
        return _fail("FAIL: Output chứa error response từ API thời tiết.")

    if intent in ("forecast", "both"):
        import re as _re
        day_hits = sum(
            1 for kw in [
                "ngày 1", "ngày 2", "ngày 3", "ngày 4", "ngày 5",
                "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu",
                "Thứ Bảy", "Chủ Nhật",
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday", "Sat", "Sun",
            ]
            if kw in analysis
        )
        date_pattern_hits = len(_re.findall(r"\d{2}/\d{2}", analysis))

        if day_hits < 3 and date_pattern_hits < 3:
            return _fail(
                "FAIL: Intent là forecast nhưng output thiếu dữ liệu theo ngày "
                "(cần liệt kê đủ 5 ngày)."
            )

    # ── Tầng 1.5: Hallucination check (chỉ khi RAG đã inject) ──────────
    rag_docs = state.get("rag_docs", [])
    if rag_docs:
        kb_text = "\n\n---\n\n".join([
            f"[Nguồn: {d.get('source', 'knowledge base')}]\n{d['page_content']}"
            for d in rag_docs
        ])
        hallucination_prompt = (
            "Knowledge base:\n"
            f"{kb_text}\n\n"
            "Analyst output:\n"
            f"{analysis[:1500]}\n\n"
            "Chỉ trả về 'hallucinated' khi output MÂU THUẪN trực tiếp với knowledge base, "
            "hoặc gán cho knowledge base nội dung không tồn tại trong đó.\n"
            "Lời khuyên an toàn phổ quát nhất quán với dữ liệu thời tiết thực tế trong output "
            "(VD: trời nóng → uống nước, giảm cường độ, tránh giờ nắng gắt) "
            "phải được coi là grounded kể cả khi knowledge base không đề cập.\n"
            "Trả về: 'grounded' hoặc 'hallucinated: <mô tả phần mâu thuẫn>'"
        )
        try:
            h_resp = await _llm.ainvoke([SystemMessage(content=hallucination_prompt)])
            h_result = (h_resp.content or "").strip().lower()
            print(f"\n  [HALLUCINATION CHECK] → {h_result[:120]}")
            if h_result.startswith("hallucinated"):
                return _fail(f"FAIL: Hallucination detected — {h_result}")
        except Exception as exc:
            logger.warning(f"[HALLUCINATION CHECK] Lỗi: {exc}. Bỏ qua.")

    # ── Tầng 2: LLM judge ────────────────────────────────────────────

    try:
        validator_input = VALIDATOR_PROMPT.format(
            analysis=analysis[:2000], intent=intent
        )
        response = await _llm.ainvoke([SystemMessage(content=validator_input)])
        result = (response.content or "").strip()
        print(f"\n  [VALIDATOR] LLM judge → {result[:120]}")

        if result.startswith("FAIL"):
            return _fail(result)
        return _pass()

    except Exception as exc:
        logger.warning(f"[VALIDATOR] LLM judge lỗi: {exc}. Fallback: PASS.")
        return _pass()


async def agent_node(state: State) -> dict:
    """LLM node — quyết định tool nào cần gọi tiếp theo."""
    msgs = [SystemMessage(content=SUPERVISOR_PROMPT)] + list(state["messages"])

    # [VALIDATOR] Inject feedback hint khi đang trong retry loop
    if state.get("validator_feedback") and state.get("retry_count", 0) <= 2:
        msgs.insert(1, SystemMessage(
            content=(
                f"[VALIDATOR FEEDBACK] {state['validator_feedback']}. "
                f"Hãy gọi lại call_weather_analyst để cải thiện kết quả "
                f"(retry lần {state['retry_count']})."
            )
        ))

    response = await _llm_with_tools.ainvoke(msgs)
    return {"messages": [response]}


async def force_retry_node(state: State) -> dict:
    """Bypass LLM — tạo tool call call_weather_analyst trực tiếp để đảm bảo retry."""
    task = ""
    for msg in reversed(state["messages"]):
        if type(msg).__name__ == "AIMessage":
            for tc in getattr(msg, "tool_calls", []):
                if tc["name"] == "call_weather_analyst":
                    task = tc.get("args", {}).get("task", "")
                    break
        if task:
            break

    print(f"\n  [VALIDATOR] Force retry → call_weather_analyst (task: {task[:80]}...)")
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{
            "name": "call_weather_analyst",
            "args": {"task": task},
            "id":   "retry_" + uuid4().hex[:8],
            "type": "tool_call",
        }],
    )]}


# ---------------------------------------------------------------------------
# Graph edge functions
# ---------------------------------------------------------------------------

def after_tools_check(state: State) -> str:
    """[VALIDATOR] Sau tools node: vào validator nếu analyst vừa chạy, END nếu reporter/plain xong."""
    for msg in reversed(state["messages"]):
        if type(msg).__name__ == "ToolMessage":
            name = getattr(msg, "name", "")
            if name == "call_weather_analyst":
                return "validator"
            if name in {"call_weather_reporter", "send_plain_message"}:
                return "done"
            return "agent"
    return "agent"


def should_retry(state: State) -> str:
    """[VALIDATOR] Trả về 'force_retry' nếu cần retry, 'reporter' nếu đã pass hoặc hết lượt."""
    if state.get("validator_feedback") is not None and state.get("retry_count", 0) <= 2:
        print(f"\n  [VALIDATOR] → Retry analyst (lần {state['retry_count']})")
        return "force_retry"
    print(f"\n  [VALIDATOR] → Tiếp tục reporter")
    return "reporter"


def should_continue(state: State) -> str:
    """Edge condition — có tool_calls thì sang tools node, không thì kết thúc."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Init + startup — compile graph một lần
# ---------------------------------------------------------------------------

async def init_resources():
    """Gọi một lần khi bot khởi động. Build toàn bộ resources + compile graph."""
    global _mcp_client, _all_tools, _vectorstore, _llm, _llm_with_tools, \
           _supervisor_graph, _analyst_graph, _weather_tools, _checkpointer

    _mcp_client = MultiServerMCPClient(MCP_CONFIG)
    _all_tools  = await _mcp_client.get_tools()

    if _vectorstore is None:
        _vectorstore = build_vectorstore()

    # Weather tools = MCP subset + retriever
    weather_base   = [t for t in _all_tools if t.name in {"get_weather", "get_5day_forecast"}]
    retriever_tool = build_retriever_tool(_vectorstore)
    _weather_tools = weather_base + [retriever_tool]

    # LLM
    _llm = build_llm()

    # Analyst graph — compile một lần
    _analyst_graph = build_analyst_graph(_llm, _weather_tools)

    # Supervisor tools + graph — compile một lần
    supervisor_tools = [call_weather_analyst, call_weather_reporter, send_plain_message]
    _llm_with_tools  = _llm.bind_tools(supervisor_tools)

    tool_node = ToolNode(supervisor_tools)

    graph = StateGraph(State)
    graph.add_node("agent",       agent_node)
    graph.add_node("tools",       tool_node)
    graph.add_node("validator",   validate_analyst_output)
    graph.add_node("force_retry", force_retry_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "tools",
        after_tools_check,
        {"validator": "validator", "agent": "agent", "done": END},
    )
    graph.add_conditional_edges(
        "validator",
        should_retry,
        {"force_retry": "force_retry", "reporter": "agent"},
    )
    graph.add_edge("force_retry", "tools")

    _checkpointer = InMemorySaver()
    _supervisor_graph = graph.compile(checkpointer=_checkpointer)
    print("[INIT] MCP tools + vectorstore + analyst graph + supervisor graph sẵn sàng.")


# ---------------------------------------------------------------------------
# Main pipeline — chỉ invoke pre-compiled graph
# ---------------------------------------------------------------------------

async def run_agent(user_message: str, chat_id: str) -> bool:
    """
    Chạy toàn bộ agent pipeline.

    Args:
        user_message: Tin nhắn từ người dùng.
        chat_id:      Telegram chat_id của người dùng.

    Returns:
        True nếu tin nhắn được gửi thành công lên Telegram, False nếu không.
    """
    print(f"\n🚀 Agent | chat_id={chat_id} | message={user_message[:80]}")
    print("=" * 60)
    print("\n[Supervisor] Bắt đầu điều phối...")

    result = await _supervisor_graph.ainvoke(
        {
            "messages":           [{"role": "user", "content": user_message}],
            "chat_id":            chat_id,
            "retry_count":        0,
            "validator_feedback": None,
            "forecast_json":      None,
            "rag_docs":           [],
        },
        config={
            "configurable": {"thread_id": chat_id},
            "run_name": f"weather-bot | {user_message[:40]}",
            "tags":     ["weather-bot"],
            "metadata": {"chat_id": chat_id},
        },
    )

    # ── Log luồng xử lý ───────────────────────────────────────────────────

    print("\n📋 LUỒNG XỬ LÝ:")
    print("-" * 60)
    for msg in result["messages"]:
        msg_type   = type(msg).__name__
        name       = getattr(msg, "name",       None)
        content    = getattr(msg, "content",    "")
        tool_calls = getattr(msg, "tool_calls", [])

        if msg_type == "HumanMessage":
            print(f"\n👤 [User]: {content[:200]}")
        elif msg_type == "AIMessage" and tool_calls:
            for tc in tool_calls:
                print(f"\n🤖 [Supervisor] → {tc['name']}")
                print(f"   {str(tc.get('args', {}))[:150]}...")
        elif msg_type == "ToolMessage":
            preview = str(content)[:2000]
            print(f"\n🔧 [{name}]: {preview}{'...' if len(str(content)) > 300 else ''}")
        elif msg_type == "AIMessage" and content and not tool_calls:
            print(f"\n✅ [Supervisor] kết luận: {content}")

    print("\n" + "=" * 60)
    print("🎉 Pipeline hoàn tất!\n")

    # ── Delivery detection ────────────────────────────────────────────────
    # call_weather_reporter: trả "Đã gửi {sent}/{tổng} ..." → delivered khi sent > 0
    # send_plain_message: MCP tool trả về "✅ ..." trực tiếp → kiểm tra "✅"
    def _reporter_delivered(content: str) -> bool:
        if "Đã gửi" not in content:
            return False
        m = re.search(r"Đã gửi\s+(\d+)\s*/", content)
        return bool(m) and int(m.group(1)) > 0

    delivered = any(
        (
            getattr(msg, "name", "") == "call_weather_reporter"
            and _reporter_delivered(str(getattr(msg, "content", "")))
        ) or (
            getattr(msg, "name", "") == "send_plain_message"
            and "✅" in str(getattr(msg, "content", ""))
        )
        for msg in result["messages"]
        if type(msg).__name__ == "ToolMessage"
    )

    # ── Safety-net ────────────────────────────────────────────────────────
    # Nếu supervisor kết thúc với AIMessage content thường mà KHÔNG gọi tool
    # (quên gửi), nội dung đó không bao giờ tới Telegram. Tự gửi qua MCP.
    if not delivered:
        last = result["messages"][-1] if result["messages"] else None
        if (last is not None
                and type(last).__name__ == "AIMessage"
                and not getattr(last, "tool_calls", None)):
            content = getattr(last, "content", "") or ""
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content).strip()
            if content:
                try:
                    raw_send_tool = next(
                        t for t in _all_tools if t.name == "send_telegram_message"
                    )
                    res = await raw_send_tool.ainvoke(
                        {"message": content, "chat_id": chat_id}
                    )
                    if "✅" in str(res):
                        delivered = True
                        print("\n[Safety-net] Đã gửi content của supervisor trực tiếp")
                except Exception as exc:
                    logger.warning(f"[Safety-net] Gửi trực tiếp lỗi: {exc}")

    return delivered
