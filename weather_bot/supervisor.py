"""
Supervisor — StateGraph orchestrator + MCP client setup.
=========================================================

Kiến trúc:
  run_agent(user_message, chat_id)
      │
      └── MultiServerMCPClient (async context manager)
              │  lấy tools từ 2 MCP servers
              │
              ├── weather_tools  → build_weather_analyst
              └── telegram_tools (chat_id pre-injected via _bind_chat_id)
                      ├── build_weather_reporter
                      └── send_plain_message (dùng trực tiếp bởi Supervisor)

FIX RACE CONDITION:
  Mỗi lần gọi run_agent, chat_id được capture trong closure riêng.
  Không còn os.environ["_CURRENT_CHAT_ID"] dùng chung — safe với concurrent users.

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
from typing import TYPE_CHECKING

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import State
from prompts import SUPERVISOR_PROMPT, VALIDATOR_PROMPT  # [VALIDATOR]
from sub_agents import build_weather_analyst, build_weather_reporter
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
        "command": sys.executable,   # dùng đúng Python env đang chạy
        "args":    [os.path.join(_DIR, "weather_mcp_server.py")],
        "transport": "stdio",
    },
    "telegram": {
        "command": sys.executable,   # dùng đúng Python env đang chạy
        "args":    [os.path.join(_DIR, "telegram_mcp_server.py")],
        "transport": "stdio",
    },
}

_vectorstore = None  # global cache — build một lần, tái dùng mọi request


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


def _bind_chat_id(mcp_tool, chat_id: str) -> StructuredTool:
    """
    Interceptor: tạo bản sao của telegram MCP tool với chat_id được pre-inject.

    Schema trả về cho LLM chỉ có `message` — agent KHÔNG cần tự điền chat_id.
    chat_id được capture trong closure, an toàn với concurrent requests.
    """
    async def _call(message: str) -> str:
        return await mcp_tool.ainvoke({"message": message, "chat_id": chat_id})

    return StructuredTool.from_function(
        coroutine=_call,
        name=mcp_tool.name,
        description=mcp_tool.description,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_agent(user_message: str, chat_id: str) -> bool:
    """
    Chạy toàn bộ agent pipeline.

    Args:
        user_message: Tin nhắn từ người dùng.
        chat_id:      Telegram chat_id của người dùng — truyền xuyên suốt pipeline,
                      không lưu vào os.environ.

    Returns:
        True nếu tin nhắn được gửi thành công lên Telegram, False nếu không.
    """
    mcp_client = MultiServerMCPClient(MCP_CONFIG)
    all_tools = await mcp_client.get_tools()

    # Phân tách tools theo server
    weather_tools  = [t for t in all_tools if t.name in {"get_weather", "get_5day_forecast"}]
    raw_tg_tools   = [t for t in all_tools if t.name in {"send_telegram_message", "send_plain_message"}]

    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    retriever_tool = build_retriever_tool(_vectorstore)
    weather_tools.append(retriever_tool)

    # Bind chat_id vào telegram tools — schema hiển thị với LLM chỉ còn `message`
    telegram_tools = [_bind_chat_id(t, chat_id) for t in raw_tg_tools]

    llm = build_llm()

    # ── Handoff tools (closures — capture MCP tools + llm) ───────────────
    # Định nghĩa bên trong run_agent để mỗi request có scope độc lập.
    # Dùng async để compatible với ToolNode.ainvoke và ainvoke của sub-agents.

    from langchain_core.tools import tool

    # Shared cache trong scope của mỗi request — tránh truyền JSON qua LLM
    # (LLM hay tự cắt block [FORECAST_JSON] khi truyền vào call_weather_reporter)
    forecast_cache: dict = {"forecast_json": None}
    feedback_store: dict = {"feedback": None}  # [VALIDATOR] validator → call_weather_analyst
    rag_cache: dict = {"docs": []}             # [RAG] docs đã inject → hallucination check

    @tool
    async def call_weather_analyst(task: str) -> str:
        """
        Giao nhiệm vụ cho Weather Analyst Agent để lấy và phân tích thời tiết.
        Dùng tool này khi cần thông tin thời tiết của một địa điểm —
        hiện tại, dự báo 5 ngày, hoặc cả hai.
        task: Task string theo format "[INTENT:current|forecast|both] tên địa điểm".
        """
        print("\n  [Weather Analyst] Đang xử lý...")

        # [VALIDATOR] Inject feedback vào đầu messages nếu đây là retry
        messages = [{"role": "user", "content": task}]

        # Force RAG: detect keyword → retrieve → inject vào system message
        RETRIEVAL_KEYWORDS = [
            "chạy bộ", "tập thể dục", "tập gym", "yoga", "picnic",
            "du lịch", "nên mặc", "mặc gì", "trang phục", "có nên",
            "sức khỏe", "hoạt động", "đi bộ", "đạp xe", "bơi lội",
            "khí hậu", "đặc điểm", "thời điểm nào", "mùa nào"
        ]

        if any(kw in task.lower() for kw in RETRIEVAL_KEYWORDS):
            retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(task)
            if docs:
                # ── Tầng A: Retrieval Grader — chỉ inject doc relevant ──────
                relevant_docs = []
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "knowledge base")
                    preview = doc.page_content[:300].replace("\n", " ")
                    print(f"\n  [RAG] Chunk {i} [{src}]: {preview}{'...' if len(doc.page_content) > 300 else ''}")
                    grade_prompt = (
                        f"Query: {task}\n\n"
                        f"Document:\n{doc.page_content[:400]}\n\n"
                        "Tài liệu này có chứa thông tin hữu ích để trả lời query không?\n"
                        "Trả về đúng một từ: yes hoặc no"
                    )
                    grade_resp = await llm.ainvoke([SystemMessage(content=grade_prompt)])
                    grade = (grade_resp.content or "").strip().lower()
                    if grade.startswith("yes"):
                        relevant_docs.append(doc)
                        print(f"  [RAG Grader] ✅ relevant — [{src}]")
                    else:
                        print(f"  [RAG Grader] ❌ not relevant — [{src}]")

                if relevant_docs:
                    rag_cache["docs"] = relevant_docs
                    context = "\n\n---\n\n".join([
                        f"[Nguồn: {doc.metadata.get('source', 'knowledge base')}]\n{doc.page_content}"
                        for doc in relevant_docs
                    ])
                    messages.insert(0, {
                        "role": "system",
                        "content": (
                            "[KNOWLEDGE BASE] Thông tin tham khảo từ knowledge base — "
                            "BẮT BUỘC dùng thông tin này để trả lời, không được tự bịa:\n\n"
                            f"{context}"
                        )
                    })
                    print(f"\n  [RAG] ✅ Injected {len(relevant_docs)}/{len(docs)} relevant chunks vào analyst context")
                else:
                    rag_cache["docs"] = []
                    print(f"\n  [RAG] ⚠️ 0/{len(docs)} docs passed grading — không inject")

        if feedback_store.get("feedback"):
            messages.insert(0, {
                "role": "system",
                "content": f"[VALIDATOR FEEDBACK] {feedback_store['feedback']}",
            })
            print(f"\n  [Analyst] ↩ Retry với feedback: {feedback_store['feedback'][:80]}")

        agent  = build_weather_analyst(llm, weather_tools)
        result = await agent.ainvoke({"messages": messages})

        # Python tự extract raw forecast JSON từ tool message history
        # Không nhờ LLM làm việc này
        forecast_json = None
        for msg in result["messages"]:
            if getattr(msg, "name", None) == "get_5day_forecast" and msg.content:
                content = msg.content
                # LangChain mới có thể trả content dạng list[dict] thay vì str
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                if content:
                    forecast_json = content
                break

        analysis = _extract_final_response(result)

        # Lưu vào cache — KHÔNG truyền qua LLM (LLM hay cắt block JSON này)
        if forecast_json:
            forecast_cache["forecast_json"] = forecast_json
            print(f"\n  [Analyst] ✅ forecast_json cached ({len(forecast_json)} chars)")
        else:
            print("\n  [Analyst] ℹ️ Không có forecast_json (intent=current hoặc tool chưa gọi)")

        return analysis

    @tool
    async def call_weather_reporter(analysis: str) -> str:
        """
        Giao nhiệm vụ cho Weather Reporter Agent để soạn và gửi báo cáo thời tiết qua Telegram.
        Chỉ dùng sau khi đã có kết quả từ call_weather_analyst.
        chat_id được inject tự động — KHÔNG cần truyền vào args.
        """
        print("\n  [Weather Reporter] Đang soạn và gửi báo cáo thời tiết...")

        # BƯỚC 1 — Đọc forecast_json từ cache (không phụ thuộc LLM truyền qua)
        cached_forecast_json = forecast_cache.get("forecast_json")
        print(f"\n    [Reporter] forecast_cache hit: {cached_forecast_json is not None}")

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

        # BƯỚC 2 — Reporter chỉ format và gửi text; chat_id đã bind trong telegram_tools
        agent  = build_weather_reporter(llm, telegram_tools)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": analysis}]})
        return _extract_final_response(result)

    # [VALIDATOR] ── Validator node ──────────────────────────────────────────

    async def validate_analyst_output(state: State) -> dict:
        """
        [VALIDATOR] Kiểm tra output của analyst trước khi chuyển sang reporter.

        Tầng 1 — rule-based (không tốn LLM):
          - Response rỗng hoặc quá ngắn
          - Chứa error response từ API
          - Intent mismatch (forecast nhưng thiếu dữ liệu đa ngày)

        Tầng 2 — LLM judge (chạy nếu tầng 1 pass):
          - Dùng VALIDATOR_PROMPT với llm.ainvoke
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
            feedback_store["feedback"] = reason
            print(f"\n  [VALIDATOR] ❌ {reason} (retry_count → {current_retry + 1})")
            return {"validator_feedback": reason, "retry_count": current_retry + 1}

        def _pass() -> dict:
            feedback_store["feedback"] = None
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
                    # format ngày tiếng Việt
                    "ngày 1", "ngày 2", "ngày 3", "ngày 4", "ngày 5",
                    # thứ tiếng Việt
                    "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu",
                    "Thứ Bảy", "Chủ Nhật",
                    # tiếng Anh
                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                    "Saturday", "Sunday", "Sat", "Sun",
                ]
                if kw in analysis
            )
            # đếm thêm pattern dd/mm hoặc dd/mm/yyyy trong analysis
            date_pattern_hits = len(_re.findall(r"\d{2}/\d{2}", analysis))

            if day_hits < 3 and date_pattern_hits < 3:
                return _fail(
                    "FAIL: Intent là forecast nhưng output thiếu dữ liệu theo ngày "
                    "(cần liệt kê đủ 5 ngày)."
                )

        # ── Tầng 1.5: Hallucination check (chỉ khi RAG đã inject) ──────────
        if rag_cache.get("docs"):
            kb_text = "\n\n---\n\n".join([
                f"[Nguồn: {doc.metadata.get('source', 'knowledge base')}]\n{doc.page_content}"
                for doc in rag_cache["docs"]
            ])
            hallucination_prompt = (
                "Knowledge base:\n"
                f"{kb_text}\n\n"
                "Analyst output:\n"
                f"{analysis[:1500]}\n\n"
                "Kiểm tra: Các lời khuyên về sức khỏe, trang phục, hoặc hoạt động trong output "
                "có được hỗ trợ bởi knowledge base không, hay analyst tự bịa?\n"
                "Trả về: 'grounded' hoặc 'hallucinated: <mô tả phần bịa>'"
            )
            try:
                h_resp = await llm.ainvoke([SystemMessage(content=hallucination_prompt)])
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
            response = await llm.ainvoke([SystemMessage(content=validator_input)])
            result = (response.content or "").strip()
            print(f"\n  [VALIDATOR] LLM judge → {result[:120]}")

            if result.startswith("FAIL"):
                return _fail(result)
            return _pass()

        except Exception as exc:
            logger.warning(f"[VALIDATOR] LLM judge lỗi: {exc}. Fallback: PASS.")
            return _pass()

    # [VALIDATOR] ── Conditional edge functions ───────────────────────────

    def after_tools_check(state: State) -> str:
        """[VALIDATOR] Sau tools node: vào validator nếu analyst vừa chạy, END nếu reporter/plain xong."""
        for msg in reversed(state["messages"]):
            if type(msg).__name__ == "ToolMessage":
                name = getattr(msg, "name", "")
                if name == "call_weather_analyst":
                    return "validator"
                if name in {"call_weather_reporter", "send_plain_message"}:
                    return "done"   # graph kết thúc — không để supervisor gọi lại
                return "agent"
        return "agent"

    def should_retry(state: State) -> str:
        """[VALIDATOR] Trả về 'analyst' nếu cần retry, 'reporter' nếu đã pass hoặc hết lượt."""
        if state.get("validator_feedback") is not None and state.get("retry_count", 0) <= 2:
            print(f"\n  [VALIDATOR] → Retry analyst (lần {state['retry_count']})")
            return "analyst"
        print(f"\n  [VALIDATOR] → Tiếp tục reporter")
        return "reporter"

    # Expose send_plain_message directly to the supervisor so it can reply to
    # plain (non-weather) messages without spawning an extra sub-agent.
    bound_send_plain = next(t for t in telegram_tools if t.name == "send_plain_message")

    supervisor_tools = [call_weather_analyst, call_weather_reporter, bound_send_plain]
    llm_with_tools   = llm.bind_tools(supervisor_tools)

    # ── StateGraph ────────────────────────────────────────────────────────

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

        response = await llm_with_tools.ainvoke(msgs)
        return {"messages": [response]}

    def should_continue(state: State) -> str:
        """Edge condition — có tool_calls thì sang tools node, không thì kết thúc."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    tool_node = ToolNode(supervisor_tools)

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("validator", validate_analyst_output)  # [VALIDATOR]
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    # [VALIDATOR] Thay direct edge tools→agent bằng conditional qua validator
    graph.add_conditional_edges(
        "tools",
        after_tools_check,
        {"validator": "validator", "agent": "agent", "done": END},
    )
    # [VALIDATOR] Cyclic edge: validator → analyst (retry) hoặc reporter (tiếp tục)
    graph.add_conditional_edges(
        "validator",
        should_retry,
        {"analyst": "agent", "reporter": "agent"},
    )

    supervisor = graph.compile()

    # ── Run ───────────────────────────────────────────────────────────────

    print(f"\n🚀 Agent | chat_id={chat_id} | message={user_message[:80]}")
    print("=" * 60)
    print("\n[Supervisor] Bắt đầu điều phối...")

    result = await supervisor.ainvoke({
        "messages":          [{"role": "user", "content": user_message}],
        "chat_id":           chat_id,
        "retry_count":       0,     # [VALIDATOR]
        "validator_feedback": None,  # [VALIDATOR]
    })

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
    # call_weather_reporter: return value là text từ sub-agent, không chứa "✅"
    #   → coi là delivered nếu ToolMessage tồn tại và không rỗng (tool đã chạy xong)
    # send_plain_message: MCP tool trả về "✅ ..." trực tiếp → kiểm tra "✅"
    delivered = any(
        (
            getattr(msg, "name", "") == "call_weather_reporter"
            and bool(str(getattr(msg, "content", "")).strip())
        ) or (
            getattr(msg, "name", "") == "send_plain_message"
            and "✅" in str(getattr(msg, "content", ""))
        )
        for msg in result["messages"]
        if type(msg).__name__ == "ToolMessage"
    )
    return delivered
