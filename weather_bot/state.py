"""
State definition cho Weather Bot StateGraph.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    chat_id: str  # Telegram chat_id, truyền xuyên suốt graph — không lưu env
    retry_count: int        # [VALIDATOR] số lần analyst đã retry, khởi tạo = 0
    validator_feedback: str | None  # [VALIDATOR] feedback từ validator gửi lại cho analyst
