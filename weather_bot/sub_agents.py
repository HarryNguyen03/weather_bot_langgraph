"""
Sub-Agent builders cho Weather Bot.

Mỗi hàm nhận llm + danh sách tools (từ MCP client, đã được bind/wrap ở ngoài)
và trả về create_react_agent đã compile.

Không import tools trực tiếp ở đây — tools đến từ MCP client trong supervisor.py.
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from prompts import (
    WEATHER_ANALYST_PROMPT,
    WEATHER_REPORTER_PROMPT,
)


def build_weather_analyst(llm: ChatOllama, tools: list):
    """
    Weather Analyst: nhận weather tools từ MCP (get_weather, get_5day_forecast).
    Phân tích và trả kết quả — không gửi Telegram.
    """
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=WEATHER_ANALYST_PROMPT,
        name="weather_analyst",
    )


# DEPRECATED: thay bằng LLM workflow trong call_weather_reporter — giữ lại để so sánh before/after
# def build_weather_reporter(llm: ChatOllama, tools: list):
#     """
#     Weather Reporter: nhận telegram tools đã bind chat_id từ supervisor.py
#     (send_telegram_message — schema chỉ có `message`).
#     Soạn và gửi text báo cáo thời tiết qua Telegram.
#     Biểu đồ PNG được gửi trước bởi Python code trong call_weather_reporter — không qua LLM.
#     """
#     return create_react_agent(
#         model=llm,
#         tools=tools,
#         prompt=WEATHER_REPORTER_PROMPT,
#         name="weather_reporter",
#     )


