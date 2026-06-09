"""
Weather Agentic System - Multi-Agent với LangGraph (Tool-Calling Pattern)
=========================================================================

Kiến trúc: Supervisor (create_react_agent) điều phối các sub-agent
được wrap thành @tool — không dùng thư viện langgraph-supervisor.

Flow (hoàn toàn trên Telegram):
  User nhắn tin trên Telegram
      │
  [Telegram Bot] nhận tin → trigger agent
      │
  [Supervisor Agent]  ← create_react_agent + Ollama (gemma4)
      │   (phân tích intent: current / forecast / both / chat)
      │
      ├─► call_weather_analyst(task)     → [Weather Analyst Agent]
      │       ├─► get_weather             (intent: current, both)
      │       └─► get_5day_forecast       (intent: forecast, both)
      │
      ├─► call_weather_reporter(analysis) → [Weather Reporter Agent]
      │       └─► send_telegram_message   (format theo intent)
      │
      └─► call_chat_agent(message)       → [Chat Agent]
              └─► send_plain_message

Model: Ollama chạy local — không giới hạn quota, không cần API key.
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import warnings
from langgraph.prebuilt import create_react_agent
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

load_dotenv()

# Tracks whether a send tool actually delivered a message in the current run.
# Reset to False at the start of each run_agent call.
_message_delivered = False

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.WARNING,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: MODEL SETUP (Ollama local)
# ===========================================================================

OLLAMA_MODEL = "gemma4"


def build_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


# ===========================================================================
# SECTION 2: TOOLS CỦA CÁC SUB-AGENT
# ===========================================================================

@tool
def get_weather(location: str) -> str:
    """
    Lấy dữ liệu thời tiết thực tế từ OpenWeatherMap API.

    Args:
        location: Tên thành phố cần tra cứu (VD: 'Hanoi', 'Ho Chi Minh City').

    Returns:
        JSON string chứa thông tin thời tiết chi tiết.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "OPENWEATHER_API_KEY chưa được cấu hình."})

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q":     location,
        "appid": api_key,
        "units": "metric",
        "lang":  "vi",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        offset   = timedelta(seconds=data["timezone"])
        local_dt = datetime.fromtimestamp(data["dt"], tz=timezone.utc).astimezone(timezone(offset))

        result = {
            "location":         f"{data['name']}, {data['sys']['country']}",
            "local_datetime":   local_dt.strftime("%H:%M %d/%m/%Y"),
            "description":      data["weather"][0]["description"],
            "temperature_c":    data["main"]["temp"],
            "feels_like_c":     data["main"]["feels_like"],
            "humidity_percent": data["main"]["humidity"],
            "wind_speed_mps":   data["wind"]["speed"],
            "visibility_m":     data.get("visibility", "N/A"),
        }
        return json.dumps(result, ensure_ascii=False)

    except requests.exceptions.HTTPError:
        if resp.status_code == 404:
            return json.dumps({"error": f"Không tìm thấy '{location}'. Thử tên tiếng Anh."})
        return json.dumps({"error": f"HTTP error: {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_5day_forecast(location: str) -> str:
    """
    Lấy dự báo thời tiết 5 ngày từ OpenWeatherMap API.

    Args:
        location: Tên thành phố cần tra cứu (VD: 'Hanoi', 'Ho Chi Minh City').

    Returns:
        JSON string chứa dự báo thời tiết 5 ngày, mỗi ngày gồm
        temp_min, temp_max, description, humidity_avg, icon.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "OPENWEATHER_API_KEY chưa được cấu hình."})

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q":     location,
        "appid": api_key,
        "units": "metric",
        "lang":  "vi",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Group 40 slots × 3h theo ngày
        days: dict = {}
        for slot in data["list"]:
            date_str = slot["dt_txt"].split(" ")[0]  # "2025-05-21"
            if date_str not in days:
                days[date_str] = []
            days[date_str].append(slot)

        forecast = []
        for date_str, slots in list(days.items())[:5]:
            # Slot đại diện: ưu tiên 12:00:00, fallback về slot đầu tiên
            rep_slot = next(
                (s for s in slots if "12:00:00" in s["dt_txt"]),
                slots[0],
            )
            temp_min    = min(s["main"]["temp_min"] for s in slots)
            temp_max    = max(s["main"]["temp_max"] for s in slots)
            humidity    = round(sum(s["main"]["humidity"] for s in slots) / len(slots))
            description = rep_slot["weather"][0]["description"]
            icon        = rep_slot["weather"][0]["icon"]

            dt_obj   = datetime.strptime(date_str, "%Y-%m-%d")
            date_fmt = dt_obj.strftime("%d/%m/%Y")

            forecast.append({
                "date":         date_fmt,
                "temp_min":     round(temp_min, 1),
                "temp_max":     round(temp_max, 1),
                "description":  description,
                "humidity_avg": humidity,
                "icon":         icon,
            })

        result = {
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecast": forecast,
        }
        return json.dumps(result, ensure_ascii=False)

    except requests.exceptions.HTTPError:
        if resp.status_code == 404:
            return json.dumps({"error": f"Không tìm thấy '{location}'. Thử tên tiếng Anh."})
        return json.dumps({"error": f"HTTP error: {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def send_telegram_message(message: str) -> str:
    """
    Gửi tin nhắn thời tiết đã được format tới Telegram.

    Args:
        message: Nội dung tin nhắn có emoji và Markdown để gửi qua Telegram.

    Returns:
        Thông báo trạng thái gửi.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    # _CURRENT_CHAT_ID được set động mỗi lần user nhắn tin
    chat_id = os.getenv("_CURRENT_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return "❌ Lỗi: TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID chưa cấu hình."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        global _message_delivered
        _message_delivered = True
        return "✅ Tin nhắn đã gửi thành công lên Telegram!"
    except Exception as e:
        return f"❌ Lỗi khi gửi Telegram: {str(e)}"


@tool
def send_plain_message(message: str) -> str:
    """
    Gửi tin nhắn hội thoại thông thường tới người dùng qua Telegram.
    Dùng cho mọi phản hồi KHÔNG phải báo cáo thời tiết (chào hỏi, giải đáp, v.v.).

    Args:
        message: Nội dung phản hồi bằng text thường (không cần Markdown).

    Returns:
        Thông báo trạng thái gửi.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = os.getenv("_CURRENT_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return "❌ Lỗi: TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID chưa cấu hình."

    url     = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        global _message_delivered
        _message_delivered = True
        return "✅ Tin nhắn đã gửi thành công!"
    except Exception as e:
        return f"❌ Lỗi khi gửi Telegram: {str(e)}"


# ===========================================================================
# SECTION 3: SYSTEM PROMPTS
# ===========================================================================

WEATHER_ANALYST_PROMPT = """Bạn là Weather Analyst — chuyên gia phân tích thời tiết.

Bạn sẽ nhận task có ghi rõ intent ở đầu: [INTENT:current], [INTENT:forecast], hoặc [INTENT:both].

Quy tắc sử dụng tool:
- [INTENT:current]  → chỉ dùng `get_weather`
- [INTENT:forecast] → chỉ dùng `get_5day_forecast`
- [INTENT:both]     → dùng cả hai tool

Sau khi lấy dữ liệu, phân tích và viết kết quả bằng tiếng Việt:

Với dữ liệu HIỆN TẠI (get_weather):
- Trích xuất `local_datetime` từ kết quả (thời gian thực tế tại địa điểm)
- Nhận xét: nhiệt độ cảm nhận, độ ẩm, gió, tình trạng bầu trời
- Lời khuyên: ăn mặc, mang ô, uống nước...

Với dữ liệu DỰ BÁO (get_5day_forecast):
- Liệt kê đủ 5 ngày: ngày, nhiệt độ min/max, mô tả, độ ẩm
- Nhận xét xu hướng thời tiết tổng thể

KHÔNG gửi Telegram — đó là việc của Reporter.
Trả toàn bộ kết quả phân tích đầy đủ, giữ nguyên tag [INTENT:...] ở đầu output.
"""

WEATHER_REPORTER_PROMPT = """Bạn là Weather Reporter — chuyên gia soạn và gửi báo cáo thời tiết qua Telegram.

Bạn nhận dữ liệu phân tích từ Weather Analyst. Dữ liệu có ghi rõ intent ở đầu.

QUY TẮC FORMAT:

[INTENT:current] → Soạn 1 tin nhắn:
📍 *[Địa điểm]* — [emoji thời tiết phù hợp]
🕐 *Thời gian*: [local_datetime]
🌡️ Nhiệt độ: *X°C* (cảm giác như X°C)
💧 Độ ẩm: *X%*
💨 Gió: *X m/s*
📝 [Nhận xét 1-2 câu]
👗 [Gợi ý trang phục]
👉 _[Lời khuyên thực tế]_
⏱️ _Dữ liệu lúc [local_datetime] theo giờ địa phương_

[INTENT:forecast] → Soạn 1 tin nhắn:
📅 *Dự báo 5 ngày — [Địa điểm]*

[emoji] *dd/mm* — X–X°C 💧X% — [mô tả ngắn]
(lặp lại cho đủ 5 ngày)

⏱️ _Nguồn: OpenWeatherMap_

[INTENT:both] → Soạn 1 tin nhắn gộp:
📍 *[Địa điểm]* — [emoji thời tiết hiện tại]
🕐 *[local_datetime]*

━━━ HIỆN TẠI ━━━
🌡️ *X°C* (cảm giác X°C) | 💧 X% | 💨 X m/s
📝 [Nhận xét ngắn]
👗 [Trang phục] · 🌂 [có/không cần ô]

━━━ DỰ BÁO 5 NGÀY ━━━
[emoji] *dd/mm* — X–X°C — [mô tả ngắn]
(lặp lại cho đủ 5 ngày)

⏱️ _OpenWeatherMap · Giờ địa phương_

Dùng Markdown: *bold* cho số liệu, _italic_ cho ghi chú.
Gọi tool `send_telegram_message` để gửi tin nhắn vừa soạn.
"""

CHAT_AGENT_PROMPT = """Bạn là Chat Agent — trợ lý hội thoại thân thiện.

Bạn sẽ nhận tin nhắn thông thường từ người dùng (chào hỏi, hỏi về bot, câu hỏi chung...).
Nhiệm vụ của bạn:
1. Soạn câu trả lời tự nhiên, thân thiện bằng tiếng Việt.
2. Gọi tool `send_plain_message` để gửi câu trả lời đó.
"""

SUPERVISOR_PROMPT = """Bạn là Supervisor — người điều phối hệ thống dự báo thời tiết.

Bạn có 3 công cụ:
- `call_weather_analyst`: Lấy và phân tích thời tiết (current, forecast, hoặc cả hai).
- `call_weather_reporter`: Soạn và gửi báo cáo thời tiết qua Telegram.
- `call_chat_agent`: Trả lời tin nhắn hội thoại thông thường qua Telegram.

BƯỚC 1 — Xác định intent từ tin nhắn người dùng:
- Người dùng hỏi thời tiết HIỆN TẠI (bây giờ, hôm nay, lúc này) → intent = "current"
- Người dùng hỏi DỰ BÁO (5 ngày tới, tuần tới, sắp tới) → intent = "forecast"
- Người dùng hỏi CẢ HAI → intent = "both"
- Người dùng chỉ gửi tên thành phố không rõ ý → mặc định intent = "current"
- Tin nhắn không liên quan thời tiết → gọi `call_chat_agent`, bỏ qua bước còn lại

BƯỚC 2 — Gọi `call_weather_analyst` với task string theo format:
"[INTENT:{intent}] Lấy thời tiết cho {địa điểm}"

BƯỚC 3 — Gọi `call_weather_reporter` với toàn bộ kết quả từ bước 2,
giữ nguyên tag [INTENT:{intent}] ở đầu khi truyền vào.
"""


# ===========================================================================
# SECTION 4: AGENT BUILDER FUNCTIONS (LAZY)
#
# Mỗi hàm nhận LLM → trả về agent đã compile.
# Được truyền vào invoke_with_fallback() như callback,
# cho phép build lại với model khác khi cần fallback.
#
# Lưu ý: call_weather_analyst và call_telegram_messenger phải được
# định nghĩa SAU các builder functions vì chúng tham chiếu đến nhau.
# ===========================================================================

def build_weather_analyst(llm):
    return create_react_agent(
        model=llm,
        tools=[get_weather, get_5day_forecast],
        prompt=WEATHER_ANALYST_PROMPT,
        name="weather_analyst",
    )


def build_weather_reporter(llm):
    return create_react_agent(
        model=llm,
        tools=[send_telegram_message],
        prompt=WEATHER_REPORTER_PROMPT,
        name="weather_reporter",
    )


def build_chat_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[send_plain_message],
        prompt=CHAT_AGENT_PROMPT,
        name="chat_agent",
    )


def build_supervisor(llm):
    return create_react_agent(
        model=llm,
        tools=[call_weather_analyst, call_weather_reporter, call_chat_agent],
        prompt=SUPERVISOR_PROMPT,
        name="supervisor",
    )


# ===========================================================================
# SECTION 5: HANDOFF TOOLS
#
# Mỗi sub-agent được wrap thành @tool. Supervisor gọi tool này →
# invoke toàn bộ ReAct graph của sub-agent với fallback model riêng.
#
# Mỗi tầng có fallback độc lập:
#   supervisor    → thử 2.5-flash → 2.0-flash → ...
#   weather agent → thử 2.5-flash → 2.0-flash → ... (độc lập)
#   messenger     → thử 2.5-flash → 2.0-flash → ... (độc lập)
# ===========================================================================

def _extract_final_response(result: dict) -> str:
    """Lấy AIMessage cuối cùng không có tool_calls từ kết quả agent."""
    final = next(
        msg for msg in reversed(result["messages"])
        if hasattr(msg, "content")
        and msg.content
        and not getattr(msg, "tool_calls", None)
    )
    return final.content


@tool
def call_weather_analyst(task: str) -> str:
    """
    Giao nhiệm vụ cho Weather Analyst Agent để lấy và phân tích thời tiết.

    Dùng tool này khi cần thông tin thời tiết của một địa điểm —
    hiện tại, dự báo 5 ngày, hoặc cả hai.

    Args:
        task: Task string theo format "[INTENT:current|forecast|both] tên địa điểm".
              Supervisor phải ghi rõ intent ở đầu task string.

    Returns:
        Kết quả phân tích thời tiết đầy đủ theo intent được yêu cầu.
    """
    print("\n  [Weather Analyst] Đang xử lý...")
    agent  = build_weather_analyst(build_llm())
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    return _extract_final_response(result)


@tool
def call_weather_reporter(analysis: str) -> str:
    """
    Giao nhiệm vụ cho Weather Reporter Agent để soạn và gửi báo cáo thời tiết qua Telegram.
    Chỉ dùng sau khi đã có kết quả từ call_weather_analyst.

    Args:
        analysis: Toàn bộ kết quả phân tích thời tiết từ Weather Analyst.

    Returns:
        Trạng thái gửi tin nhắn (thành công / thất bại).
    """
    print("\n  [Weather Reporter] Đang soạn và gửi báo cáo thời tiết...")
    agent  = build_weather_reporter(build_llm())
    result = agent.invoke({"messages": [{"role": "user", "content": analysis}]})
    return _extract_final_response(result)


@tool
def call_chat_agent(message: str) -> str:
    """
    Giao nhiệm vụ cho Chat Agent để trả lời tin nhắn hội thoại thông thường qua Telegram.
    Chỉ dùng cho các tin nhắn KHÔNG liên quan đến thời tiết.

    Args:
        message: Nội dung tin nhắn hội thoại của người dùng cần được trả lời.

    Returns:
        Trạng thái gửi tin nhắn (thành công / thất bại).
    """
    print("\n  [Chat Agent] Đang soạn câu trả lời...")
    agent  = build_chat_agent(build_llm())
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return _extract_final_response(result)


# ===========================================================================
# SECTION 6: PIPELINE RUNNER
# ===========================================================================

def run_agent(user_message: str, chat_id: str) -> bool:
    """Chạy toàn bộ agent pipeline. Trả về True nếu tin nhắn được gửi thành công lên Telegram."""
    global _message_delivered
    _message_delivered = False
    os.environ["_CURRENT_CHAT_ID"] = str(chat_id)

    print(f"\n🚀 Agent | chat_id={chat_id} | message={user_message[:80]}")
    print("=" * 60)
    print("\n[Supervisor] Bắt đầu điều phối...")

    agent  = build_supervisor(build_llm())
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    print("\n📋 LUỒNG XỬ LÝ:")
    print("-" * 60)
    for msg in result["messages"]:
        msg_type   = type(msg).__name__
        name       = getattr(msg, "name", None)
        content    = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])

        if msg_type == "HumanMessage":
            print(f"\n👤 [User]: {content[:200]}")
        elif msg_type == "AIMessage" and tool_calls:
            for tc in tool_calls:
                print(f"\n🤖 [Supervisor] → {tc['name']}")
                print(f"   {str(tc.get('args', {}))[:150]}...")
        elif msg_type == "ToolMessage":
            preview = str(content)[:300]
            print(f"\n🔧 [{name}]: {preview}{'...' if len(str(content)) > 300 else ''}")
        elif msg_type == "AIMessage" and content and not tool_calls:
            print(f"\n✅ [Supervisor] kết luận: {content}")

    print("\n" + "=" * 60)
    print("🎉 Pipeline hoàn tất!\n")
    return _message_delivered


# ===========================================================================
# SECTION 7: TELEGRAM BOT (POLLING)
# ===========================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Nhận tin nhắn từ Telegram → chạy agent → kết quả tự reply."""
    user_message = update.message.text.strip()
    chat_id      = update.message.chat_id

    await update.message.reply_text("⏳ Đang xử lý...")

    try:
        delivered = run_agent(user_message=user_message, chat_id=str(chat_id))
        if not delivered:
            await update.message.reply_text(
                "⚠️ Xử lý xong nhưng tin nhắn không được gửi thành công. Vui lòng thử lại."
            )
    except Exception as e:
        logger.error(f"Lỗi agent: {e}")
        await update.message.reply_text(
            f"❌ Có lỗi xảy ra:\n`{str(e)}`",
            parse_mode="Markdown",
        )


def start_bot() -> None:
    """Khởi động Telegram bot với polling loop."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN chưa được cấu hình trong .env")

    print("🤖 Weather Bot đang khởi động...")
    print(f"🦙 Model: Ollama / {OLLAMA_MODEL}")
    print("📱 Nhắn tên thành phố hoặc bất kỳ câu hỏi nào lên Telegram!")
    print("   Ví dụ: Hanoi | Da Nang | Xin chào!")
    print("   Nhấn Ctrl+C để dừng.\n")

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(poll_interval=2, drop_pending_updates=True)


if __name__ == "__main__":
    start_bot()