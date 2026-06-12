"""
Telegram MCP Server
===================
Cung cấp telegram tools qua stdio transport.
Expose: send_telegram_message, send_plain_message

FIX RACE CONDITION: chat_id là parameter thực sự của từng tool call,
KHÔNG đọc từ os.environ — mỗi request có chat_id độc lập, không xung đột.

Chạy độc lập:
    python telegram_mcp_server.py
"""

import os
import json
import requests
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# matplotlib được import LAZY bên trong render_forecast_chart — KHÔNG import ở đây.
# Lý do: subprocess telegram_mcp_server.py dùng stdio transport; bất kỳ thứ gì
# ghi ra stdout trong lúc khởi động (kể cả matplotlib init) sẽ làm hỏng
# MCP handshake và gây ExceptionGroup ở phía client.

# .env nằm ở Dependencies/ — một cấp trên weather_bot/
load_dotenv(Path(__file__).parent.parent / "Dependencies" / ".env")


# ===========================================================================
# HELPER: render_forecast_chart — tạo ảnh biểu đồ từ JSON dự báo
# ===========================================================================

def render_forecast_chart(forecast_json: str) -> bytes:
    # ── Lazy import matplotlib ──────────────────────────────────────────────
    # Import BÊN TRONG hàm, không phải module level.
    # Tránh matplotlib ghi ra stdout trong lúc MCP subprocess khởi động
    # (bất kỳ stdout nào trước mcp.run() sẽ làm hỏng stdio protocol).
    import sys
    import warnings
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib
        matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import matplotlib.pyplot as plt
    from io import BytesIO
    # ───────────────────────────────────────────────────────────────────────

    data     = json.loads(forecast_json)
    forecast = data["forecast"]
    location = data["location"]

    dates    = [f["date"][:5]     for f in forecast]
    temp_max = [f["temp_max"]     for f in forecast]
    temp_min = [f["temp_min"]     for f in forecast]
    humidity = [f["humidity_avg"] for f in forecast]
    descs    = [f["description"]  for f in forecast]
    x = list(range(len(dates)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0f1923"
    )

    # Temperature panel
    ax1.set_facecolor("#0f1923")
    ax1.fill_between(x, temp_min, temp_max, alpha=0.25, color="#4fc3f7")
    ax1.plot(x, temp_max, "o-", color="#ff7043", linewidth=2.5,
             markersize=8, label="Cao nhất")
    ax1.plot(x, temp_min, "o-", color="#4fc3f7", linewidth=2.5,
             markersize=8, label="Thấp nhất")

    for i, (mx, mn, desc) in enumerate(zip(temp_max, temp_min, descs)):
        ax1.annotate(f"{mx}°", (i, mx), textcoords="offset points",
                     xytext=(0, 10), ha="center",
                     color="#ff7043", fontsize=10, fontweight="bold")
        ax1.annotate(f"{mn}°", (i, mn), textcoords="offset points",
                     xytext=(0, -15), ha="center",
                     color="#4fc3f7", fontsize=10, fontweight="bold")
        ax1.annotate(desc, (i, (mx + mn) / 2), textcoords="offset points",
                     xytext=(0, 0), ha="center",
                     color="#b0bec5", fontsize=8, style="italic")

    ax1.set_title(f"Dự báo 5 ngày — {location}",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, fontsize=11, color="white")
    ax1.set_ylabel("°C", color="#b0bec5")
    ax1.tick_params(colors="#b0bec5")
    ax1.grid(axis="y", alpha=0.12, color="white")
    ax1.legend(loc="upper right", framealpha=0.2, fontsize=9)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color("#2d3748")

    # Humidity panel
    ax2.set_facecolor("#0f1923")
    bars = ax2.bar(x, humidity, color="#4fc3f7", alpha=0.55, width=0.5)
    for bar, h in zip(bars, humidity):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{h}%", ha="center", color="#b0bec5", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dates, fontsize=10, color="white")
    ax2.set_ylabel("Độ ẩm", color="#b0bec5", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.tick_params(colors="#b0bec5")
    ax2.grid(axis="y", alpha=0.1, color="white")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#2d3748")

    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor="#0f1923")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


mcp = FastMCP("telegram")


# ===========================================================================
# TOOL 1: send_telegram_message — gửi Markdown
# ===========================================================================

@mcp.tool()
def send_telegram_message(message: str, chat_id: str) -> str:
    """
    Gửi tin nhắn thời tiết đã được format tới Telegram.

    Args:
        message: Nội dung tin nhắn có emoji và Markdown để gửi qua Telegram.
        chat_id: ID của cuộc trò chuyện Telegram nhận tin nhắn.

    Returns:
        Thông báo trạng thái gửi.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return "❌ Lỗi: TELEGRAM_BOT_TOKEN chưa được cấu hình."
    if not chat_id:
        return "❌ Lỗi: chat_id không hợp lệ."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 400:
            body = resp.text.lower()
            if "parse" in body or "entit" in body:
                plain_payload = {"chat_id": chat_id, "text": message}
                resp2 = requests.post(url, json=plain_payload, timeout=10)
                resp2.raise_for_status()
                return "✅ Tin nhắn đã gửi (plain text fallback do lỗi Markdown)."
        resp.raise_for_status()
        return "✅ Tin nhắn đã gửi thành công lên Telegram!"
    except Exception as e:
        return f"❌ Lỗi khi gửi Telegram: {str(e)}"


# ===========================================================================
# TOOL 2: send_plain_message — gửi plain text
# ===========================================================================

@mcp.tool()
def send_plain_message(message: str, chat_id: str) -> str:
    """
    Gửi tin nhắn hội thoại thông thường tới người dùng qua Telegram.
    Dùng cho mọi phản hồi KHÔNG phải báo cáo thời tiết (chào hỏi, giải đáp, v.v.).

    Args:
        message: Nội dung phản hồi bằng text thường (không cần Markdown).
        chat_id: ID của cuộc trò chuyện Telegram nhận tin nhắn.

    Returns:
        Thông báo trạng thái gửi.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return "❌ Lỗi: TELEGRAM_BOT_TOKEN chưa được cấu hình."
    if not chat_id:
        return "❌ Lỗi: chat_id không hợp lệ."

    url     = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return "✅ Tin nhắn đã gửi thành công!"
    except Exception as e:
        return f"❌ Lỗi khi gửi Telegram: {str(e)}"


# ===========================================================================
# TOOL 3: send_forecast_photo — render biểu đồ và gửi ảnh PNG
# ===========================================================================

@mcp.tool()
def send_forecast_photo(forecast_json: str, chat_id: str) -> str:
    """
    Render biểu đồ dự báo thời tiết 5 ngày và gửi ảnh PNG lên Telegram.

    Args:
        forecast_json: Raw JSON string từ get_5day_forecast.
                       KHÔNG phải text phân tích — phải là JSON gốc.
        chat_id: Telegram chat ID của người dùng.

    Returns:
        Thông báo trạng thái gửi ảnh.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return "❌ Lỗi: TELEGRAM_BOT_TOKEN chưa cấu hình."
    try:
        image_bytes = render_forecast_chart(forecast_json)
        url   = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        files = {"photo": ("forecast.png", image_bytes, "image/png")}
        resp  = requests.post(
            url, data={"chat_id": chat_id}, files=files, timeout=15
        )
        resp.raise_for_status()
        return "✅ Biểu đồ dự báo đã gửi thành công!"
    except Exception as e:
        return f"❌ Lỗi gửi ảnh: {str(e)}"


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
