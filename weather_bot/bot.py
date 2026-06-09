"""
Telegram Bot Handler — Entry point của Weather Bot.
====================================================

Nhận tin nhắn từ Telegram → gọi run_agent → kết quả tự reply.

Thay đổi so với file gốc:
  - Xóa os.environ["_CURRENT_CHAT_ID"]: chat_id truyền trực tiếp vào run_agent()
  - Xóa _message_delivered global flag: run_agent() trả về bool trực tiếp
  - handle_message gọi await run_agent() — compatible với Telegram async handler
"""

import os
import re
import logging
import traceback

from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from supervisor import run_agent

MAX_MESSAGE_LENGTH = 500

INJECTION_PATTERNS = [
    r"ignore.{0,20}(previous|prior|above|instruction)",
    r"(system|hidden|secret).{0,10}prompt",
    r"(api.?key|token|password|secret|\.env)",
    r"jailbreak|do anything now",
    r"bỏ qua.{0,20}(hướng dẫn|lệnh|instruction)",
    r"(tiết lộ|reveal|leak|show).{0,20}(key|token|secret|prompt|config)",
    r"pretend|roleplay|act as|you are now",
    r"forget.{0,20}(instruction|rule|constraint)",
]

def is_safe_input(text: str) -> bool:
    if len(text) > MAX_MESSAGE_LENGTH:
        return False
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return False
    return True

# .env nằm ở Dependencies/ — một cấp trên weather_bot/
load_dotenv(Path(__file__).parent.parent / "Dependencies" / ".env")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.WARNING,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ===========================================================================
# TELEGRAM HANDLER
# ===========================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Nhận tin nhắn từ Telegram → chạy agent → kết quả tự reply."""
    user_message = update.message.text.strip()
    chat_id      = str(update.message.chat_id)

    # Lớp 1: validate input trước khi vào agent
    if not is_safe_input(user_message):
        await update.message.reply_text(
            "⚠️ Tin nhắn không hợp lệ. Vui lòng chỉ hỏi về thời tiết."
        )
        return

    await update.message.reply_text("⏳ Đang xử lý...")

    try:
        delivered = await run_agent(user_message=user_message, chat_id=chat_id)
        if not delivered:
            await update.message.reply_text(
                "⚠️ Xử lý xong nhưng tin nhắn không được gửi thành công. Vui lòng thử lại."
            )
    except Exception as e:
        logger.error(f"Lỗi agent: {e}\n{traceback.format_exc()}")
        # ExceptionGroup (Python 3.11+): log từng sub-exception để dễ debug
        if hasattr(e, "exceptions"):
            for i, sub in enumerate(e.exceptions, 1):
                logger.error(
                    f"  Sub-exception {i}: {sub}\n"
                    + "".join(traceback.format_exception(type(sub), sub, sub.__traceback__))
                )
        await update.message.reply_text(
            f"❌ Có lỗi xảy ra:\n`{str(e)}`",
            parse_mode="Markdown",
        )


# ===========================================================================
# BOT STARTUP
# ===========================================================================

def start_bot() -> None:
    """Khởi động Telegram bot với polling loop."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN chưa được cấu hình trong .env")

    print("🤖 Weather Bot đang khởi động...")
    print(f"🦙 Model: Ollama / gemma4")
    print("📱 Nhắn tên thành phố hoặc bất kỳ câu hỏi nào lên Telegram!")
    print("   Ví dụ: Hanoi | Da Nang | 5 ngày tới Đà Lạt | Xin chào!")
    print("   Nhấn Ctrl+C để dừng.\n")

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(poll_interval=2, drop_pending_updates=True)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    start_bot()
