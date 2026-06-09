# 🌤️ Weather Bot — LangGraph + MCP + Agentic RAG

> Telegram bot dự báo thời tiết thông minh, xây dựng trên kiến trúc **Multi-Agent** với **LangGraph**, tích hợp **MCP (Model Context Protocol)** và **Agentic RAG**.

---

## 📌 Tổng quan

Weather Bot là một hệ thống AI agent hoàn chỉnh cho phép người dùng tra cứu thời tiết hiện tại và dự báo 5 ngày tới trực tiếp qua **Telegram**. Bot không chỉ cung cấp số liệu thời tiết mà còn đưa ra **lời khuyên thông minh** về trang phục, sức khỏe và hoạt động ngoài trời dựa trên knowledge base nội bộ.

### Điểm nổi bật
- 🤖 **Multi-Agent Architecture** — Supervisor điều phối 2 sub-agent chuyên biệt
- 🔌 **MCP Integration** — Weather tools và Telegram tools chạy qua MCP stdio transport
- 🧠 **Agentic RAG** — Retrieval Grader lọc document trước khi inject vào context
- ✅ **Validator Loop** — Tự động kiểm tra và retry nếu output không đạt chất lượng
- 🛡️ **Prompt Injection Defense** — Lớp bảo vệ chống tấn công qua Telegram input
- 📊 **Auto Chart** — Tự động render và gửi biểu đồ dự báo PNG qua Telegram

---

## 🏗️ Kiến trúc hệ thống

```
Telegram User
     │
     ▼
  bot.py  ──────────────────────────────────────────────────────┐
  (Input validation + Prompt injection defense)                  │
     │                                                           │
     ▼                                                           │
supervisor.py  (LangGraph StateGraph)                           │
     │                                                           │
     ├── [MCP Client] ──► weather_mcp_server.py                 │
     │        └── get_weather                                    │
     │        └── get_5day_forecast                              │
     │                                                           │
     ├── [MCP Client] ──► telegram_mcp_server.py                │
     │        └── send_telegram_message                          │
     │        └── send_plain_message                             │
     │                                                           │
     ├── Weather Analyst Agent                                   │
     │        └── RAG (InMemoryVectorStore + nomic-embed-text)   │
     │        └── Retrieval Grader                               │
     │                                                           │
     ├── Validator Node (Rule-based + LLM Judge)                 │
     │        └── Hallucination Check                            │
     │                                                           │
     └── Weather Reporter Agent                                  │
              └── chart_utils.py (matplotlib PNG chart) ────────┘
```

### StateGraph Flow

```
START → agent → tools → validator → agent (retry nếu FAIL)
                  ↑                        │
                  └────────────────────────┘
        validator PASS / max retries → reporter → END
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (`gemma4`) |
| Orchestration | LangGraph `StateGraph` |
| Agent Framework | LangChain `create_react_agent` |
| MCP | FastMCP + `langchain-mcp-adapters` |
| Embeddings | Ollama `nomic-embed-text` |
| Vector Store | LangChain `InMemoryVectorStore` |
| Weather API | OpenWeatherMap |
| Messaging | Telegram Bot API (`python-telegram-bot`) |
| Charting | Matplotlib |

---

## 📁 Cấu trúc thư mục

```
Program_code/
├── Dependencies/
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # API keys (không push lên Git)
│
├── weather_bot/
│   ├── bot.py                 # Telegram handler + input validation
│   ├── supervisor.py          # LangGraph StateGraph orchestrator
│   ├── sub_agents.py          # Weather Analyst & Reporter builders
│   ├── state.py               # State definition
│   ├── prompts.py             # System prompts cho tất cả agents
│   ├── rag.py                 # RAG module + retriever tool
│   ├── chart_utils.py         # Matplotlib chart renderer
│   ├── weather_mcp_server.py  # MCP server: weather tools
│   ├── telegram_mcp_server.py # MCP server: telegram tools
│   └── knowledge/             # Knowledge base (.md files)
│       ├── activity_tips.md
│       ├── health_guidelines.md
│       └── vietnam_climate.md
│
└── Test_files/                # File thử nghiệm trong quá trình dev
```

---

## ⚙️ Cài đặt

### Yêu cầu
- Python 3.11+
- [Ollama](https://ollama.ai) đang chạy local
- Telegram Bot Token (tạo qua [@BotFather](https://t.me/botfather))
- OpenWeatherMap API Key (đăng ký tại [openweathermap.org](https://openweathermap.org))

### 1. Clone repo

```bash
git clone https://github.com/HarryNguyen03/weather_bot_langgraph.git
cd weather_bot_langgraph
```

### 2. Cài dependencies

```bash
pip install -r Dependencies/requirements.txt
```

### 3. Pull models Ollama

```bash
ollama pull gemma4
ollama pull nomic-embed-text
```

### 4. Tạo file `.env`

Tạo file `Dependencies/.env` với nội dung:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENWEATHER_API_KEY=your_openweather_api_key
```

### 5. Chạy bot

```bash
cd weather_bot
python bot.py
```

---

## 💬 Cách sử dụng

Nhắn tin trực tiếp với bot trên Telegram:

| Tin nhắn | Kết quả |
|---|---|
| `Hanoi` | Thời tiết hiện tại tại Hà Nội |
| `5 ngày tới Đà Nẵng` | Dự báo 5 ngày + biểu đồ PNG |
| `Thời tiết Đà Lạt hôm nay và tuần tới` | Hiện tại + dự báo + lời khuyên |
| `Có nên chạy bộ ở Hà Nội không?` | Tư vấn hoạt động dựa trên thời tiết + RAG |
| `Xin chào` | Phản hồi hội thoại thông thường |

---

## 🔒 Bảo mật

- **Prompt Injection Defense**: Lọc các pattern nguy hiểm trước khi vào agent pipeline
- **Input Length Limit**: Giới hạn 500 ký tự mỗi tin nhắn
- **chat_id Isolation**: Mỗi request có `chat_id` độc lập, không dùng global state — tránh race condition với concurrent users
- **MCP Tool Interceptors**: `chat_id` được bind qua closure, không expose cho LLM

---

## 📄 License

MIT License — feel free to use and modify.

---

*Built with ❤️ using LangGraph + MCP + Agentic RAG*
