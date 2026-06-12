# 🌤️ Weather Bot — LangGraph + MCP + Self-RAG

> Telegram bot dự báo thời tiết thông minh, xây dựng trên kiến trúc **Supervisor + Subagents** với **LangGraph**, tích hợp **MCP (Model Context Protocol)**, **Self-RAG** và **Validator reflection loop**.

---

## 📌 Tổng quan

Weather Bot là một hệ thống AI agent hoàn chỉnh cho phép người dùng tra cứu thời tiết hiện tại và dự báo 5 ngày tới trực tiếp qua **Telegram**. Bot không chỉ cung cấp số liệu thời tiết mà còn đưa ra **lời khuyên thông minh** về trang phục, sức khỏe và hoạt động ngoài trời dựa trên knowledge base nội bộ, với cơ chế tự kiểm định chất lượng trước khi gửi cho người dùng.

### Điểm nổi bật
- 🤖 **Hybrid Multi-Agent Architecture** — Agent ở nơi cần quyết định, workflow ở nơi không: Supervisor (agent) + Analyst (agent) + Reporter (workflow)
- 🧠 **Self-RAG** — Analyst tự quyết định khi nào tra cứu knowledge base, tự chấm điểm tài liệu (grade), tự viết lại query nếu kết quả chưa phù hợp (rewrite)
- ✅ **Validator Reflection Loop** — 3 tầng kiểm định (rule-based → hallucination check → LLM judge) với **deterministic retry**: đường retry là code thuần, không phụ thuộc LLM
- 🔌 **MCP Integration** — Weather tools và Telegram tools chạy qua MCP stdio transport
- 📊 **Observability** — LangSmith tracing đầy đủ với run name, tags, metadata theo từng request
- ⚡ **Production Hardening** — First token giảm **~96%**, tổng latency giảm tới **65%** sau refactor (số liệu bên dưới)
- 🛡️ **Prompt Injection Defense** — Lớp bảo vệ chống tấn công qua Telegram input
- 📈 **Auto Chart** — Tự động render và gửi biểu đồ dự báo PNG qua Telegram

---

## 🏗️ Kiến trúc hệ thống

### Tổng quan

```
Telegram User
     │
     ▼
bot.py — Input validation + Prompt injection defense
     │
     ▼
Supervisor StateGraph  (compile MỘT LẦN lúc startup)
     │
     ├── tool: call_weather_analyst ──► Weather Analyst        [SUB-AGENT]
     │                                  Self-RAG StateGraph
     │                                    ├── get_weather          (MCP)
     │                                    ├── get_5day_forecast    (MCP)
     │                                    └── retrieve_weather_knowledge (RAG)
     │
     ├── tool: call_weather_reporter ─► Weather Reporter       [LLM WORKFLOW]
     │                                    ├── chart_utils (PNG, main process)
     │                                    ├── 1 lần LLM call soạn nội dung
     │                                    └── Python split & gửi qua MCP
     │
     └── tool: send_plain_message ────► Telegram MCP (hội thoại thường)
```

### Supervisor StateGraph

```
START → agent ──(tool_calls)──► tools ──┬──(analyst xong)──► validator
          ▲                             │                       │
          │                             │                  PASS │ FAIL (retry ≤ 2)
          │                             │                       │      │
          └────────(reporter)───────────┘                       │      ▼
          │                                                     │  force_retry
END ◄──(reporter/plain done)                                    │      │
                                                  agent ◄───────┘      └──► tools
                                                                      (re-run analyst,
                                                                       KHÔNG qua LLM)
```

Điểm thiết kế quan trọng: khi Validator FAIL, node `force_retry` **tự tạo tool call** gọi lại Analyst bằng code thuần — không để LLM quyết định, loại bỏ hoàn toàn rủi ro LLM bỏ qua lệnh retry và gửi output chưa kiểm định cho người dùng.

### Weather Analyst — Self-RAG StateGraph

```
START → agent ──(tool_calls)──► tools ──┬──(có retrieve)──► grade_documents
          ▲                             │                        │
          │                             │              relevant  │  not relevant
          │◄──────(weather tools)───────┘              hoặc hết  │  (quota 1 lần)
          │                                            quota     │
          │◄───────────────────────────────────────────┘         ▼
          │                                              rewrite_question
          │◄─────────────────────────────────────────────────────┘
          │
          └──(không còn tool_calls)──► END
```

Analyst **tự quyết định** khi nào cần tra cứu knowledge base và tự soạn query tập trung (VD: `"lời khuyên chạy bộ thời tiết nóng"`) thay vì dùng raw input — tránh nhiễu embedding. Nếu `grade_documents` đánh giá tài liệu không phù hợp, `rewrite_question` viết lại query và tra cứu lại (giới hạn 1 lần để không kẹt vòng lặp).

### Agent hay Workflow?

Nguyên tắc thiết kế: **chỉ trao quyền tự quyết (agency) khi task thực sự cần nó.**

| Component | Loại | Lý do |
|---|---|---|
| Supervisor | Agent (StateGraph) | Cần LLM phân tích intent và routing |
| Weather Analyst | Agent (Self-RAG StateGraph) | Cần tự quyết gọi tool nào, tự soạn query RAG |
| Weather Reporter | LLM Workflow | Format cố định theo intent — không có gì để quyết, 1 LLM call + Python gửi |
| Validator | Workflow node | Rule-based + hallucination check + LLM judge |
| force_retry | Deterministic node | Retry không được phép phụ thuộc LLM |

---

## ⚡ Production Hardening — Số liệu thực đo

Đo bằng LangSmith tracing, cùng điều kiện (Ollama gemma4 local), trước và sau chuỗi refactor:

| Test case | Total latency (trước) | Total latency (sau) | Cải thiện | First token (trước) | First token (sau) |
|---|---|---|---|---|---|
| Current weather | 151.55s | 111.43s | **-26%** | 21.85s | **0.94s** |
| RAG (lời khuyên) | 557.91s | 266.41s | **-52%** | 21.92s | **0.98s** |
| Forecast 5 ngày | 382.05s | 135.26s | **-65%** | 21.39s | **0.87s** |

**First token giảm ~96%** nhờ các tối ưu chính:

1. **Hoist resources lên startup** — MCP client + `get_tools()` (2 subprocess spawn), vectorstore build, LLM init, graph compile: tất cả chạy một lần lúc bot khởi động thay vì mỗi request
2. **Reporter: ReAct sub-agent → LLM workflow** — từ 4-6 LLM call xuống đúng 1 call; số tin nhắn do code quyết định, loại bỏ hoàn toàn bug LLM loop gửi trùng tin
3. **State-based data flow** — `chat_id`, `forecast_json`, `rag_docs`, `validator_feedback` chảy qua LangGraph State (`InjectedState` + `Command`), không còn closure dict — an toàn với concurrent users
4. **Telegram Markdown fallback** — tin nhắn lỗi parse Markdown (400) tự động gửi lại dạng plain text thay vì fail
5. **Grade batch + rewrite quota** — chấm cả batch docs trong 1 LLM call, giới hạn cứng 1 lần rewrite

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (`gemma4`, chạy local) |
| Orchestration | LangGraph `StateGraph` (custom graphs, compile-once) |
| State mechanics | `InjectedState`, `Command`, `add_messages` reducer |
| MCP | FastMCP + `langchain-mcp-adapters` (stdio transport) |
| RAG | Self-RAG: retrieve → grade → rewrite loop |
| Embeddings | Ollama `nomic-embed-text` |
| Vector Store | LangChain `InMemoryVectorStore` (k=4) |
| Observability | LangSmith tracing (run_name, tags, metadata) |
| Weather API | OpenWeatherMap |
| Messaging | Telegram Bot API (`python-telegram-bot`) |
| Charting | Matplotlib (main process, lazy import) |

---

## 📁 Cấu trúc thư mục

```
Program_code/
├── Dependencies/
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # API keys (không push lên Git)
│
├── weather_bot/
│   ├── bot.py                 # Telegram handler + input validation + post_init
│   ├── supervisor.py          # Supervisor StateGraph + tools + validator + force_retry
│   ├── analyst.py             # Weather Analyst — Self-RAG StateGraph
│   ├── state.py               # State definition (messages, chat_id, forecast_json, rag_docs...)
│   ├── prompts.py             # System prompts cho mọi component
│   ├── rag.py                 # Vectorstore builder + retriever tool
│   ├── chart_utils.py         # Matplotlib chart renderer (main process)
│   ├── weather_mcp_server.py  # MCP server: get_weather, get_5day_forecast
│   ├── telegram_mcp_server.py # MCP server: send tools + Markdown fallback
│   └── knowledge/             # Knowledge base (.md)
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
- (Tùy chọn) LangSmith API Key cho tracing — [smith.langchain.com](https://smith.langchain.com)

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

Tạo file `Dependencies/.env`:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENWEATHER_API_KEY=your_openweather_api_key

# Tùy chọn — bật LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=weather-bot
```

### 5. Chạy bot

```bash
cd weather_bot
python bot.py
```

Đợi log `[INIT] MCP tools + vectorstore + analyst graph + supervisor graph sẵn sàng.` rồi nhắn tin cho bot trên Telegram.

---

## 💬 Cách sử dụng

| Tin nhắn | Kết quả |
|---|---|
| `Hanoi` | Thời tiết hiện tại tại Hà Nội (2 tin) |
| `5 ngày tới Đà Nẵng` | Biểu đồ PNG + dự báo 5 ngày (2 tin) |
| `Thời tiết Đà Lạt hôm nay và tuần tới` | Biểu đồ + hiện tại + dự báo + lời khuyên (3 tin) |
| `Có nên chạy bộ ở Hà Nội không?` | Self-RAG: tra cứu knowledge base + tư vấn theo thời tiết thực |
| `Xin chào` | Phản hồi hội thoại thông thường |

---

## 🔒 Bảo mật & Độ tin cậy

- **Prompt Injection Defense** — lọc pattern nguy hiểm (cả tiếng Việt lẫn tiếng Anh) trước khi vào agent pipeline; Supervisor prompt có lớp giới hạn hành vi thứ hai
- **Input Length Limit** — tối đa 500 ký tự mỗi tin nhắn
- **chat_id qua InjectedState** — đọc từ LangGraph State theo từng request, không global, không closure — an toàn với concurrent users; schema tool mà LLM nhìn thấy không chứa `chat_id`
- **Location validation** — chặn injection qua tham số location ở MCP server
- **Deterministic guards** — số tin nhắn gửi do code giới hạn (hard limit + truncate 4096 ký tự theo Telegram), không phụ thuộc LLM
- **Hallucination check** — lời khuyên từ RAG được đối chiếu với knowledge base; chỉ fail khi mâu thuẫn trực tiếp hoặc gán sai nguồn

---

## 📊 Observability

Mỗi request được trace đầy đủ trên LangSmith:
- `run_name` chứa nội dung câu hỏi → nhìn list trace biết ngay request nào
- `tags` để filter theo version khi so sánh before/after
- `metadata.chat_id` để truy trace của đúng user khi debug
- Analyst graph hiện thành nested subgraph — quan sát được từng bước retrieve → grade → rewrite

---

## 📄 License

MIT License — feel free to use and modify.

---

*Built with ❤️ using LangGraph + MCP + Self-RAG*
