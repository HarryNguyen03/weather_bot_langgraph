"""
chart_utils.py — Render biểu đồ dự báo thời tiết và gửi lên Telegram.

Chạy trong MAIN PROCESS (không qua MCP subprocess) để tránh:
  - Python subprocess cold-start overhead mỗi request
  - matplotlib cold-import (~0.92s) mỗi lần subprocess restart

matplotlib được import lazy BÊN TRONG render_forecast_chart —
lần đầu ~0.92s, các lần sau ~0s (đã cache trong sys.modules của main process).
"""

import json
import os
import requests


def render_forecast_chart(forecast_json: str) -> bytes:
    """Tạo ảnh biểu đồ dự báo 5 ngày từ JSON string. Trả về PNG bytes."""
    # Lazy import — lần đầu ~0.92s, sau đó cache trong sys.modules
    import sys
    import warnings
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib
        matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import matplotlib.pyplot as plt
    from io import BytesIO

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
    plt.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", facecolor="#0f1923")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def send_chart_to_telegram(forecast_json: str, chat_id: str) -> str:
    """
    Render biểu đồ và gửi ảnh PNG lên Telegram.
    Gọi trực tiếp trong main process — KHÔNG qua MCP subprocess.

    Returns:
        Status string (bắt đầu "✅" nếu thành công).
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
