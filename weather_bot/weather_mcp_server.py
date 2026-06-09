"""
Weather MCP Server
==================
Cung cấp weather tools qua stdio transport.
Expose: get_weather, get_5day_forecast

Chạy độc lập:
    python weather_mcp_server.py
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# .env nằm ở Dependencies/ — một cấp trên weather_bot/
load_dotenv(Path(__file__).parent.parent / "Dependencies" / ".env")

import re as _re

_LOCATION_MAX_LEN = 100
_LOCATION_UNSAFE  = _re.compile(r'[<>{}\[\]\\;`\'"=]')

def _validate_location(location: str) -> bool:
    """Kiểm tra tên địa điểm hợp lệ — chặn injection qua tham số location."""
    if not location or len(location.strip()) == 0:
        return False
    if len(location) > _LOCATION_MAX_LEN:
        return False
    if _LOCATION_UNSAFE.search(location):
        return False
    return True

mcp = FastMCP("weather")


# ===========================================================================
# TOOL 1: get_weather — thời tiết hiện tại
# ===========================================================================

@mcp.tool()
def get_weather(location: str) -> str:
    """
    Lấy dữ liệu thời tiết thực tế từ OpenWeatherMap API.

    Args:
        location: Tên thành phố cần tra cứu (VD: 'Hanoi', 'Ho Chi Minh City').

    Returns:
        JSON string chứa thông tin thời tiết chi tiết, bao gồm local_datetime
        — thời gian thực tế tại địa điểm đó theo múi giờ địa phương.
    """
    if not _validate_location(location):
        return json.dumps({"error": "Tên địa điểm không hợp lệ."})
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

        # Tính thời gian địa phương từ Unix timestamp + timezone offset (giây)
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


# ===========================================================================
# TOOL 2: get_5day_forecast — dự báo 5 ngày
# ===========================================================================

@mcp.tool()
def get_5day_forecast(location: str) -> str:
    """
    Lấy dự báo thời tiết 5 ngày từ OpenWeatherMap API.

    Args:
        location: Tên thành phố cần tra cứu (VD: 'Hanoi', 'Ho Chi Minh City').

    Returns:
        JSON string chứa dự báo thời tiết 5 ngày, mỗi ngày gồm
        temp_min, temp_max, description, humidity_avg, icon.
    """
    if not _validate_location(location):
        return json.dumps({"error": "Tên địa điểm không hợp lệ."})
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


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
