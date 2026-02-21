import argparse
import json
import os
import platform
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from main import generate_hybrid

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONTACTS_FILE = DATA_DIR / "contacts.json"
STATE_FILE = DATA_DIR / "actions_state.json"

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                "minute": {"type": "integer", "description": "Minute to set the alarm for"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                "message": {"type": "string", "description": "The message content to send"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Reminder title"},
                "time": {"type": "string", "description": "Time for the reminder"},
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search for a contact by name",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "Song or playlist name"}
            },
            "required": ["song"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Number of minutes"}
            },
            "required": ["minutes"],
        },
    },
]

WEATHER_CODE = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "snow",
    80: "rain showers",
    95: "thunderstorm",
}


def _http_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "functiongemma-hackathon-demo/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ensure_state() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        state = {"alarms": [], "timers": [], "reminders": [], "messages": []}
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_contacts() -> List[Dict[str, str]]:
    if not CONTACTS_FILE.exists():
        return []
    return json.loads(CONTACTS_FILE.read_text(encoding="utf-8"))


def get_weather(location: str) -> Dict[str, Any]:
    try:
        geo_q = urllib.parse.quote(location)
        geo = _http_json(f"https://geocoding-api.open-meteo.com/v1/search?name={geo_q}&count=1")
        results = geo.get("results") or []
        if not results:
            return {"ok": False, "error": f"No geocoding result for '{location}'"}

        item = results[0]
        lat, lon = item["latitude"], item["longitude"]
        weather = _http_json(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m"
        )

        current = weather.get("current", {})
        code = current.get("weather_code")
        return {
            "ok": True,
            "location": item.get("name", location),
            "country": item.get("country"),
            "temperature_c": current.get("temperature_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "condition": WEATHER_CODE.get(code, f"code={code}"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Weather API request failed: {exc}"}


def search_contacts(query: str) -> Dict[str, Any]:
    q = query.strip().lower()
    contacts = load_contacts()
    matches = [c for c in contacts if q in c.get("name", "").lower()]
    return {"ok": True, "matches": matches}


def send_message(recipient: str, message: str) -> Dict[str, Any]:
    state = _ensure_state()
    entry = {
        "recipient": recipient,
        "message": message,
        "timestamp": int(time.time()),
    }

    # Optional real integration: post outbound message to a webhook if configured.
    webhook = os.environ.get("MESSAGE_WEBHOOK_URL")
    if webhook:
        payload = json.dumps(entry).encode("utf-8")
        req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as _:
            pass

    state["messages"].append(entry)
    _save_state(state)
    return {"ok": True, "queued": True, "delivery": "webhook" if webhook else "local-log", "entry": entry}


def set_alarm(hour: int, minute: int) -> Dict[str, Any]:
    state = _ensure_state()
    entry = {"hour": hour, "minute": minute, "timestamp": int(time.time())}
    state["alarms"].append(entry)
    _save_state(state)
    return {"ok": True, "scheduled": entry}


def set_timer(minutes: int) -> Dict[str, Any]:
    state = _ensure_state()
    end_epoch = int(time.time()) + int(minutes) * 60
    entry = {"minutes": int(minutes), "end_epoch": end_epoch, "timestamp": int(time.time())}
    state["timers"].append(entry)
    _save_state(state)
    return {"ok": True, "scheduled": entry}


def create_reminder(title: str, time_text: str) -> Dict[str, Any]:
    state = _ensure_state()
    entry = {"title": title, "time": time_text, "timestamp": int(time.time())}
    state["reminders"].append(entry)
    _save_state(state)
    return {"ok": True, "scheduled": entry}


def play_music(song: str) -> Dict[str, Any]:
    query = urllib.parse.quote(song)
    url = f"https://music.youtube.com/search?q={query}"
    opened = False
    if platform.system() == "Darwin" and os.environ.get("OPEN_MUSIC", "0") == "1":
        try:
            subprocess.run(["open", url], check=False)
            opened = True
        except Exception:
            opened = False
    return {"ok": True, "song": song, "url": url, "opened": opened}


def execute_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments", {}) or {}

    if name == "get_weather":
        return {"tool": name, "result": get_weather(args.get("location", ""))}
    if name == "search_contacts":
        return {"tool": name, "result": search_contacts(args.get("query", ""))}
    if name == "send_message":
        return {"tool": name, "result": send_message(args.get("recipient", ""), args.get("message", ""))}
    if name == "set_alarm":
        return {"tool": name, "result": set_alarm(int(args.get("hour", 0)), int(args.get("minute", 0)))}
    if name == "set_timer":
        return {"tool": name, "result": set_timer(int(args.get("minutes", 0)))}
    if name == "create_reminder":
        return {"tool": name, "result": create_reminder(args.get("title", ""), args.get("time", ""))}
    if name == "play_music":
        return {"tool": name, "result": play_music(args.get("song", ""))}

    return {"tool": name, "result": {"ok": False, "error": "Unknown tool"}}


def run_text_command(user_text: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": user_text}]
    route = generate_hybrid(messages, TOOLS)
    executions = [execute_call(c) for c in route.get("function_calls", [])]
    return {
        "input": user_text,
        "route": route,
        "executions": executions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end product demo: route and execute tool calls")
    parser.add_argument("--text", required=True, help="User utterance")
    args = parser.parse_args()

    output = run_text_command(args.text)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
