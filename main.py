import sys
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "cactus/python/src")
functiongemma_path = "weights/functiongemma-270m-it"


# Lazy-loaded external clients so local heuristic routing works even if optional deps are missing.
_CACTUS_API = None
_GENAI_API = None
_GENAI_TYPES = None


def _load_cactus():
    global _CACTUS_API
    if _CACTUS_API is None:
        try:
            from cactus import cactus_init, cactus_complete, cactus_destroy
            _CACTUS_API = (cactus_init, cactus_complete, cactus_destroy)
        except Exception:
            _CACTUS_API = ()
    return _CACTUS_API


def _load_genai():
    global _GENAI_API, _GENAI_TYPES
    if _GENAI_API is None or _GENAI_TYPES is None:
        try:
            from google import genai
            from google.genai import types
            _GENAI_API = genai
            _GENAI_TYPES = types
        except Exception:
            _GENAI_API = None
            _GENAI_TYPES = None
    return _GENAI_API, _GENAI_TYPES


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_time(text: str) -> Optional[Tuple[int, int, str]]:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    meridiem = m.group(3).upper()

    # Keep values intuitive for benchmark-style schemas.
    if meridiem == "PM" and hour != 12:
        hour += 12
    if meridiem == "AM" and hour == 12:
        hour = 0

    return hour, minute, f"{m.group(1)}:{minute:02d} {meridiem}" if m.group(2) else f"{m.group(1)} {meridiem}"


def _split_segments(text: str) -> List[str]:
    normalized = _normalize_space(text)
    segments = re.split(r",\s*|\s+and\s+", normalized, flags=re.IGNORECASE)
    return [s.strip(" .") for s in segments if s.strip(" .")]


def _extract_weather(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(weather|forecast|temperature|temp)\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s'\-]+)", segment)
    if not m:
        return None
    location = _normalize_space(m.group(1)).strip("?.!")
    return {"location": location}


def _extract_alarm(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(alarm|wake me up)\b", segment, re.IGNORECASE):
        return None
    found = _find_time(segment)
    if not found:
        return None
    hour, minute, _ = found
    return {"hour": hour, "minute": minute}


def _extract_timer(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(timer|countdown)\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\b(\d+)\s*(?:minute|minutes|min)\b", segment, re.IGNORECASE)
    if not m:
        return None
    return {"minutes": int(m.group(1))}


def _extract_message(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(send|text|message)\b", segment, re.IGNORECASE):
        return None

    recipient = None
    m_to = re.search(r"\bto\s+([A-Z][a-zA-Z'\-]+)\b", segment, re.IGNORECASE)
    m_text = re.search(r"\btext\s+([A-Z][a-zA-Z'\-]+)\b", segment, re.IGNORECASE)
    if m_to:
        recipient = m_to.group(1)
    elif m_text:
        recipient = m_text.group(1)

    message = None
    m_saying = re.search(r"\bsaying\s+(.+)$", segment, re.IGNORECASE)
    if m_saying:
        message = m_saying.group(1).strip(" .")

    if recipient and message:
        return {"recipient": recipient, "message": message}
    return None


def _extract_reminder(segment: str) -> Optional[Dict]:
    if not re.search(r"\bremind me\b", segment, re.IGNORECASE):
        return None

    m = re.search(
        r"\bremind me\s+(?:to|about)?\s*(.+?)\s+at\s+((?:\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)))",
        segment,
        re.IGNORECASE,
    )
    if not m:
        return None

    title = m.group(1).strip(" .")
    time_str = m.group(2).upper().replace("  ", " ")

    title = re.sub(r"^the\s+", "", title, flags=re.IGNORECASE)
    title = _normalize_space(title)
    return {"title": title, "time": time_str}


def _extract_search_contacts(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(find|look up|search)\b", segment, re.IGNORECASE):
        return None
    if not re.search(r"\bcontacts?\b", segment, re.IGNORECASE):
        return None

    m = re.search(r"\b(?:find|look up|search for?)\s+([A-Z][a-zA-Z'\-]+)\b", segment, re.IGNORECASE)
    if not m:
        return None
    return {"query": m.group(1)}


def _extract_music(segment: str) -> Optional[Dict]:
    if not re.search(r"\bplay\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\bplay\s+(?:some\s+)?(.+)$", segment, re.IGNORECASE)
    if not m:
        return None
    song = m.group(1).strip(" .")
    if re.search(r"\bplay\s+some\s+", segment, re.IGNORECASE):
        song = re.sub(r"\bmusic\b$", "", song, flags=re.IGNORECASE).strip()
    if not song:
        return None
    return {"song": song}


def _required_ok(args: Dict, tool: Dict) -> bool:
    required = tool.get("parameters", {}).get("required", [])
    return all(key in args and args[key] not in (None, "") for key in required)


def _latest_user_text(messages: List[Dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _heuristic_route(messages: List[Dict], tools: List[Dict]):
    text = _latest_user_text(messages)
    if not text:
        return [], 0.0

    tool_map = {t.get("name"): t for t in tools}
    segments = _split_segments(text)

    extractors = [
        ("search_contacts", _extract_search_contacts),
        ("send_message", _extract_message),
        ("get_weather", _extract_weather),
        ("set_alarm", _extract_alarm),
        ("set_timer", _extract_timer),
        ("create_reminder", _extract_reminder),
        ("play_music", _extract_music),
    ]

    calls = []
    matched_segments = 0
    last_contact = None
    for segment in segments:
        for tool_name, extractor in extractors:
            if tool_name not in tool_map:
                continue
            args = extractor(segment)

            # Resolve simple pronouns after contact lookup (e.g., "find Tom ... and send him ..."). 
            if (
                tool_name == "send_message"
                and args is None
                and last_contact
                and re.search(r"\b(send|text|message)\b", segment, re.IGNORECASE)
                and re.search(r"\b(him|her|them)\b", segment, re.IGNORECASE)
            ):
                m_saying = re.search(r"\bsaying\s+(.+)$", segment, re.IGNORECASE)
                if m_saying:
                    args = {"recipient": last_contact, "message": m_saying.group(1).strip(" .")}

            if args and _required_ok(args, tool_map[tool_name]):
                calls.append({"name": tool_name, "arguments": args})
                matched_segments += 1
                if tool_name == "search_contacts":
                    last_contact = args.get("query")
                elif tool_name == "send_message":
                    last_contact = args.get("recipient")
                break

    confidence = 0.0
    if segments:
        confidence = min(1.0, (matched_segments / len(segments)) + (0.15 if calls else 0.0))

    return calls, confidence


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    cactus_api = _load_cactus()
    if not cactus_api:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    cactus_init, cactus_complete, cactus_destroy = cactus_api
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    genai, types = _load_genai()
    if genai is None or types is None:
        return {"function_calls": [], "total_time_ms": 0}
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Local-first hybrid strategy:
    1) Fast deterministic parser for common assistant intents (weather/alarm/message/reminder/search/music/timer)
    2) On-device FunctionGemma fallback if parser is uncertain
    3) Cloud fallback only when both local stages are low-confidence
    """
    start = time.time()
    heuristic_calls, heuristic_conf = _heuristic_route(messages, tools)
    heuristic_time_ms = (time.time() - start) * 1000

    if heuristic_calls and heuristic_conf >= 0.70:
        return {
            "function_calls": heuristic_calls,
            "total_time_ms": heuristic_time_ms,
            "confidence": heuristic_conf,
            "source": "on-device",
            "strategy": "heuristic-local-router",
        }

    local = generate_cactus(messages, tools)

    # If parser found any valid call, prefer it over weak/empty model output to keep latency low.
    if heuristic_calls and not local.get("function_calls"):
        return {
            "function_calls": heuristic_calls,
            "total_time_ms": heuristic_time_ms + local.get("total_time_ms", 0),
            "confidence": max(heuristic_conf, local.get("confidence", 0)),
            "source": "on-device",
            "strategy": "heuristic-plus-cactus",
        }

    if local.get("confidence", 0) >= confidence_threshold or local.get("function_calls"):
        local["source"] = "on-device"
        local["strategy"] = "cactus-local"
        local["total_time_ms"] = local.get("total_time_ms", 0) + heuristic_time_ms
        local["confidence"] = max(local.get("confidence", 0), heuristic_conf)
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["strategy"] = "cloud-after-local"
    cloud["local_confidence"] = max(local.get("confidence", 0), heuristic_conf)
    cloud["total_time_ms"] += local.get("total_time_ms", 0) + heuristic_time_ms
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "strategy" in result:
        print(f"Strategy: {result['strategy']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
