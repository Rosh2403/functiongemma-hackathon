"""
FunctionGemma Hybrid Routing — 4-Agent Architecture (No Regex)
===============================================================
Agent 1 — Complexity Assessor  : Predictive routing signals (Stage 1)
Agent 2 — Local Executor        : FunctionGemma only (Stage 2)
Agent 3 — Cloud Executor        : Anthropic Claude / Gemini fallback
Agent 4 — Orchestrator          : Coordinates agents, makes final routing decision

No regex used anywhere — all extraction done by FunctionGemma.
Complexity assessed via plain word counting and string splitting only.
"""

import sys
import json
import os
import time
from typing import Dict, List, Tuple

# Paths — set via environment variables
_cactus_src        = os.environ.get("CACTUS_SRC",     "/Users/bviveka/cactus/python/src")
functiongemma_path = os.environ.get("CACTUS_WEIGHTS", "/Users/bviveka/cactus/weights/functiongemma-270m-it")
sys.path.insert(0, _cactus_src)

# ---------------------------------------------------------------------------
# Lazy-loaded clients
# ---------------------------------------------------------------------------
_CACTUS_API       = None
_CACTUS_MODEL     = None
_GENAI_API        = None
_GENAI_TYPES      = None
_ANTHROPIC_CLIENT = None


def _load_cactus():
    global _CACTUS_API
    if _CACTUS_API is None:
        try:
            from cactus import cactus_init, cactus_complete, cactus_destroy
            _CACTUS_API = (cactus_init, cactus_complete, cactus_destroy)
        except Exception:
            _CACTUS_API = ()
    return _CACTUS_API


def _get_cactus_model():
    """Return cached FunctionGemma model — init once, reuse across all calls."""
    global _CACTUS_MODEL
    cactus_api = _load_cactus()
    if not cactus_api:
        return None
    if _CACTUS_MODEL is None:
        cactus_init, _, _ = cactus_api
        _CACTUS_MODEL = cactus_init(functiongemma_path)
    return _CACTUS_MODEL


def _load_genai():
    global _GENAI_API, _GENAI_TYPES
    if _GENAI_API is None:
        try:
            from google import genai
            from google.genai import types
            _GENAI_API   = genai
            _GENAI_TYPES = types
        except Exception:
            _GENAI_API   = None
            _GENAI_TYPES = None
    return _GENAI_API, _GENAI_TYPES


def _load_anthropic():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        try:
            import anthropic
            _ANTHROPIC_CLIENT = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except Exception:
            _ANTHROPIC_CLIENT = None
    return _ANTHROPIC_CLIENT


# ---------------------------------------------------------------------------
# Shared helpers — no regex, plain string/word counting only
# ---------------------------------------------------------------------------

def _latest_user_text(messages: List[Dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _token_count(text: str) -> int:
    return len(text.split())


def _count_any(text: str, words: List[str]) -> int:
    tokens = text.lower().split()
    return sum(tokens.count(w.lower()) for w in words)


def _estimate_segment_count(text: str) -> int:
    """Estimate intent count using plain string matching only."""
    count = 1
    for sep in [" and ", " then ", " also ", " plus ", ", "]:
        count += text.lower().count(sep)
    return min(count, 6)


def _required_ok(args: Dict, tool: Dict) -> bool:
    required = tool.get("parameters", {}).get("required", [])
    return all(k in args and args[k] not in (None, "") for k in required)


# ---------------------------------------------------------------------------
# Performance tracking for dynamic thresholds
# ---------------------------------------------------------------------------
_perf_window: List[int] = []
_PERF_WINDOW_SIZE = 10


def _record_local_result(success: bool) -> None:
    _perf_window.append(1 if success else 0)
    if len(_perf_window) > _PERF_WINDOW_SIZE:
        _perf_window.pop(0)


def _get_dynamic_thresholds(assessment: Dict) -> Tuple[float, float]:
    """
    Compute routing thresholds dynamically.
    Stage 1 (skip-local): 0.80 – 0.97
    Stage 2 (trust-local): 0.15 – 0.55
    """
    success_rate = (sum(_perf_window) / len(_perf_window)) if len(_perf_window) >= 3 else 0.6
    complexity   = assessment.get("semantic_complexity", 0.0)

    stage1 = max(0.80, min(0.97, round(0.80 + (success_rate * 0.17), 3)))
    stage2 = max(0.15, min(0.55, round(0.55 - (success_rate * 0.35) + (complexity * 0.15), 3)))

    return stage1, stage2


# ---------------------------------------------------------------------------
# AGENT 1 — Complexity Assessor (no regex)
# ---------------------------------------------------------------------------

_AMBIGUOUS_WORDS   = ["maybe", "probably", "around", "somewhere", "something", "sort", "kind"]
_CONDITIONAL_WORDS = ["then", "after", "before", "if", "unless", "when", "while"]
_PRONOUN_WORDS     = ["him", "her", "them", "it", "that", "there", "his", "hers"]


def agent_complexity_assessor(messages: List[Dict], tools: List[Dict]) -> Dict:
    text          = _latest_user_text(messages)
    token_count   = _token_count(text)
    segment_count = _estimate_segment_count(text)

    pronoun_refs  = _count_any(text, _PRONOUN_WORDS)
    cross_step    = _count_any(text, _CONDITIONAL_WORDS)
    ambiguous     = _count_any(text, _AMBIGUOUS_WORDS)
    multi_intent  = max(0, segment_count - 1)
    length_factor = max(0, token_count - 12) / 24.0

    semantic_complexity = min(1.0, sum([
        min(1.0, multi_intent  * 0.22),
        min(1.0, pronoun_refs  * 0.15),
        min(1.0, cross_step    * 0.20),
        min(1.0, length_factor * 0.35),
        min(1.0, ambiguous     * 0.18),
    ]))

    estimated_output_tokens = max(20, segment_count * 30)
    token_speed_score       = min(1.0, (estimated_output_tokens / 50) * 1000 / 3000)
    tool_count_score        = min(1.0, len(tools) / 10.0)

    stage1_difficulty = (
        token_speed_score   * 0.15 +
        tool_count_score    * 0.15 +
        semantic_complexity * 0.40 +
        min(1.0, (segment_count - 1) * 0.25) * 0.30
    )

    return {
        "stage1_difficulty":   round(stage1_difficulty, 4),
        "semantic_complexity": round(semantic_complexity, 4),
        "token_speed_score":   round(token_speed_score, 4),
        "tool_count_score":    round(tool_count_score, 4),
        "token_count":         token_count,
        "segment_count":       segment_count,
        "pronoun_refs":        pronoun_refs,
        "cross_step":          cross_step,
        "ambiguous":           ambiguous,
    }


# ---------------------------------------------------------------------------
# AGENT 2 — Local Executor (FunctionGemma only, no regex)
# ---------------------------------------------------------------------------

def agent_local_executor(messages: List[Dict], tools: List[Dict], assessment: Dict) -> Dict:
    """
    FunctionGemma handles all tool selection and argument extraction.
    No regex involved — model does everything natively.
    Model is cached — loaded once, reused across all calls.
    """
    segment_count       = assessment.get("segment_count", 1)
    semantic_complexity = assessment.get("semantic_complexity", 0.0)

    # Adaptive token budget — large enough to avoid JSON truncation
    if segment_count == 1 and semantic_complexity < 0.10:
        max_tokens = 300
    elif segment_count <= 2 and semantic_complexity < 0.30:
        max_tokens = 400
    else:
        max_tokens = 512

    model = _get_cactus_model()
    if model is None:
        return {
            "available":         False,
            "function_calls":    [],
            "model_confidence":  0.0,
            "stage2_confidence": 0.0,
            "total_time_ms":     0,
            "validation":        {},
            "source":            "unavailable",
        }

    _, cactus_complete, _ = _load_cactus()
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": (
            "You are a helpful assistant that uses tools to complete user requests. "
            "Always call the appropriate tool(s) with correct arguments. "
            "If multiple tools are needed, call all of them."
        )}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        raw = {}

    model_calls = raw.get("function_calls", [])
    model_conf  = raw.get("confidence", 0.0)
    total_time  = raw.get("total_time_ms", 0.0)

    # Validate model output
    valid_tool_names = {t["name"] for t in tools}
    validation  = {}
    valid_count = 0
    for call in model_calls:
        name        = call.get("name", "")
        args        = call.get("arguments", {})
        tool_match  = next((t for t in tools if t["name"] == name), None)
        required_ok = _required_ok(args, tool_match) if tool_match else False
        validation[name] = {
            "valid_name":         name in valid_tool_names,
            "required_fields_ok": required_ok,
        }
        if name in valid_tool_names and required_ok:
            valid_count += 1

    got_calls_score    = 1.0 if model_calls else 0.0
    output_valid_score = (valid_count / len(model_calls)) if model_calls else 0.0

    stage2_confidence = (
        model_conf         * 0.50 +
        output_valid_score * 0.30 +
        got_calls_score    * 0.20
    )

    return {
        "available":         True,
        "function_calls":    model_calls,
        "model_confidence":  round(model_conf, 4),
        "stage2_confidence": round(stage2_confidence, 4),
        "total_time_ms":     total_time,
        "validation":        validation,
        "source":            "functiongemma",
    }


# ---------------------------------------------------------------------------
# AGENT 3 — Cloud Executor
# ---------------------------------------------------------------------------

def agent_cloud_executor(messages: List[Dict], tools: List[Dict]) -> Dict:
    start = time.time()

    anthropic_client = _load_anthropic()
    if anthropic_client and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            claude_tools = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
            response = anthropic_client.messages.create(
                model=os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022"),
                max_tokens=512,
                tools=claude_tools,
                messages=messages,
            )
            function_calls = [
                {"name": block.name, "arguments": dict(block.input)}
                for block in response.content if block.type == "tool_use"
            ]
            return {
                "function_calls": function_calls,
                "total_time_ms":  (time.time() - start) * 1000,
                "provider":       "anthropic-claude",
            }
        except Exception:
            pass

    genai, types = _load_genai()
    if genai is None:
        return {"function_calls": [], "total_time_ms": 0, "provider": "none"}

    try:
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
        gemini_response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )
        function_calls = [
            {"name": part.function_call.name, "arguments": dict(part.function_call.args)}
            for candidate in gemini_response.candidates
            for part in candidate.content.parts
            if part.function_call
        ]
        return {
            "function_calls": function_calls,
            "total_time_ms":  (time.time() - start) * 1000,
            "provider":       "gemini",
        }
    except Exception as e:
        return {
            "function_calls": [],
            "total_time_ms":  (time.time() - start) * 1000,
            "provider":       "error",
            "error":          str(e),
        }


# ---------------------------------------------------------------------------
# AGENT 4 — Orchestrator
# ---------------------------------------------------------------------------

def agent_orchestrator(messages: List[Dict], tools: List[Dict]) -> Dict:
    """
    Routing:
      1. segment_count >= 3 AND complexity > 0.35  → cloud directly
      2. stage1_difficulty > dynamic threshold      → cloud directly
      3. Run FunctionGemma locally
      4. stage2_confidence >= dynamic threshold     → on-device
      5. Otherwise                                  → cloud fallback
    """
    t_start = time.time()

    assessment = agent_complexity_assessor(messages, tools)
    stage1_threshold, stage2_threshold = _get_dynamic_thresholds(assessment)

    segment_count      = assessment["segment_count"]
    semantic_complexity = assessment["semantic_complexity"]
    stage1_difficulty  = assessment["stage1_difficulty"]

    # Rule 1: multi-segment complex → cloud
    if segment_count >= 3 and semantic_complexity > 0.35:
        _record_local_result(False)
        cloud = agent_cloud_executor(messages, tools)
        return {
            "function_calls":    cloud["function_calls"],
            "total_time_ms":     (time.time() - t_start) * 1000,
            "source":            "cloud-direct",
            "strategy":          "high-complexity-multi-step",
            "provider":          cloud.get("provider"),
            "stage1_difficulty": stage1_difficulty,
            "stage2_confidence": None,
            "assessment":        assessment,
        }

    # Rule 2: overall difficulty too high → cloud
    if stage1_difficulty > stage1_threshold:
        _record_local_result(False)
        cloud = agent_cloud_executor(messages, tools)
        return {
            "function_calls":    cloud["function_calls"],
            "total_time_ms":     (time.time() - t_start) * 1000,
            "source":            "cloud-direct",
            "strategy":          "stage1-difficulty-exceeded",
            "provider":          cloud.get("provider"),
            "stage1_difficulty": stage1_difficulty,
            "stage2_confidence": None,
            "assessment":        assessment,
        }

    # Run FunctionGemma locally
    local = agent_local_executor(messages, tools, assessment)

    if local["stage2_confidence"] >= stage2_threshold:
        _record_local_result(True)
        return {
            "function_calls":    local["function_calls"],
            "total_time_ms":     (time.time() - t_start) * 1000,
            "source":            "on-device",
            "strategy":          "functiongemma-confident",
            "stage1_difficulty": stage1_difficulty,
            "stage2_confidence": local["stage2_confidence"],
            "model_confidence":  local.get("model_confidence"),
            "assessment":        assessment,
        }

    # Cloud fallback
    _record_local_result(False)
    cloud = agent_cloud_executor(messages, tools)
    return {
        "function_calls":    cloud["function_calls"],
        "total_time_ms":     (time.time() - t_start) * 1000,
        "source":            "cloud-fallback",
        "strategy":          "functiongemma-low-confidence",
        "provider":          cloud.get("provider"),
        "stage1_difficulty": stage1_difficulty,
        "stage2_confidence": local["stage2_confidence"],
        "model_confidence":  local.get("model_confidence"),
        "assessment":        assessment,
    }


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------

def generate_cactus(messages: List[Dict], tools: List[Dict]) -> Dict:
    assessment = agent_complexity_assessor(messages, tools)
    return agent_local_executor(messages, tools, assessment)


def generate_cloud(messages: List[Dict], tools: List[Dict]) -> Dict:
    return agent_cloud_executor(messages, tools)


def generate_hybrid(messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
    return agent_orchestrator(messages, tools)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_result(label: str, result: Dict) -> None:
    print(f"\n=== {label} ===\n")
    for key in ("source", "strategy", "provider"):
        if key in result:
            print(f"{key.capitalize()}: {result[key]}")
    if result.get("stage1_difficulty") is not None:
        print(f"Stage1 difficulty:  {result['stage1_difficulty']:.4f}")
    if result.get("stage2_confidence") is not None:
        print(f"Stage2 confidence:  {result['stage2_confidence']:.4f}")
    if result.get("model_confidence") is not None:
        print(f"Model confidence:   {result['model_confidence']:.4f}")
    print(f"Total time: {result.get('total_time_ms', 0):.2f}ms")
    for call in result.get("function_calls", []):
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Cloud (Anthropic / Gemini)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (Orchestrated)", hybrid)