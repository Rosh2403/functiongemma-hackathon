import argparse
import json
from pathlib import Path

from product_demo import TOOLS, execute_call
from main import generate_hybrid


def transcribe_audio(audio_path: str, whisper_model_path: str = "weights/whisper-small") -> str:
    # Local import keeps script importable even when cactus python bindings are unavailable.
    from cactus import cactus_init, cactus_destroy, cactus_transcribe

    model = cactus_init(whisper_model_path)
    try:
        prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        raw = cactus_transcribe(model, audio_path, prompt=prompt)
        data = json.loads(raw)
        return (data.get("response") or "").strip()
    finally:
        cactus_destroy(model)


def run_voice_command(audio_path: str, whisper_model_path: str) -> dict:
    transcript = transcribe_audio(audio_path, whisper_model_path)
    messages = [{"role": "user", "content": transcript}]
    route = generate_hybrid(messages, TOOLS)
    executions = [execute_call(c) for c in route.get("function_calls", [])]
    return {
        "audio": str(Path(audio_path).resolve()),
        "transcript": transcript,
        "route": route,
        "executions": executions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-to-action pipeline with cactus_transcribe")
    parser.add_argument("--audio", required=True, help="Path to WAV audio file")
    parser.add_argument("--whisper", default="weights/whisper-small", help="Path to whisper model")
    args = parser.parse_args()

    output = run_voice_command(args.audio, args.whisper)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
