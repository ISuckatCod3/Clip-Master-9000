from __future__ import annotations

import argparse

from live_video_interpreter import run_whisperlive_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WhisperLive for Clip Master voice commands.")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rest_port", type=int, default=8000)
    parser.add_argument("--backend", default="faster_whisper")
    parser.add_argument("--model", "-fw", default="base.en")
    parser.add_argument("--max_clients", type=int, default=2)
    parser.add_argument("--max_connection_time", type=int, default=43200)
    parser.add_argument("--cache_path", "-c", default="models/whisper-live-cache")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    run_whisperlive_server(
        host="0.0.0.0",
        port=args.port,
        rest_port=args.rest_port,
        backend=args.backend,
        model=args.model,
        max_clients=args.max_clients,
        max_connection_time=args.max_connection_time,
        cache_path=args.cache_path,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
