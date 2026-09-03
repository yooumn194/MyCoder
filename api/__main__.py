"""Console entry point for the optional MyCoder API service."""

from __future__ import annotations

import argparse
import os


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MyCoder FastAPI service")
    parser.add_argument("--host", default=os.getenv("MYCODER_API_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("MYCODER_API_PORT", "8000"))
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.getenv("MYCODER_API_WORKERS", "1"))
    )
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.reload and args.workers != 1:
        raise SystemExit("--reload requires --workers 1")

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "API dependencies are missing; install with: pip install 'mycoder[api]'"
        ) from exc
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
