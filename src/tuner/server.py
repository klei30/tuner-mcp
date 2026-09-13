from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from tuner.mcp_server import create_server
from tuner.settings import Settings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the Tuner MCP server")
    result.add_argument("--transport", choices=("stdio", "http"), default=None)
    result.add_argument("--host", default=None)
    result.add_argument("--port", type=int, default=None)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    transport = args.transport or settings.transport
    if transport == "http" and (not settings.auth_token or len(settings.auth_token) < 32):
        raise SystemExit("HTTP transport requires TUNER_AUTH_TOKEN with at least 32 characters")
    server = create_server(settings)
    if transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(
            transport="http",
            host=args.host or settings.host,
            port=args.port or settings.port,
            show_banner=False,
        )


if __name__ == "__main__":
    main()
