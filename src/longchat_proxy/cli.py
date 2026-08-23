from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv

from . import __version__
from .config import ConfigurationError, Settings, openai_url
from .proxy import create_app


def _settings(env_file: str | None) -> Settings:
    if env_file:
        path = Path(env_file).expanduser()
        if not path.exists():
            raise ConfigurationError(f"Environment file not found: {path}")
        load_dotenv(path, override=False)
    else:
        load_dotenv(override=False)
    return Settings.from_env()


async def _doctor(settings: Settings) -> int:
    print("Configuration: valid")
    print(f"Listener:      http://{settings.host}:{settings.port}")
    print(f"Upstream:      {settings.upstream_base_url}")
    print(f"State:         {settings.state_mode}")
    print(f"Summary model: {settings.summary_model or '<same as request model>'}")
    headers = {}
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                openai_url(settings.upstream_base_url, "/v1/models"), headers=headers
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        print(f"Upstream:      FAILED ({error})", file=sys.stderr)
        return 1
    model_ids = [str(model.get("id")) for model in payload.get("data", [])]
    print(f"Upstream:      reachable ({len(model_ids)} models)")
    if settings.summary_model and settings.summary_model not in model_ids:
        print(
            f"Summary model: FAILED ({settings.summary_model!r} is not listed)",
            file=sys.stderr,
        )
        return 1
    print("Doctor result: healthy")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="longchat-proxy",
        description="Transparent rolling-context proxy for OpenAI-compatible LLM servers.",
    )
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument("--env-file", help="Load configuration from this .env file")
    subparsers = result.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Start the proxy (default)")
    subparsers.add_parser("doctor", help="Validate configuration and upstream connectivity")
    subparsers.add_parser(
        "print-config", help="Print effective configuration with secrets redacted"
    )
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        settings = _settings(args.env_file)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    command = args.command or "serve"
    if command == "doctor":
        raise SystemExit(asyncio.run(_doctor(settings)))
    if command == "print-config":
        print(json.dumps(settings.public_dict(), indent=2, sort_keys=True))
        return
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
