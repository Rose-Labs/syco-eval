#!/usr/bin/env python3
"""Check whether configured benchmark models are callable.

The script reads PATIENT_MODEL, EVALUATOR_MODELS, and DOCTOR_MODELS from
src/main/sycophancy.py, sends a minimal chat request to each unique model, and
prints a status table. x-ai/... models are checked directly against xAI;
openai/gpt-oss-20b is checked directly against OpenAI as gpt-oss-20b; all
other models are checked against OpenRouter. Judge/evaluator models are
included explicitly.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_SOURCE = Path("src/main/sycophancy.py")


def read_model_config(source: Path) -> dict[str, Any]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in {"PATIENT_MODEL", "EVALUATOR_MODELS", "DOCTOR_MODELS"}:
                config[target.id] = ast.literal_eval(node.value)

    missing = {"PATIENT_MODEL", "EVALUATOR_MODELS", "DOCTOR_MODELS"} - set(config)
    if missing:
        raise ValueError(f"Missing model config in {source}: {sorted(missing)}")
    return config


def model_roles(config: dict[str, Any]) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}

    roles.setdefault(config["PATIENT_MODEL"], set()).add("patient")
    for model in config["EVALUATOR_MODELS"]:
        roles.setdefault(model, set()).add("judge")
    for model in config["DOCTOR_MODELS"]:
        roles.setdefault(model, set()).add("doctor")

    return roles


def is_xai_model(model: str) -> bool:
    return model.startswith("x-ai/")


def xai_model_name(model: str) -> str:
    return model.split("/", 1)[1] if is_xai_model(model) else model


def is_direct_openai_model(model: str) -> bool:
    return model in {"openai/gpt-oss-20b", "openai/gpt-oss-20b:free", "gpt-oss-20b"}


def openai_model_name(model: str) -> str:
    if model in {"openai/gpt-oss-20b", "openai/gpt-oss-20b:free"}:
        return "gpt-oss-20b"
    return model


def extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def check_model(
    openrouter_api_key: str | None,
    xai_api_key: str | None,
    openai_api_key: str | None,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    if is_xai_model(model):
        provider = "xAI"
        api_key = xai_api_key
        url = XAI_URL
        request_model = xai_model_name(model)
    elif is_direct_openai_model(model):
        provider = "OpenAI"
        api_key = openai_api_key
        url = OPENAI_URL
        request_model = openai_model_name(model)
    else:
        provider = "OpenRouter"
        api_key = openrouter_api_key
        url = OPENROUTER_URL
        request_model = model

    if not api_key:
        if provider == "xAI":
            key_name = "XAI_API_KEY or X_AI_API_KEY"
        elif provider == "OpenAI":
            key_name = "OPENAI_API_KEY or OPEN_AI_API_KEY"
        else:
            key_name = "OPENROUTER_API_KEY"
        return {
            "model": model,
            "provider": provider,
            "callable": False,
            "status_code": None,
            "latency_seconds": 0,
            "error": f"{key_name} is missing",
        }

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://medsycobench.com",
                "X-Title": "MedSycoBench model check",
            },
            json={
                "model": request_model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with the exact word OK. Do not explain.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 256,
            },
            timeout=timeout,
        )
        elapsed = round(time.monotonic() - started, 2)

        if not response.ok:
            return {
                "model": model,
                "provider": provider,
                "callable": False,
                "status_code": response.status_code,
                "latency_seconds": elapsed,
                "error": response.text[:1000],
            }

        data = response.json()
        content = extract_content(data)
        return {
            "model": model,
            "provider": provider,
            "callable": bool(content),
            "status_code": response.status_code,
            "latency_seconds": elapsed,
            "response": content[:200],
            "error": "" if content else f"Empty model response: {data}",
        }
    except Exception as exc:
        return {
            "model": model,
            "provider": provider,
            "callable": False,
            "status_code": None,
            "latency_seconds": round(time.monotonic() - started, 2),
            "error": str(exc),
        }


def print_table(results: list[dict[str, Any]]) -> None:
    print()
    print(f"{'CALLABLE':8s} {'STATUS':6s} {'LATENCY':8s} {'PROVIDER':10s} {'ROLES':20s} MODEL")
    print("-" * 112)
    for result in results:
        callable_text = "yes" if result["callable"] else "no"
        status = str(result.get("status_code") or "-")
        latency = f"{result.get('latency_seconds', 0):.2f}s"
        provider = result.get("provider", "-")
        roles = ",".join(result["roles"])
        print(f"{callable_text:8s} {status:6s} {latency:8s} {provider:10s} {roles:20s} {result['model']}")
        if not result["callable"]:
            print(f"  error: {result.get('error', '')[:300]}")
    print("-" * 112)
    print(
        f"Callable: {sum(1 for result in results if result['callable'])}/"
        f"{len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check provider callability for sycophancy benchmark models."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Python file containing PATIENT_MODEL, EVALUATOR_MODELS, DOCTOR_MODELS.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write detailed check results as JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds per model.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay between model checks to reduce rate-limit pressure.",
    )
    args = parser.parse_args()

    load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    xai_api_key = os.getenv("XAI_API_KEY") or os.getenv("X_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")

    config = read_model_config(args.source)
    roles_by_model = model_roles(config)

    print(f"Checking {len(roles_by_model)} unique models from {args.source}")
    results: list[dict[str, Any]] = []
    for model in sorted(roles_by_model):
        roles = sorted(roles_by_model[model])
        if is_xai_model(model):
            provider = "xAI"
        elif is_direct_openai_model(model):
            provider = "OpenAI"
        else:
            provider = "OpenRouter"
        print(f"Checking {model} via {provider} ({', '.join(roles)})...")
        result = check_model(
            openrouter_api_key,
            xai_api_key,
            openai_api_key,
            model,
            args.timeout,
        )
        result["roles"] = roles
        results.append(result)
        time.sleep(args.sleep)

    print_table(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Detailed results written to {args.output_json}")


if __name__ == "__main__":
    main()

