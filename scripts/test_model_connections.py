"""
Test connectivity to all 7 models in the 3-level ABC cascade.

Usage
-----
# 1. Fill in .env (see .env.example for the required keys)
# 2. Run:
python scripts/test_model_connections.py

Each model receives a single short message. The script prints a pass/fail
table and exits with code 1 if any model fails.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ── Model registry ──────────────────────────────────────────────────────────────

MODELS = [
    # (level, label, provider, model_name, extra_kwargs)
    (1, "DeepSeek-V3.2-Speciale",      "azure",  "DeepSeek-V3.2-Speciale",       {}),
    (1, "Gemini 2.5 Flash Lite",        "vertex", "gemini-2.5-flash-lite",         {"location": "global"}),
    (1, "Mistral-Large-3",              "azure",  "mistral-large-3",               {}),
    (2, "Gemini 2.5 Flash",             "vertex", "gemini-2.5-flash",              {}),
    (2, "kimi-k2.5",                    "azure",  "kimi-k2.5",                     {}),
    (2, "Haiku 4.5",                    "claude_vertex", "claude-haiku-4-5@20251001",  {}),
    (3, "Sonnet 4.6  [escalation]",     "claude_vertex", "claude-sonnet-4-6@default",  {}),
]

PING = "Respond with exactly one word: OK"

# Candidate API versions tried in order when AZURE_FOUNDRY_API_VERSION is not set
_AZURE_VERSION_CANDIDATES = [
    "2025-04-01-preview",
    "2025-03-01-preview",
    "2025-02-01-preview",
    "2025-01-01-preview",
    "2024-12-01-preview",
    "2024-10-21",
    "2024-05-01-preview",
]


def _make_azure_client(api_version: str):
    from openai import OpenAI
    endpoint = os.environ["AZURE_FOUNDRY_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_FOUNDRY_API_KEY"]
    # Project-scoped endpoint (services.ai.azure.com/api/projects/...):
    # chat completions live at {endpoint}/models/chat/completions
    return OpenAI(
        base_url=f"{endpoint}/models",
        api_key=api_key,
        default_query={"api-version": api_version},
        default_headers={"api-key": api_key},
    )


def _probe_azure_version() -> str:
    """Try each candidate version on a harmless request; return first that isn't 'version not supported'."""
    if v := os.environ.get("AZURE_FOUNDRY_API_VERSION"):
        return v
    endpoint = os.environ["AZURE_FOUNDRY_ENDPOINT"].rstrip("/")
    probe_model = next(mn for _, _, p, mn, _ in MODELS if p == "azure")
    print(f"  Probing Azure endpoint: {endpoint}/chat/completions")
    for v in _AZURE_VERSION_CANDIDATES:
        try:
            client = _make_azure_client(v)
            client.chat.completions.create(
                model=probe_model,
                messages=[{"role": "user", "content": PING}],
                max_tokens=4,
            )
            return v
        except Exception as exc:
            msg = str(exc)
            if "API version not supported" in msg or "InvalidApiVersionParameter" in msg:
                print(f"    {v}: not supported")
                continue
            print(f"    {v}: OK (different error: {msg[:80]})")
            return v
    print("  WARNING: no working API version found — check AZURE_FOUNDRY_ENDPOINT in .env")
    return _AZURE_VERSION_CANDIDATES[-1]


# ── Provider builders ───────────────────────────────────────────────────────────

_azure_version: str | None = None  # cached after first probe


def _azure_call(model_name: str) -> tuple[bool, str]:
    global _azure_version
    if _azure_version is None:
        _azure_version = _probe_azure_version()
    client = _make_azure_client(_azure_version)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": PING}],
        max_tokens=512,
    )
    msg = resp.choices[0].message if resp.choices else None
    if msg is None or msg.content is None:
        finish = resp.choices[0].finish_reason if resp.choices else "no choices"
        return False, f"None content — finish_reason={finish}"
    return True, msg.content.strip()


def _claude_vertex_call(model_name: str) -> tuple[bool, str]:
    import anthropic  # anthropic 0.46.0 supports AnthropicVertex
    project = os.environ["VERTEX_PROJECT"]
    location = os.environ.get("CLAUDE_VERTEX_LOCATION", "us-east5")
    client = anthropic.AnthropicVertex(project_id=project, region=location)
    try:
        msg = client.messages.create(
            model=model_name,
            max_tokens=512,
            messages=[{"role": "user", "content": PING}],
        )
        text = msg.content[0].text if msg.content else None
        if not text:
            return False, f"empty response: stop_reason={msg.stop_reason}"
        return True, text.strip()
    except anthropic.RateLimitError:
        return True, "(rate limited — model reachable)"


def _vertex_call(model_name: str, location: str | None = None) -> tuple[bool, str]:
    from google import genai  # google-genai is in requirements (1.56.0)
    project = os.environ["VERTEX_PROJECT"]
    loc = location or os.environ.get("VERTEX_LOCATION", "us-central1")
    client = genai.Client(vertexai=True, project=project, location=loc)
    resp = client.models.generate_content(model=model_name, contents=PING)
    text = resp.text
    if not text:
        return False, f"empty response: {resp}"
    return True, text.strip()


# ── Runner ──────────────────────────────────────────────────────────────────────

def main() -> int:
    results = []

    # Probe Azure version once up front so it shows before the first model line
    if any(p == "azure" for _, _, p, _, _ in MODELS):
        global _azure_version
        _azure_version = _probe_azure_version()
        src = "env" if os.environ.get("AZURE_FOUNDRY_API_VERSION") else "probed"
        print(f"  Azure API version: {_azure_version} ({src})")
        print()

    for level, label, provider, model_name, extra in MODELS:
        print(f"  L{level}  {label:<35}", end="", flush=True)
        t0 = time.monotonic()
        try:
            if provider == "azure":
                ok, reply = _azure_call(model_name)
            elif provider == "claude_vertex":
                ok, reply = _claude_vertex_call(model_name)
            else:
                ok, reply = _vertex_call(model_name, **extra)
        except KeyError as exc:
            ok, reply = False, f"missing env var {exc}"
        except Exception as exc:
            ok, reply = False, str(exc)
        elapsed = (time.monotonic() - t0) * 1000

        status = "PASS" if ok else "FAIL"
        print(f"{status}  ({elapsed:.0f} ms)  {reply}")
        results.append((label, ok, reply))

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} models reachable")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
