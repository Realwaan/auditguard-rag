"""Optional venue-only API probe. Sends a synthetic greeting, never document content."""
import argparse
import json
import os
import time
from urllib.parse import urlsplit

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        parser.error("Supply the actual ALMA base URL; API compatibility must be confirmed by the organizers.")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        parser.error("Use a base URL without credentials, query parameters, or fragments.")
    if not args.model or "REPLACE_" in args.model or "REPLACE_" in args.base_url:
        parser.error("Supply the exact served model ID and ALMA base URL.")
    started = time.perf_counter()
    # No Authorization header, public provider fallback, redirects, or environment proxy.
    try:
        with httpx.Client(timeout=args.timeout, follow_redirects=False, trust_env=False) as client:
            response = client.post(
                args.base_url.rstrip("/") + "/chat/completions",
                json={"model": args.model, "messages": [{"role": "user", "content": "Reply with exactly READY."}], "max_tokens": 64, "temperature": 0},
            )
            if response.status_code != 200:
                print(json.dumps({"ok": False, "http_status": response.status_code, "note": "Confirm ALMA URL, API format, model ID and access requirements. No fallback was attempted."}))
                return 1
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty completion")
            print(json.dumps({"ok": True, "requested_model": args.model, "reported_model": data.get("model"), "response": text[:200], "latency_ms": round((time.perf_counter()-started)*1000), "note": "API smoke test only; this is not a RAG accuracy or vision test."}, indent=2))
            return 0
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "note": "No public API fallback was attempted."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
