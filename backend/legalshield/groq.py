import asyncio
import json
import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
from fastapi import HTTPException

from .config import settings

logger = logging.getLogger(__name__)
_usage = ContextVar("laven_groq_usage", default=None)


class Groq:
    def __init__(self):
        config = settings()
        self.key = config.groq_api_key.get_secret_value()
        self.model = config.groq_model.strip()
        self.model_checked = 0.0
        self.client = httpx.AsyncClient(base_url="https://api.groq.com/openai/v1/",
                                       timeout=httpx.Timeout(48, connect=10),
                                       headers={"Authorization": f"Bearer {self.key}"})

    @contextmanager
    def track_usage(self):
        usage = {"completion_attempts": 0, "usage_responses": 0, "prompt_tokens": 0, "completion_tokens": 0}
        scope = _usage.set(usage)
        try:
            yield usage
        finally:
            _usage.reset(scope)

    async def request(self, method: str, path: str, **kwargs):
        for attempt in range(3):
            usage = _usage.get()
            if usage is not None and path == "chat/completions":
                usage["completion_attempts"] += 1
            try:
                response = await self.client.request(method, path, **kwargs)
            except httpx.RequestError:
                if attempt == 2:
                    raise HTTPException(503, "Groq could not be reached. Check network access and try again.")
                await asyncio.sleep(2 ** attempt)
                continue
            if response.is_success:
                return response.json()
            if response.status_code in (401, 403):
                raise HTTPException(503, "Groq rejected access. Verify GROQ_API_KEY and model permissions.")
            if response.status_code == 400:
                # The model produced valid JSON that isn't a JSON object (e.g. a bare array).
                # Groq's json_object validator rejects it, but the content is in failed_generation.
                try:
                    body = response.json()
                    raw = body.get("error", {}).get("failed_generation", "")
                    if raw:
                        parsed = json.loads(raw)
                        # Wrap a bare array under the key the caller expects ("findings", "claims", etc.)
                        if isinstance(parsed, list):
                            # Determine key from the request payload if available
                            req_body = kwargs.get("json", {})
                            msgs = req_body.get("messages", [])
                            system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
                            key = "findings" if "findings" in system else "claims" if "claims" in system else "items"
                            return {key: parsed}
                        if isinstance(parsed, dict):
                            return parsed
                except (ValueError, KeyError, TypeError):
                    pass
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 2:
                    raise HTTPException(503, "Groq is busy or rate limited. Try again shortly.")
                retry = response.headers.get("retry-after", "")
                await asyncio.sleep(min(float(retry), 12) if retry.replace(".", "", 1).isdigit() else 2 ** attempt)
                continue
            logger.error("Groq request failed: status=%s body=%s", response.status_code, response.text)
            raise HTTPException(502, f"Groq rejected the model request (HTTP {response.status_code}): {response.text[:200]}")

    async def verify(self):
        if not self.key or not self.model:
            raise HTTPException(503, "Set GROQ_API_KEY and GROQ_MODEL to your configured Groq Qwen model.")
        if time.monotonic() < self.model_checked:
            return self.model
        result = await self.request("GET", "models")
        if self.model not in {entry["id"] for entry in result.get("data", [])}:
            raise HTTPException(503, f"The configured model {self.model} is unavailable to this Groq account. Check the model list and update GROQ_MODEL.")
        if "qwen" not in self.model.lower():
            raise HTTPException(503, "GROQ_MODEL must identify a Qwen model available to your Groq account.")
        self.model_checked = time.monotonic() + 300
        return self.model

    async def json(self, instruction: str, data: dict, *, max_tokens: int = 2200, temperature: float = 0,
                   top_p: float | None = None) -> dict:
        await self.verify()
        result = await self.request("POST", "chat/completions", json={
            "model": self.model, "temperature": temperature, "max_completion_tokens": max_tokens,
            **({"top_p": top_p} if top_p is not None else {}),
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "Return a valid JSON object.\n\n" + instruction},
                         {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
        })
        usage = _usage.get()
        metadata = result.get("usage") if isinstance(result, dict) else None
        if usage is not None and isinstance(metadata, dict):
            fields = ("prompt_tokens", "completion_tokens")
            if all(type(metadata.get(key)) is int and metadata[key] >= 0 for key in fields):
                usage["usage_responses"] += 1
                for key in fields:
                    usage[key] += metadata[key]
        # If request() already recovered a dict from a bare-array response, return it directly.
        if isinstance(result, dict) and "choices" not in result:
            return result
        try:
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, str):
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            value = json.loads(content)
            if isinstance(value, dict):
                return value
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        raise HTTPException(502, "Groq returned incomplete or malformed structured output. Retry the operation.")
