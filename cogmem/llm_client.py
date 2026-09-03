"""
LLM client with multi-channel failover for OpenAI-compatible APIs.

Supports any OpenAI-compatible API endpoint (DeepSeek, Qwen/DashScope,
OpenAI, etc.) with automatic channel switching on failure.
"""

import json
import os
import re
import time
import requests
from typing import Optional


class LLMClient:
    """
    LLM API client with multi-channel failover.

    Configuration via environment variables or constructor params:

    .. code-block:: python

        client = LLMClient({
            "api_key": "sk-xxx",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        })
    """

    DEFAULT_CHANNEL = {
        "name": "default",
        "api_key": os.getenv("COGMEM_API_KEY", ""),
        "base_url": os.getenv("COGMEM_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("COGMEM_MODEL", "deepseek-chat"),
    }

    def __init__(self, channels: list[dict] | dict | None = None):
        if channels is None:
            self.channels = [dict(self.DEFAULT_CHANNEL)]
        elif isinstance(channels, dict):
            self.channels = [channels]
        else:
            self.channels = channels

        self._active_channel = 0
        self._channel_fail_time = {}
        self._CHANNEL_COOLDOWN = 60
        self._last_call = 0
        self._min_interval = 0.3
        self._token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }

    def reset_token_stats(self):
        self._token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }

    def get_token_stats(self) -> dict:
        return dict(self._token_stats)

    def _get_active_channel(self) -> dict:
        return self.channels[self._active_channel]

    def _try_next_channel(self, failed_idx: int):
        self._channel_fail_time[failed_idx] = time.time()
        for i in range(len(self.channels)):
            if i == failed_idx:
                continue
            fail_time = self._channel_fail_time.get(i, 0)
            if time.time() - fail_time < self._CHANNEL_COOLDOWN:
                continue
            ch = self.channels[i]
            if not ch.get("api_key"):
                continue
            self._active_channel = i
            return ch
        self._active_channel = 0
        return self.channels[0]

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def chat_completion(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> str:
        self._rate_limit()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        tried_channels = set()

        for attempt in range(max_retries):
            ch = self._get_active_channel()

            if not ch.get("api_key") or self._active_channel in tried_channels:
                self._try_next_channel(self._active_channel)
                ch = self._get_active_channel()
                if self._active_channel in tried_channels:
                    break

            tried_channels.add(self._active_channel)

            try:
                resp = requests.post(
                    f"{ch['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {ch['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": ch["model"],
                        "messages": messages,
                        "temperature": temperature,
                    },
                    timeout=120,
                )

                if resp.status_code in (401, 402, 403):
                    self._try_next_channel(self._active_channel)
                    continue

                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage", {})
                if usage:
                    self._token_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self._token_stats["completion_tokens"] += usage.get("completion_tokens", 0)
                    self._token_stats["total_tokens"] += usage.get("total_tokens", 0)
                self._token_stats["call_count"] += 1

                return data["choices"][0]["message"]["content"]

            except requests.exceptions.ConnectionError:
                self._try_next_channel(self._active_channel)
                continue
            except requests.exceptions.Timeout:
                self._try_next_channel(self._active_channel)
                continue
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    self._try_next_channel(self._active_channel)
                    continue
                raise

        for i, ch in enumerate(self.channels):
            if not ch.get("api_key"):
                continue
            try:
                resp = requests.post(
                    f"{ch['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {ch['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": ch["model"],
                        "messages": messages,
                        "temperature": temperature,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                if usage:
                    self._token_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self._token_stats["completion_tokens"] += usage.get("completion_tokens", 0)
                    self._token_stats["total_tokens"] += usage.get("total_tokens", 0)
                self._token_stats["call_count"] += 1
                self._active_channel = i
                return data["choices"][0]["message"]["content"]
            except Exception:
                continue

        raise RuntimeError("All LLM channels unavailable")

    def extract_json(self, text: str) -> Optional[dict]:
        if not text:
            return None
        try:
            return json.loads(text.strip())
        except Exception:
            pass
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return None

    def chat_completion_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> dict:
        for attempt in range(max_retries):
            text = self.chat_completion(system, user, temperature=temperature)
            result = self.extract_json(text)
            if result is not None:
                return result
            user = user + "\n\nImportant: Your response must be strictly valid JSON, without any other text, explanation, or code block markers."
        return {}
