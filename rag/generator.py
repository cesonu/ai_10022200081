# Author: Comfort Chiadi Esonu | Index: 10022200081

"""LLM generation layer with Groq primary and provider fallbacks."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from groq import Groq
import anthropic
from openai import OpenAI


class Generator:
    """Generate grounded answers with retry and provider fallback."""

    def __init__(self, log_path: str = "logs/generator_log.jsonl") -> None:
        """Initialize clients using environment variables."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.log_path = Path(log_path)
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

    def _append_log(self, payload: dict) -> None:
        """Append generation metadata to JSONL log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def generate(self, prompt: str) -> str:
        """Generate text using Groq first, then Anthropic and OpenAI."""
        if self.groq_client:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            output = response.choices[0].message.content or ""
            usage = response.usage
            self._append_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": "groq",
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                }
            )
            return output

        if self.anthropic_client:
            for attempt in range(2):
                try:
                    response = self.anthropic_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=700,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    output = response.content[0].text if response.content else ""
                    usage = getattr(response, "usage", None)
                    self._append_log(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "provider": "anthropic",
                            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                        }
                    )
                    return output
                except anthropic.RateLimitError:
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    raise
                except Exception:
                    raise

        if self.openai_client:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            output = response.choices[0].message.content or ""
            usage = response.usage
            self._append_log(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": "openai",
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                }
            )
            return output

        raise RuntimeError("No API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")
