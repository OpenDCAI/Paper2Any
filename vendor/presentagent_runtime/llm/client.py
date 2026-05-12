from __future__ import annotations

import json
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        client_type: str = "llm",
    ) -> None:
        self._sdk_client = None
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            OpenAI = None

        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "").strip().rstrip("/")
        self.model = (model or "").strip()
        self.client_type = (client_type or "llm").strip()
        if OpenAI is not None:
            self._sdk_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=120.0,
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        response_format: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if self._sdk_client is not None:
            response = self._sdk_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        payload = self._post_json("/chat/completions", kwargs)
        return str(payload["choices"][0]["message"]["content"] or "")

    def chat_with_image(self, text: str, image_url: str) -> str:
        return self.chat_with_images(text, [image_url])

    def chat_with_images(self, text: str, image_urls: list[str]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image_url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        if self._sdk_client is not None:
            response = self._sdk_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
            return response.choices[0].message.content or ""
        payload = self._post_json(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
            },
        )
        return str(payload["choices"][0]["message"]["content"] or "")

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        if self._sdk_client is not None:
            response = self._sdk_client.embeddings.create(model=model, input=text)
            return list(response.data[0].embedding)
        payload = self._post_json("/embeddings", {"model": model, "input": text})
        return list(payload["data"][0]["embedding"])

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_base:
            raise RuntimeError("LLM client api_base is empty")

        url = f"{self.api_base}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib_request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(request, timeout=120.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"{self.client_type} request failed: {exc}") from exc
