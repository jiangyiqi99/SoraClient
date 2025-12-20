from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mimetypes
import requests

DEFAULT_BASE_URL = "https://api.openai.com/v1/audio"
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"


@dataclass
class OpenAIAudioClient:
    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 60

    def _resolve_api_key(self) -> str:
        api_key = self.api_key or self._load_api_key()
        if not api_key:
            raise RuntimeError("API key missing. Pass api_key or save it in config/config.json.")
        return api_key

    def _load_api_key(self) -> Optional[str]:
        if not CONFIG_FILE.exists():
            return None
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        api_key = data.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
        return None

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._resolve_api_key()}"}

    def _parse_text_response(self, response: requests.Response) -> Tuple[str, Any]:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
            return str(data.get("text", "")), data
        text = response.text
        return text, text

    def _request_audio(
        self,
        endpoint: str,
        data: Dict[str, Any],
        file_path: Path,
    ) -> Dict[str, Any]:
        if not file_path:
            raise ValueError("audio file path is required.")
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_tuple = (file_path.name, file_path.open("rb"), mime_type or "application/octet-stream")
        files = {"file": file_tuple}
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                headers=self._headers(),
                data=data,
                files=files,
                timeout=self.timeout,
            )
        finally:
            file_tuple[1].close()
        if not response.ok:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_body}"
            )
        text, raw = self._parse_text_response(response)
        return {"text": text, "raw": raw}

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {"model": model}
        if language:
            data["language"] = language
        if response_format:
            data["response_format"] = response_format
        return self._request_audio("transcriptions", data, audio_path)

    def translate(
        self,
        audio_path: Path,
        model: str,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {"model": model}
        if response_format:
            data["response_format"] = response_format
        return self._request_audio("translations", data, audio_path)

    def speech(
        self,
        text: str,
        output_path: Path,
        model: str,
        voice: str,
        instructions: Optional[str] = None,
    ) -> Path:
        if not text:
            raise ValueError("input text is required.")
        if not output_path:
            raise ValueError("output_path is required.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"model": model, "voice": voice, "input": text}
        if instructions:
            payload["instructions"] = instructions
        response = requests.post(
            f"{self.base_url}/speech",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        if not response.ok:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_body}"
            )
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return output_path

    def speech_bytes(
        self,
        text: str,
        model: str,
        voice: str,
        instructions: Optional[str] = None,
    ) -> bytes:
        if not text:
            raise ValueError("input text is required.")
        payload: Dict[str, Any] = {"model": model, "voice": voice, "input": text}
        if instructions:
            payload["instructions"] = instructions
        response = requests.post(
            f"{self.base_url}/speech",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        if not response.ok:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_body}"
            )
        return response.content
