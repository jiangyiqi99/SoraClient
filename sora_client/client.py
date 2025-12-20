from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config import get_api_key

DEFAULT_BASE_URL = "https://api.openai.com/v1/videos"


@dataclass
class SoraClient:
    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 60

    def _resolve_api_key(self) -> str:
        api_key = self.api_key or get_api_key()
        if not api_key:
            raise RuntimeError(
                "API key missing. Set it in ./config/config.json or pass api_key."
            )
        return api_key

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        if not response.ok:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_body}"
            )
        return response.json()

    def create_video(
        self,
        prompt: str,
        model: str | None = None,
        seconds: int | None = None,
        size: str | None = None,
        input_reference: Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key = self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        fields: Dict[str, Any] = {"prompt": prompt}
        if model:
            fields["model"] = model
        if seconds is not None:
            if seconds not in {4, 8, 12}:
                raise ValueError("seconds must be one of 4, 8, or 12")
            fields["seconds"] = str(seconds)
        if size:
            fields["size"] = size
        for key, value in kwargs.items():
            fields[key] = str(value)
        files = {key: (None, value) for key, value in fields.items()}
        if input_reference:
            files["input_reference"] = (
                input_reference.name,
                input_reference.open("rb"),
                "application/octet-stream",
            )
        try:
            return self._request_json("POST", self.base_url, headers=headers, files=files)
        finally:
            ref = files.get("input_reference")
            if ref and hasattr(ref[1], "close"):
                ref[1].close()

    def retrieve_video(self, video_id: str) -> Dict[str, Any]:
        api_key = self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{self.base_url}/{video_id}"
        return self._request_json("GET", url, headers=headers)

    def remix_video(self, video_id: str, prompt: str) -> Dict[str, Any]:
        api_key = self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{self.base_url}/{video_id}/remix"
        payload = {"prompt": prompt}
        return self._request_json("POST", url, headers=headers, json=payload)

    def delete_video(self, video_id: str) -> Dict[str, Any]:
        api_key = self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{self.base_url}/{video_id}"
        return self._request_json("DELETE", url, headers=headers)

    def download_video_content(self, video_id: str, output_path: Path) -> Path:
        api_key = self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{self.base_url}/{video_id}/content"
        response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
        if not response.ok:
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_body}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return output_path

    def wait_for_completion(
        self,
        video_id: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        start = time.time()
        while True:
            job = self.retrieve_video(video_id)
            status = job.get("status")
            if status in {"completed", "failed", "canceled"}:
                return job
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for video {video_id}")
            time.sleep(poll_interval)
