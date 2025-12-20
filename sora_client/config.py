from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(data: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def get_api_key() -> str | None:
    config = load_config()
    api_key = config.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return None


def set_api_key(api_key: str) -> None:
    data = load_config()
    data["api_key"] = api_key
    save_config(data)
