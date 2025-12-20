# Sora-2 Videos API Client

Minimal Python client for the OpenAI Videos API (`https://api.openai.com/v1/videos`).

## Setup

```bash
pip install -r requirements.txt
```

## Save API key locally

```bash
python -m sora_client --set-key "YOUR_API_KEY"
```

This writes `./config/config.json` with an `api_key` field.

## Use in code

```python
from pathlib import Path
from sora_client import SoraClient

client = SoraClient()
job = client.create_video(
    prompt="A calico cat playing a piano on stage",
    model="sora-2",
    seconds=4,
    size="1280x720",
)
video_id = job["id"]
final = client.wait_for_completion(video_id)
if final.get("status") == "completed":
    client.download_video_content(video_id, Path("./output/cat.mp4"))
```

## CLI usage

Create a job and poll until completion, then download:

```bash
python -m sora_client \
  --prompt "A calico cat playing a piano on stage" \
  --model "sora-2" \
  --seconds 4 \
  --size "1280x720" \
  --poll \
  --output "./output/cat.mp4"
```

Retrieve an existing job:

```bash
python -m sora_client --video-id "video_123" --poll
```

## Web UI

```bash
python main.py
```

The UI saves each job response under `./jobs` with a timestamped filename and lets you select a saved job for retrieve/remix/delete.
