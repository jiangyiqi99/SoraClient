from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import SoraClient
from .config import set_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Sora-2 Videos API client")
    parser.add_argument("--set-key", dest="set_key", help="Save API key to ./config")

    parser.add_argument("--prompt", help="Prompt for video generation")
    parser.add_argument("--model", help="Model name (sora-2 or sora-2-pro)")
    parser.add_argument("--seconds", type=int, help="Duration in seconds (4/8/12)")
    parser.add_argument("--size", help="Frame size, e.g. 1280x720")
    parser.add_argument(
        "--extra",
        help="Extra JSON fields to merge into the request payload",
    )

    parser.add_argument("--video-id", help="Retrieve a video job by id")
    parser.add_argument("--poll", action="store_true", help="Poll until completed")
    parser.add_argument("--timeout", type=float, default=600.0, help="Poll timeout")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval")
    parser.add_argument("--output", help="Path to save MP4 when completed")

    args = parser.parse_args()

    if args.set_key:
        set_api_key(args.set_key)
        print("API key saved to ./config/config.json")
        return

    client = SoraClient()

    if args.video_id:
        job = args.video_id
    elif args.prompt:
        extra = {}
        if args.extra:
            extra = json.loads(args.extra)
        result = client.create_video(
            prompt=args.prompt,
            model=args.model,
            seconds=args.seconds,
            size=args.size,
            **extra,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        job = result.get("id")
    else:
        parser.error("Either --prompt or --video-id is required unless --set-key is used")

    if not job:
        return

    if args.poll:
        final = client.wait_for_completion(
            job,
            poll_interval=args.interval,
            timeout=args.timeout,
        )
        print(json.dumps(final, indent=2, ensure_ascii=True))
        status = final.get("status")
        if args.output and status == "completed":
            path = client.download_video_content(job, Path(args.output))
            print(f"Saved video to {path}")
    elif args.output:
        path = client.download_video_content(job, Path(args.output))
        print(f"Saved video to {path}")


if __name__ == "__main__":
    main()
