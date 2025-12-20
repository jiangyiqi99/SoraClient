from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr

from openai_audio_client import OpenAIAudioClient
from sora_client import SoraClient
from sora_client.config import set_api_key


PROJECT_ROOT = Path(__file__).resolve().parent
JOBS_DIR = PROJECT_ROOT / "jobs"

BASE_SIZES = [
    "1280x720",
    "1920x1080",
    "720x1280",
    "1080x1920",
    "1024x1024",
]
PRO_SIZES = ["1024x1792", "1792x1024"]
TRANSCRIPTION_MODELS = [
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
]
TRANSLATION_MODELS = ["whisper-1"]
TTS_MODELS = ["gpt-4o-mini-tts"]


def _size_choices(model_name: str) -> list[str]:
    choices = ["default"] + BASE_SIZES
    if model_name == "sora-2-pro":
        choices += PRO_SIZES
    return choices


def _update_size_choices(model_name: str, current_size: str) -> gr.Dropdown:
    choices = _size_choices(model_name)
    value = current_size if current_size in choices else "default"
    return gr.update(choices=choices, value=value)


def _parse_extra(extra_json: str) -> Dict[str, Any]:
    if not extra_json:
        return {}
    return json.loads(extra_json)


def _error_result(exc: Exception) -> Dict[str, Any]:
    return {"error": {"type": exc.__class__.__name__, "message": str(exc)}}


def _normalize_seconds(seconds: str) -> Optional[int]:
    if not seconds or seconds == "default":
        return None
    return int(seconds)


def _normalize_size(size: str) -> Optional[str]:
    if not size or size == "default":
        return None
    return size.strip()


def _normalize_optional(value: str) -> Optional[str]:
    if not value or value == "default":
        return None
    return value.strip()


def _write_temp_file(filename: str, payload: bytes) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="sora_client_"))
    path = temp_dir / filename
    with path.open("wb") as f:
        f.write(payload)
    return str(path)


def _update_download_button(path: str) -> Any:
    if path:
        return gr.update(value=path, label="Download", interactive=True)
    return gr.update(value=None, label="Download", interactive=False)


def _cleanup_download(path: str) -> Tuple[Any, str]:
    if path:
        time.sleep(0.5)
        file_path = Path(path)
        try:
            file_path.unlink()
        except OSError:
            pass
        try:
            file_path.parent.rmdir()
        except OSError:
            pass
    return gr.update(value=None, label="Downloaded", interactive=False), ""


def _audio_model_choices(mode: str) -> list[str]:
    if mode == "translations":
        return TRANSLATION_MODELS
    return TRANSCRIPTION_MODELS


def _update_audio_models(mode: str, current_model: str) -> gr.Dropdown:
    choices = _audio_model_choices(mode)
    value = current_model if current_model in choices else choices[0]
    return gr.update(choices=choices, value=value)


def _toggle_transcription_language(mode: str) -> gr.Textbox:
    if mode == "translations":
        return gr.update(value="", interactive=False)
    return gr.update(interactive=True)


def _toggle_transcription_format(mode: str, current_value: str) -> gr.Dropdown:
    choices = ["default", "json", "text", "srt", "vtt", "verbose_json"]
    value = current_value if current_value in choices else "default"
    return gr.update(choices=choices, value=value)


def _save_job_json(data: Dict[str, Any]) -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = JOBS_DIR / f"{timestamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return path


def _update_job_json(job_label: str, data: Dict[str, Any]) -> bool:
    filename = _job_label_to_filename(job_label)
    if not filename:
        return False
    path = JOBS_DIR / filename
    if not path.exists():
        return False
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return True


def _list_job_choices() -> list[str]:
    if not JOBS_DIR.exists():
        return []
    choices: list[str] = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        video_id = data.get("id", "unknown")
        status = data.get("status", "unknown")
        label = f"{path.name} | {video_id} | {status}"
        choices.append(label)
    return choices


def _job_label_to_filename(job_label: str) -> str:
    if not job_label or job_label == "Custom":
        return ""
    return job_label.split(" | ", 1)[0]


def _video_id_from_job_file(job_label: str) -> str:
    filename = _job_label_to_filename(job_label)
    if not filename:
        return ""
    path = JOBS_DIR / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("id", ""))


def _delete_job_file_for_id(job_label: str, video_id: str) -> None:
    filename = _job_label_to_filename(job_label)
    if filename:
        path = JOBS_DIR / filename
        if path.exists():
            path.unlink()
        return
    if not video_id or not JOBS_DIR.exists():
        return
    for path in JOBS_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("id")) == video_id:
            path.unlink()
            return


def _job_choices() -> list[str]:
    return ["Custom"] + _list_job_choices()


def _refresh_jobs_dropdown(current_value: str | None = None) -> gr.Dropdown:
    choices = _job_choices()
    value = current_value if current_value in choices else "Custom"
    return gr.update(choices=choices, value=value)


def _select_job_video_id(job_file: str) -> gr.Textbox:
    if not job_file or job_file == "Custom":
        return gr.update(value="", interactive=True)
    return gr.update(value=_video_id_from_job_file(job_file), interactive=False)


def save_key(api_key: str) -> str:
    if not api_key:
        return "API key is empty."
    set_api_key(api_key)
    return "API key saved to ./config/config.json"


def create_video_job(
    prompt: str,
    model: str,
    seconds: str,
    size: str,
    input_reference: Optional[str],
    extra_json: str,
    api_key: str,
    poll: bool,
    poll_interval: float,
    timeout: float,
    download: bool,
) -> Tuple[str, Optional[str]]:
    try:
        client = SoraClient(api_key=api_key or None)
        ref_path = Path(input_reference) if input_reference else None
        job = client.create_video(
            prompt=prompt,
            model=model or None,
            seconds=_normalize_seconds(seconds),
            size=_normalize_size(size),
            input_reference=ref_path,
            **_parse_extra(extra_json),
        )
        job_id = job.get("id")
        result = job

        if poll and job_id:
            result = client.wait_for_completion(
                job_id,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        _save_job_json(result)

        video_path = None
        if download and job_id and result.get("status") == "completed":
            content = client.download_video_content_bytes(job_id)
            video_path = _write_temp_file(f"{job_id}.mp4", content)

        return json.dumps(result, indent=2, ensure_ascii=True), video_path
    except Exception as exc:
        result = _error_result(exc)
        return json.dumps(result, indent=2, ensure_ascii=True), None


def retrieve_video_job(
    video_id: str,
    job_label: str,
    api_key: str,
    poll: bool,
    poll_interval: float,
    timeout: float,
    download: bool,
) -> Tuple[str, Optional[str]]:
    try:
        client = SoraClient(api_key=api_key or None)
        result = client.retrieve_video(video_id)

        if poll:
            result = client.wait_for_completion(
                video_id,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        if job_label == "Custom" or not job_label:
            _save_job_json(result)
        else:
            _update_job_json(job_label, result)

        video_path = None
        if download and result.get("status") == "completed":
            content = client.download_video_content_bytes(video_id)
            video_path = _write_temp_file(f"{video_id}.mp4", content)

        return json.dumps(result, indent=2, ensure_ascii=True), video_path
    except Exception as exc:
        result = _error_result(exc)
        return json.dumps(result, indent=2, ensure_ascii=True), None


def delete_video_job(video_id: str, job_file: str, api_key: str) -> str:
    try:
        client = SoraClient(api_key=api_key or None)
        result = client.delete_video(video_id)
        _delete_job_file_for_id(job_file, video_id)
        return json.dumps(result, indent=2, ensure_ascii=True)
    except Exception as exc:
        result = _error_result(exc)
        return json.dumps(result, indent=2, ensure_ascii=True)


def audio_to_text(
    audio_file: str,
    mode: str,
    model: str,
    language: str,
    response_format: str,
    api_key: str,
) -> Tuple[str, str]:
    try:
        client = OpenAIAudioClient(api_key=api_key or None)
        audio_path = Path(audio_file) if audio_file else None
        if not audio_path:
            raise ValueError("Audio file is required.")
        if mode == "translations":
            result = client.translate(
                audio_path,
                model=_normalize_optional(model) or TRANSLATION_MODELS[0],
                response_format=_normalize_optional(response_format),
            )
        else:
            result = client.transcribe(
                audio_path,
                model=_normalize_optional(model) or TRANSCRIPTION_MODELS[0],
                language=_normalize_optional(language),
                response_format=_normalize_optional(response_format),
            )
        raw = result.get("raw", {})
        return result.get("text", ""), json.dumps(raw, indent=2, ensure_ascii=True)
    except Exception as exc:
        result = _error_result(exc)
        return "", json.dumps(result, indent=2, ensure_ascii=True)


def text_to_speech(
    text: str,
    model: str,
    voice: str,
    instructions: str,
    api_key: str,
) -> Tuple[str, Optional[str]]:
    try:
        client = OpenAIAudioClient(api_key=api_key or None)
        payload = client.speech_bytes(
            text=text,
            model=_normalize_optional(model) or TTS_MODELS[0],
            voice=_normalize_optional(voice) or "coral",
            instructions=_normalize_optional(instructions),
        )
        output_path = _write_temp_file("speech.mp3", payload)
        return "Ready to download.", output_path
    except Exception as exc:
        result = _error_result(exc)
        return json.dumps(result, indent=2, ensure_ascii=True), None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sora-2 Videos Client") as demo:
        gr.Markdown("# Sora-2 Videos Client")
        gr.Markdown("Create and retrieve video jobs via the OpenAI Videos API.")

        with gr.Row():
            api_key = gr.Textbox(label="API key", type="password", placeholder="sk-... (optional if saved)")
            save_btn = gr.Button("Save API key")
            save_status = gr.Textbox(label="Status", interactive=False)
            save_btn.click(save_key, inputs=api_key, outputs=save_status)

        with gr.Tab("Text → Video"):
            prompt = gr.Textbox(label="Prompt", lines=3)
            model = gr.Dropdown(
                label="Model",
                choices=["sora-2", "sora-2-pro"],
                value="sora-2",
            )
            seconds = gr.Dropdown(
                label="Seconds",
                choices=["default", "4", "8", "12"],
                value="default",
            )
            size = gr.Dropdown(
                label="Size",
                choices=_size_choices("sora-2"),
                value="default",
            )
            extra = gr.Textbox(
                label="Extra JSON",
                lines=3,
                placeholder='{"seed": 123}',
            )
            poll = gr.Checkbox(label="Poll until completed", value=False)
            poll_interval = gr.Number(label="Poll interval (seconds)", value=5.0, precision=1)
            timeout = gr.Number(label="Timeout (seconds)", value=600.0, precision=0)
            download = gr.Checkbox(label="Download MP4 when completed", value=True)
            create_btn = gr.Button("Create video job")
            create_result = gr.Textbox(label="Result JSON", lines=12)
            create_download_state = gr.State("")
            create_download = gr.DownloadButton(label="Download", interactive=False)
            create_btn.click(
                create_video_job,
                inputs=[
                    prompt,
                    model,
                    seconds,
                    size,
                    gr.State(None),
                    extra,
                    api_key,
                    poll,
                    poll_interval,
                    timeout,
                    download,
                ],
                outputs=[create_result, create_download_state],
            )
            create_download_state.change(
                _update_download_button,
                inputs=create_download_state,
                outputs=create_download,
            )
            create_download.click(
                _cleanup_download,
                inputs=create_download_state,
                outputs=[create_download, create_download_state],
            )
            model.change(_update_size_choices, inputs=[model, size], outputs=size)

        with gr.Tab("Image → Video"):
            prompt_i = gr.Textbox(label="Prompt", lines=3)
            input_reference = gr.File(
                label="Input reference (image or mp4)",
                type="filepath",
                file_types=[".jpg", ".jpeg", ".png", ".webp", ".mp4"],
            )
            model_i = gr.Dropdown(
                label="Model",
                choices=["sora-2", "sora-2-pro"],
                value="sora-2",
            )
            seconds_i = gr.Dropdown(
                label="Seconds",
                choices=["default", "4", "8", "12"],
                value="default",
            )
            size_i = gr.Dropdown(
                label="Size",
                choices=_size_choices("sora-2"),
                value="default",
            )
            extra_i = gr.Textbox(
                label="Extra JSON",
                lines=3,
                placeholder='{"seed": 123}',
            )
            poll_i = gr.Checkbox(label="Poll until completed", value=False)
            poll_interval_i = gr.Number(label="Poll interval (seconds)", value=5.0, precision=1)
            timeout_i = gr.Number(label="Timeout (seconds)", value=600.0, precision=0)
            download_i = gr.Checkbox(label="Download MP4 when completed", value=True)
            create_btn_i = gr.Button("Create video job")
            create_result_i = gr.Textbox(label="Result JSON", lines=12)
            create_download_state_i = gr.State("")
            create_download_i = gr.DownloadButton(label="Download", interactive=False)
            create_btn_i.click(
                create_video_job,
                inputs=[
                    prompt_i,
                    model_i,
                    seconds_i,
                    size_i,
                    input_reference,
                    extra_i,
                    api_key,
                    poll_i,
                    poll_interval_i,
                    timeout_i,
                    download_i,
                ],
                outputs=[create_result_i, create_download_state_i],
            )
            create_download_state_i.change(
                _update_download_button,
                inputs=create_download_state_i,
                outputs=create_download_i,
            )
            create_download_i.click(
                _cleanup_download,
                inputs=create_download_state_i,
                outputs=[create_download_i, create_download_state_i],
            )
            model_i.change(_update_size_choices, inputs=[model_i, size_i], outputs=size_i)

        with gr.Tab("Remix (Video → Video)"):
            jobs_refresh_m = gr.Button("Refresh jobs")
            jobs_m = gr.Dropdown(label="Jobs", choices=_job_choices(), value="Custom")
            remix_id = gr.Textbox(label="Completed video ID")
            remix_prompt = gr.Textbox(label="Remix prompt", lines=3)
            remix_btn = gr.Button("Start remix job")
            remix_result = gr.Textbox(label="Result JSON", lines=12)
            remix_poll = gr.Checkbox(label="Poll until completed", value=False)
            remix_interval = gr.Number(label="Poll interval (seconds)", value=5.0, precision=1)
            remix_timeout = gr.Number(label="Timeout (seconds)", value=600.0, precision=0)
            remix_download = gr.Checkbox(label="Download MP4 when completed", value=True)
            remix_download_state = gr.State("")
            remix_download_btn = gr.DownloadButton(label="Download", interactive=False)

            def _remix_flow(
                video_id: str,
                prompt_text: str,
                api_key_value: str,
                poll_flag: bool,
                interval_value: float,
                timeout_value: float,
                download_flag: bool,
            ) -> Tuple[str, Optional[str]]:
                try:
                    client = SoraClient(api_key=api_key_value or None)
                    job = client.remix_video(video_id, prompt_text)
                    result = job
                    job_id = job.get("id") or video_id
                    if poll_flag and job_id:
                        result = client.wait_for_completion(
                            job_id,
                            poll_interval=interval_value,
                            timeout=timeout_value,
                        )
                    video_path = None
                    if download_flag and job_id and result.get("status") == "completed":
                        content = client.download_video_content_bytes(job_id)
                        video_path = _write_temp_file(f"{job_id}.mp4", content)
                    _save_job_json(result)
                    return (
                        json.dumps(result, indent=2, ensure_ascii=True),
                        video_path,
                    )
                except Exception as exc:
                    result = _error_result(exc)
                    return json.dumps(result, indent=2, ensure_ascii=True), None

            remix_btn.click(
                _remix_flow,
                inputs=[
                    remix_id,
                    remix_prompt,
                    api_key,
                    remix_poll,
                    remix_interval,
                    remix_timeout,
                    remix_download,
                ],
                outputs=[remix_result, remix_download_state],
            )
            remix_download_state.change(
                _update_download_button,
                inputs=remix_download_state,
                outputs=remix_download_btn,
            )
            remix_download_btn.click(
                _cleanup_download,
                inputs=remix_download_state,
                outputs=[remix_download_btn, remix_download_state],
            )
            jobs_refresh_m.click(_refresh_jobs_dropdown, inputs=jobs_m, outputs=jobs_m)
            jobs_m.change(_select_job_video_id, inputs=jobs_m, outputs=remix_id)

        with gr.Tab("Delete"):
            jobs_refresh_d = gr.Button("Refresh jobs")
            jobs_d = gr.Dropdown(label="Jobs", choices=_job_choices(), value="Custom")
            delete_id = gr.Textbox(label="Video ID")
            delete_btn = gr.Button("Delete video job")
            delete_result = gr.Textbox(label="Result JSON", lines=8)
            delete_btn.click(
                delete_video_job,
                inputs=[
                    delete_id,
                    jobs_d,
                    api_key,
                ],
                outputs=delete_result,
            )
            jobs_refresh_d.click(_refresh_jobs_dropdown, inputs=jobs_d, outputs=jobs_d)
            jobs_d.change(_select_job_video_id, inputs=jobs_d, outputs=delete_id)

        with gr.Tab("Retrieve"):
            jobs_refresh_r = gr.Button("Refresh jobs")
            jobs_r = gr.Dropdown(label="Jobs", choices=_job_choices(), value="Custom")
            video_id = gr.Textbox(label="Video ID")
            poll_r = gr.Checkbox(label="Poll until completed", value=False)
            poll_interval_r = gr.Number(label="Poll interval (seconds)", value=5.0, precision=1)
            timeout_r = gr.Number(label="Timeout (seconds)", value=600.0, precision=0)
            download_r = gr.Checkbox(label="Download MP4 when completed", value=True)
            retrieve_btn = gr.Button("Retrieve video job")
            retrieve_result = gr.Textbox(label="Result JSON", lines=12)
            retrieve_download_state = gr.State("")
            retrieve_download_btn = gr.DownloadButton(label="Download", interactive=False)
            retrieve_btn.click(
                retrieve_video_job,
                inputs=[
                    video_id,
                    jobs_r,
                    api_key,
                    poll_r,
                    poll_interval_r,
                    timeout_r,
                    download_r,
                ],
                outputs=[retrieve_result, retrieve_download_state],
            )
            retrieve_download_state.change(
                _update_download_button,
                inputs=retrieve_download_state,
                outputs=retrieve_download_btn,
            )
            retrieve_download_btn.click(
                _cleanup_download,
                inputs=retrieve_download_state,
                outputs=[retrieve_download_btn, retrieve_download_state],
            )
            jobs_refresh_r.click(_refresh_jobs_dropdown, inputs=jobs_r, outputs=jobs_r)
            jobs_r.change(_select_job_video_id, inputs=jobs_r, outputs=video_id)

        with gr.Tab("Audio → Text"):
            audio_file = gr.File(
                label="Audio file",
                type="filepath",
                file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"],
            )
            audio_mode = gr.Dropdown(
                label="Mode",
                choices=["transcriptions", "translations"],
                value="transcriptions",
            )
            audio_model = gr.Dropdown(
                label="Model",
                choices=TRANSCRIPTION_MODELS,
                value=TRANSCRIPTION_MODELS[0],
            )
            audio_language = gr.Textbox(
                label="Language (optional, e.g. zh)",
                placeholder="Leave empty for auto-detect",
            )
            audio_format = gr.Dropdown(
                label="Response format",
                choices=["default", "json", "text", "srt", "vtt", "verbose_json"],
                value="default",
            )
            audio_btn = gr.Button("Run audio → text")
            audio_text = gr.Textbox(label="Text", lines=6)
            audio_raw = gr.Textbox(label="Raw JSON", lines=10)
            audio_btn.click(
                audio_to_text,
                inputs=[
                    audio_file,
                    audio_mode,
                    audio_model,
                    audio_language,
                    audio_format,
                    api_key,
                ],
                outputs=[audio_text, audio_raw],
            )
            audio_mode.change(
                _update_audio_models,
                inputs=[audio_mode, audio_model],
                outputs=audio_model,
            )
            audio_mode.change(_toggle_transcription_language, inputs=audio_mode, outputs=audio_language)
            audio_mode.change(
                _toggle_transcription_format,
                inputs=[audio_mode, audio_format],
                outputs=audio_format,
            )

        with gr.Tab("Text → Speech"):
            tts_text = gr.Textbox(label="Input text", lines=4)
            tts_model = gr.Dropdown(label="Model", choices=TTS_MODELS, value=TTS_MODELS[0])
            tts_voice = gr.Textbox(label="Voice", value="coral")
            tts_instructions = gr.Textbox(
                label="Instructions (optional)",
                placeholder="e.g. Speak in a cheerful and positive tone.",
                lines=2,
            )
            tts_btn = gr.Button("Generate speech")
            tts_status = gr.Textbox(label="Status", interactive=False)
            tts_download_state = gr.State("")
            tts_download_btn = gr.DownloadButton(label="Download", interactive=False)
            tts_btn.click(
                text_to_speech,
                inputs=[
                    tts_text,
                    tts_model,
                    tts_voice,
                    tts_instructions,
                    api_key,
                ],
                outputs=[tts_status, tts_download_state],
            )
            tts_download_state.change(
                _update_download_button,
                inputs=tts_download_state,
                outputs=tts_download_btn,
            )
            tts_download_btn.click(
                _cleanup_download,
                inputs=tts_download_state,
                outputs=[tts_download_btn, tts_download_state],
            )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
