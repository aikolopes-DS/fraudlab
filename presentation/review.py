"""Technical video checks, review frames and optional whole-audio ASR.

This records automatic verification. It never claims a person watched or heard
the presentation. Run with python -m presentation.review --help.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .build import read_json, sha256, write_json


def probe_video(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "quiet", "-show_streams", "-show_format",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def verify_artifacts(root: Path, expected_hashes: dict[str, str]) -> None:
    if "timeline.json" not in expected_hashes:
        raise ValueError("The manifest must include timeline.json before review.")
    for name, expected in expected_hashes.items():
        if Path(name).name != name or sha256(root / name) != expected:
            raise ValueError("An artifact no longer matches its recorded hash: " + name)


def decode_all(video: Path) -> dict:
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                              "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if decoded.returncode or decoded.stderr.strip():
        raise RuntimeError("Full AV decoding failed: " + decoded.stderr[-1500:])
    return {"scope": "entire video and audio streams", "exit_code": 0, "error_output": ""}


def timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_captions(path: Path, duration: float) -> list[dict]:
    cues = []
    previous_end = 0.0
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        timing_index = next((index for index, line in enumerate(lines) if " --> " in line), None)
        match = (re.fullmatch(r"(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})",
                             lines[timing_index]) if timing_index is not None else None)
        if match is None:
            if block.strip() == "WEBVTT":
                continue
            raise ValueError("Malformed caption block.")
        start, end = (timestamp(value) for value in match.groups())
        if start < 0 or end <= start or start + 0.05 < previous_end or end > duration + 0.15:
            raise ValueError("Caption timing is invalid or falls outside the video.")
        cue_text = "\n".join(lines[timing_index + 1:])
        if not cue_text.strip():
            raise ValueError("Empty caption cue.")
        cues.append({"start": start, "end": end, "text": cue_text})
        previous_end = end
    if not cues:
        raise ValueError("No caption cues found.")
    return cues


def validate_timeline(timeline: list[dict], slide_count: int, duration: float) -> None:
    if len(timeline) != slide_count:
        raise ValueError("The timeline does not cover every slide.")
    elapsed = 0.0
    for expected, item in enumerate(timeline, 1):
        if item["slide"] != expected or item["duration"] <= 0 or abs(item["start"] - elapsed) > 0.001:
            raise ValueError("The slide timeline is not consecutive and complete.")
        elapsed += item["duration"]
    if abs(elapsed - duration) > 0.15:
        raise ValueError("Timeline and video durations differ by more than 150 ms.")


def extract_frames(video: Path, timeline: list[dict], output: Path) -> list[dict]:
    from PIL import Image, ImageDraw, ImageOps

    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def extract(item: dict) -> dict:
        point = item["start"] + item["duration"] / 2
        filename = f"slide-{item['slide']:02}.png"
        result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-y", "-ss", f"{point:.6f}",
                        "-i", str(video), "-frames:v", "1", "-vf", "scale=960:-2", "-threads", "1",
                        str(frames_dir / filename)], capture_output=True, text=True)
        if result.returncode or result.stderr.strip():
            raise RuntimeError(f"Could not extract slide {item['slide']}: " + result.stderr[-1500:])
        return {"slide": item["slide"], "timestamp": point, "file": "frames/" + filename}

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(extract, timeline))
    for start in range(0, len(frames), 9):
        contact = Image.new("RGB", (1920, 1170), "#f5f3ec")
        draw = ImageDraw.Draw(contact)
        for index, item in enumerate(frames[start:start + 9]):
            x, y = (index % 3) * 640, (index // 3) * 390
            with Image.open(output / item["file"]) as image:
                contact.paste(ImageOps.contain(image, (640, 360)), (x, y))
            draw.text((x + 16, y + 365), f"Slide {item['slide']} | {item['timestamp']:.2f}s", fill="#152c2a")
        contact.save(output / f"contact-{start // 9 + 1:02}.png")
    return frames


def transcribe_audio(audio: Path, output: Path, model_name: str, device: str, threads: int) -> dict:
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(model_name, device=device,
                         compute_type="float16" if device == "cuda" else "int8", cpu_threads=threads)
    generated, info = model.transcribe(str(audio), language="pt", beam_size=5,
                                       vad_filter=True, condition_on_previous_text=False)
    segments = []
    for item in generated:
        segments.append({"start": item.start, "end": item.end, "text": item.text.strip(),
                         "avg_logprob": item.avg_logprob, "no_speech_probability": item.no_speech_prob})
        if len(segments) % 30 == 0:
            print(f"ASR: {len(segments)} segmentos; posição {item.end:.1f}s", flush=True)
    if not segments:
        raise ValueError("ASR did not return any speech segments.")
    report = {"method": "automatic speech recognition", "engine": "faster-whisper",
              "version": importlib.metadata.version("faster-whisper"), "model": model_name,
              "device": device, "language": info.language, "input_duration_seconds": info.duration,
              "input_media": audio.name, "input_media_sha256": sha256(audio), "whole_input_processed": True,
              "elapsed_seconds": round(time.perf_counter() - started, 3), "segments": segments,
              "limitation": "ASR can misrecognize speech and is not a human listening review."}
    write_json(output / "asr.json", report)
    (output / "asr.txt").write_text("\n".join(item["text"] for item in segments) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", type=Path, default=Path("artifacts/presentation"))
    parser.add_argument("--output", type=Path, help="Default: <presentation-dir>/review")
    parser.add_argument("--asr", action="store_true", help="Transcribe the entire exported video's audio locally")
    parser.add_argument("--asr-model", default="base", help="Multilingual faster-whisper model name or local model path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    root = args.presentation_dir.resolve()
    output = (args.output or root / "review").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "manifest.json")
    if not manifest.get("video_generated"):
        raise ValueError("This presentation has no generated video to inspect.")
    verify_artifacts(root, manifest["files"])
    video = root / "video.mp4"
    probe = probe_video(video)
    duration = float(probe["format"]["duration"])
    streams = probe["streams"]
    visual = next(item for item in streams if item["codec_type"] == "video")
    audio = next(item for item in streams if item["codec_type"] == "audio")
    if (visual["width"], visual["height"], visual["codec_name"], audio["codec_name"]) != (1920, 1080, "h264", "aac"):
        raise ValueError("Expected 1080p H.264 video and AAC narration.")
    timeline = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    script = read_json(root / "script.json")["slides"]
    validate_timeline(timeline, len(script), duration)
    srt, vtt = (read_captions(root / name, duration) for name in ("captions.srt", "captions.vtt"))
    if srt != vtt:
        raise ValueError("SRT and WebVTT captions differ.")
    print("Decodificando integralmente vídeo e áudio...", flush=True)
    full_decode = decode_all(video)
    print("Decodificação integral passou; extraindo quadros...", flush=True)
    frames = extract_frames(video, timeline, output)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "video_sha256": sha256(video),
              "duration_seconds": duration, "slides": len(script), "caption_cues": len(srt),
              "technical_checks": "passed", "full_av_decode": full_decode, "frames_extracted": frames,
              "visually_reviewed_slides": [], "watched_in_full": False,
              "human_listening_review": False, "asr": None}
    # Persist completed mechanical checks even if an optional model download fails.
    write_json(output / "review.json", report)
    if args.asr:
        transcription = transcribe_audio(video, output, args.asr_model, args.device, args.threads)
        report["asr"] = {key: value for key, value in transcription.items() if key != "segments"}
        comparisons = []
        for item, slide in zip(timeline, script):
            end = item["start"] + item["duration"]
            recognized = [segment["text"] for segment in transcription["segments"]
                          if item["start"] <= (segment["start"] + segment["end"]) / 2 < end]
            comparisons.append({"slide": item["slide"], "title": slide["title"],
                                "expected_narration": slide["narration"], "asr_text": " ".join(recognized)})
        write_json(output / "asr-by-slide.json", {"method": "segment midpoint alignment", "slides": comparisons})
        write_json(output / "review.json", report)
    print(json.dumps({"technical_checks": "passed", "slides": len(script), "duration_seconds": duration,
                      "whole_audio_asr": bool(report["asr"]), "visual_review": "pending",
                      "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
