"""Reuse exact speech inputs across slide reordering without changing old media."""

import hashlib
import json
from pathlib import Path
import shutil


def edge_paths(audio_dir: Path, request: dict, reuse_dir: Path | None = None):
    key = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    cache = audio_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    target = [cache / f"{key}{suffix}" for suffix in (".mp3", ".jsonl", ".request.json")]
    if target[2].is_file() and all(path.is_file() and path.stat().st_size for path in target[:2]):
        try:
            if json.loads(target[2].read_text(encoding="utf-8")) == request:
                return *target, True
        except (OSError, ValueError):
            pass
    roots = [audio_dir]
    if reuse_dir is not None:
        roots += [reuse_dir, reuse_dir / "cache"]
    for source in roots:
        for previous in sorted(source.glob("*.request.json")):
            try:
                if json.loads(previous.read_text(encoding="utf-8")) != request:
                    continue
            except (OSError, ValueError):
                continue
            prefix = previous.name.removesuffix(".request.json")
            files = [source / f"{prefix}{suffix}" for suffix in (".mp3", ".jsonl")]
            if all(path.is_file() and path.stat().st_size for path in files):
                for origin, destination in zip([*files, previous], target):
                    if origin.resolve() != destination.resolve():
                        shutil.copyfile(origin, destination)
                return *target, True
    return *target, False
