"""Package the reviewed presentation, public source and evidence for the AED."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp(seconds: float) -> str:
    minutes, second = divmod(int(seconds), 60)
    return f"{minutes:02}:{second:02}"


def check_source_path(name: str, mode: int) -> None:
    """Git tracking does not make data, checkpoints or credentials publishable."""
    path = PurePosixPath(name)
    private = {".local", ".venv", ".kaggle", ".aws", ".ssh", ".cache", "data", "artifacts", "__pycache__"}
    credential_names = {"kaggle.json", "credentials", "credentials.json", "secrets.json", "token.json", "id_rsa", "id_ed25519"}
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or any(part.lower() in private for part in path.parts)
            or path.name.lower().startswith(".env") or path.name.lower() in credential_names
            or path.suffix.lower() in {".csv", ".npy", ".npz", ".parquet", ".pkl", ".joblib", ".pt", ".pem", ".key"}
            or stat.S_ISLNK(mode)):
        raise ValueError("Private data, credentials or links cannot be packaged from Git: " + name)


def check_review(manifest: dict, files: dict[str, bytes], proofs: dict[str, bytes]) -> None:
    """Check provenance of existing QA; this does not perform audiovisual QA."""
    checks = manifest["verification"]
    video_sha = digest(files["apresentacao/video.mp4"])
    review, asr = (json.loads(proofs[name]) for name in ("review.json", "asr.json"))
    visual, alignment = (json.loads(proofs[name]) for name in ("visual-review.json", "asr-by-slide.json"))
    count = manifest["slides"]
    expected = list(range(1, count + 1))
    if (checks.get("review_report_sha256") != digest(proofs["review.json"])
            or checks.get("asr_report_sha256") != digest(proofs["asr.json"])
            or checks.get("visual_report_sha256") != digest(proofs["visual-review.json"])
            or checks.get("asr_alignment_sha256") != digest(proofs["asr-by-slide.json"])):
        raise ValueError("QA report bytes differ from the reviewed manifest hashes.")
    decoded = review.get("full_av_decode", {})
    recorded_asr = review.get("asr") or {}
    if (review.get("video_sha256") != video_sha or review.get("slides") != count
            or review.get("technical_checks") != "passed"
            or decoded.get("exit_code") != 0 or decoded.get("error_output") != ""
            or decoded.get("scope") != "entire video and audio streams"
            or asr.get("input_media_sha256") != video_sha or asr.get("input_media") != "video.mp4"
            or asr.get("whole_input_processed") is not True or not asr.get("segments")
            or recorded_asr.get("input_media_sha256") != video_sha
            or recorded_asr.get("whole_input_processed") is not True):
        raise ValueError("Review, full AV decode and ASR must identify the same complete video.")
    if (review.get("visually_reviewed_slides") != expected
            or checks.get("visual_reviewed_slides") != expected
            or visual.get("status") != "passed" or visual.get("reviewed_slides") != expected
            or [item.get("slide") for item in alignment.get("slides", [])] != expected
            or any(not item.get("asr_text", "").strip() for item in alignment["slides"])):
        raise ValueError("Visual review and nonempty ASR alignment must cover exactly slides 1..N.")


def package(media: Path, snapshot: Path, source_ref: str, output: Path) -> dict:
    """Use an immutable Git commit and hash-verified, already reviewed media."""
    manifest_body, snapshot_body = (media / "manifest.json").read_bytes(), snapshot.read_bytes()
    manifest, evidence = json.loads(manifest_body), json.loads(snapshot_body)
    checks = manifest.get("verification", {})
    if (manifest.get("status") != "evidence-backed" or not manifest.get("video_generated")
            or checks.get("technical_checks") != "passed"
            or checks.get("whole_audio_asr") is not True
            or checks.get("all_slides_visually_reviewed") is not True):
        raise ValueError("Complete audiovisual and visual review is required before packaging.")
    try:
        snapshot_key = snapshot.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        snapshot_key = snapshot.name
    if manifest.get("inputs", {}).get(snapshot_key) != digest(snapshot_body):
        raise ValueError("The evidence snapshot does not match the presentation inputs.")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", source_ref + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    source_zip = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=ROOT)
    files = {}
    with zipfile.ZipFile(io.BytesIO(source_zip)) as source:
        for item in source.infolist():
            if item.is_dir():
                continue
            check_source_path(item.filename, item.external_attr >> 16)
            files["codigo/" + item.filename] = source.read(item)
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Presentation file names must be local to their directory.")
        body = (media / name).read_bytes()
        if digest(body) != expected:
            raise ValueError("Presentation output differs from the reviewed bytes: " + name)
        files["apresentacao/" + name] = body
    files["apresentacao/manifest.json"] = manifest_body
    files["apresentacao/evidence-snapshot.json"] = snapshot_body
    files["apresentacao/video.srt"] = files["apresentacao/captions.srt"]
    proofs = {name: (media / "review" / name).read_bytes()
              for name in ("review.json", "visual-review.json", "asr.json", "asr-by-slide.json")}
    check_review(manifest, files, proofs)
    files.update({"provas/" + name: body for name, body in proofs.items()})
    script = json.loads(files["apresentacao/script.json"])["slides"]
    timeline = json.loads(files["apresentacao/timeline.json"])
    if len(script) != manifest["slides"] or len(timeline) != len(script):
        raise ValueError("Script, manifest and timeline must cover the same slides.")
    chapters, previous = [], None
    for slide, time in zip(script, timeline):
        if slide["chapter"] != previous:
            chapters.append((timestamp(time["start"]), slide["chapter"], time["slide"]))
        previous = slide["chapter"]
    chapters_text = "\n".join(f"{start} — {title} (slide {number})" for start, title, number in chapters)
    result = evidence["results"]
    readme = (
        "AED — REDES NEURAIS\nFraudLab: detecção de fraude em cartão\n\n"
        "ABRIR\nExtraia todo este ZIP e abra ENTREGA.html.\n"
        "Vídeo: apresentacao/video.mp4; slides: apresentacao/slides.html.\n"
        "Legendas: captions.srt/captions.vtt; video.srt permite associação com o MP4 em um player local.\n"
        "Código completo: codigo/; configuração e execução: codigo/docs/reproduce.md.\n"
        "Comprovantes: codigo/evidence/ e provas/. MANIFEST.json registra os hashes dos arquivos.\n\n"
        f"PARTICIPAÇÃO\nSubmissão Kaggle {result['submission_id']}; AUC pública {result['public_auc']:.5f}; "
        f"posição {result['ranking']}/{result['total_teams']}. Consulta: {result['observed_at']}.\n"
        "O ranking é provisório. Primeiras posições não foram alcançadas. Não há promessa de vitória.\n"
        "A CV foi usada para selecionar a solução; não existe novo holdout independente.\n\n"
        "AUTORIA E LICENÇAS\nProjeto: perfis aikolopes-DS (GitHub) e aikoog (Kaggle).\n"
        "Código próprio MIT. GPT-2 externo treinado por Masaya Kawamata; previsões v2 CC BY 4.0.\n"
        "O treinamento externo não foi reexecutado pelo projeto. Créditos completos nos slides e no código.\n"
        "Dados e credenciais são obtidos separadamente pelo mecanismo oficial do Kaggle.\n\n"
        "VALIDAÇÃO\nDecodificação audiovisual integral, transcrição automática integral e quadros revisados.\n"
        "Isso não significa escuta humana ou reprodução contínua integral; ASR pode errar termos e números.\n"
        "Participação e publicação foram verificadas. Este pacote não envia a atividade ao portal da faculdade.\n\n"
        f"Código congelado: {revision}\n\nSUMÁRIO DO VÍDEO\n{chapters_text}\n"
    )
    files["LEIA-ME.txt"] = readme.encode("utf-8")
    items = "".join(f"<li>{html.escape(start)} · {html.escape(title)} · slide {number}</li>"
                    for start, title, number in chapters)
    index = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FraudLab · AED</title>
<style>body{{max-width:900px;margin:5vh auto;padding:24px;background:#f3f1e9;color:#182126;font:18px/1.6 system-ui}}a{{color:#00798d}}li{{margin:.5em 0}}small{{overflow-wrap:anywhere}}h1{{line-height:1.2}}nav a{{display:inline-block;margin:0 24px 16px 0}}</style>
<h1>FraudLab<br>AED · Redes Neurais</h1>
<p>Problema, proposta de solução, construção, matemática, analogias do cotidiano, código e participação real no Kaggle.</p>
<nav><a href="apresentacao/video.mp4">Assistir ao vídeo</a><a href="apresentacao/slides.html">Abrir os slides</a><a href="codigo/README.md">Consultar o código</a><a href="LEIA-ME.txt">Instruções e créditos</a></nav>
<p>{len(script)} slides · {timestamp(manifest['duration_seconds'])} de vídeo em português.</p>
<p>AUC pública: <strong>{result['public_auc']:.5f}</strong> · posição <strong>{result['ranking']}/{result['total_teams']}</strong> · submissão {result['submission_id']}.</p>
<p><small>Consulta: {html.escape(result['observed_at'])}. Ranking provisório; primeiras posições ainda não alcançadas.</small></p>
<h2>Sumário do vídeo</h2><ul>{items}</ul>
<p>O código, as evidências e os arquivos desta apresentação estão neste pacote. Dados e credenciais do Kaggle ficam fora dele.</p>
<p><small>Código congelado: {revision}. Código MIT; previsões GPT-2 de Masaya Kawamata sob CC BY 4.0.</small></p></html>"""
    files["ENTREGA.html"] = index.encode("utf-8")
    package_manifest = {"source_commit": revision, "presentation_slides": len(script),
                        "duration_seconds": manifest["duration_seconds"], "result_snapshot": result,
                        "files": {name: digest(body) for name, body in files.items()}}
    files["MANIFEST.json"] = (json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError("Use a new output path; existing delivery packages are immutable.")
    try:
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
            for name, body in files.items():
                archive.writestr(name, body)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("The delivery ZIP failed its integrity check.")
            for name, body in files.items():
                if digest(archive.read(name)) != digest(body):
                    raise ValueError("A ZIP entry differs from its source: " + name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    sha256 = digest(output.read_bytes())
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256}  {output.name}\n", encoding="utf-8")
    return {"path": str(output), "sha256": sha256, "bytes": output.stat().st_size,
            "files": len(files), "source_commit": revision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.presentation_dir, args.snapshot, args.source_ref, args.output)))


if __name__ == "__main__":
    main()
