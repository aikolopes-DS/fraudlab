"""Build an editable, evidence-backed deck and optional narrated MP4.

Run from the repository root: python -m presentation.build --help
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import wave

from .content import Slide, create_slides, ensure_pairs, validate_evidence


WIDTH, HEIGHT = 1920, 1080
BACKGROUND = "#f5f3ec"
INK = "#152c2a"
ACCENT = "#007b68"
MUTED = "#536561"
FLOWS = {
    "architecture": ["Entradas", "Camadas ocultas", "Sigmoid", "Probabilidade"],
    "pipeline": ["Dados", "Treino e OOF", "CSV validado", "Kaggle"],
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def write_html(slides: list[Slide], output: Path, draft: bool) -> None:
    parts = []
    for index, slide in enumerate(slides, 1):
        points = "".join(f"<li>{html.escape(point)}</li>" for point in slide.points)
        formula = (f'<div class="formula"><img src="formula-{index:02}.png" '
                   f'alt="{html.escape(slide.formula_text)}"></div>' if slide.formula else "")
        code = (f'<pre><code>{html.escape(slide.code)}</code></pre><p class="code-source">'
                f'{html.escape(slide.code_reference)}</p>' if slide.code else "")
        diagram = ('<div class="flow" aria-label="Fluxo">' +
                   '<span aria-hidden="true">→</span>'.join('<strong>' + html.escape(label) + '</strong>'
                   for label in FLOWS.get(slide.visual, [])) + '</div>' if slide.visual else '')
        parts.append(f'''<section class="slide" id="slide-{index}" aria-label="Slide {index}">
  <p class="eyebrow">{html.escape(slide.chapter)} · {html.escape(slide.kind)}</p>
  <h1>{html.escape(slide.title)}</h1><ul>{points}</ul>{formula}{diagram}{code}
  <details><summary>Roteiro da narração</summary><p>{html.escape(slide.narration)}</p></details>
  <footer><span>{"RASCUNHO · EVIDÊNCIAS INCOMPLETAS" if draft else "FRAUDLAB · EVIDÊNCIAS DOCUMENTADAS"}</span>
  <span>{index:02} / {len(slides):02}</span></footer></section>''')
    document = '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FraudLab — apresentação</title><style>
:root{color-scheme:light;--bg:#f5f3ec;--ink:#152c2a;--accent:#007b68}*{box-sizing:border-box}
body{margin:0;background:#e1e4dc;color:var(--ink);font:20px/1.45 system-ui,sans-serif}
nav{position:sticky;top:0;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 4vw;background:var(--ink);color:var(--bg)}
button,a{color:inherit;font:inherit}button{background:transparent;border:1px solid #859a94;padding:6px 14px;border-radius:4px;cursor:pointer}a{margin-left:auto}
button:focus-visible,a:focus-visible{outline:3px solid #7ce6c4;outline-offset:3px}
main{max-width:1440px;margin:32px auto}.slide{position:relative;min-height:740px;background:var(--bg);padding:60px 7% 100px;margin:24px;box-shadow:0 3px 12px #152c2a14;border-top:6px solid var(--accent)}
.eyebrow{font-size:15px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}h1{font-size:clamp(30px,4vw,60px);line-height:1.12;font-weight:650;max-width:1100px;margin:28px 0 40px}
ul{list-style:none;padding:0;max-width:1100px}li{padding:12px 0 12px 24px;position:relative;overflow-wrap:anywhere}li:before{content:' ';position:absolute;left:0;top:25px;background:var(--accent);width:7px;height:7px}
.formula img{max-width:100%;max-height:125px;object-fit:contain;margin:18px 0}details{font-size:17px;color:#536561;margin-top:30px}summary{cursor:pointer}
pre{padding:24px;background:#e1e8df;white-space:pre-wrap;overflow-wrap:anywhere;font:20px/1.65 ui-monospace,monospace}.code-source{font-size:15px;color:#536561}
.flow{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:35px 0;flex-wrap:wrap}.flow strong{padding:15px;border:1px solid var(--accent);font-size:17px;border-radius:5px}
footer{position:absolute;bottom:30px;left:7%;right:7%;display:flex;justify-content:space-between;color:#536561;font-size:13px;letter-spacing:.05em}
body.present main{max-width:none;margin:0}body.present .slide{display:none;min-height:calc(100vh - 70px);margin:0;box-shadow:none;padding-top:4vh}body.present .slide.active{display:block}
@media(max-width:700px){body{font-size:17px}.slide{margin:16px 10px;padding:30px 24px 95px;min-height:650px}h1{margin:20px 0 30px}footer{left:24px;right:24px;gap:20px;font-size:10px}nav{font-size:14px}}
@media print{nav,details{display:none}.slide{break-after:page;margin:0;box-shadow:none;min-height:95vh}body{background:white}.slide:last-child{break-after:auto}}
</style></head><body><nav aria-label="Controles da apresentação"><button id="mode">Apresentar</button>
<button id="previous" aria-label="Slide anterior">←</button><span id="progress" aria-live="polite"></span><button id="next" aria-label="Próximo slide">→</button>
<a href="script.json" download>Roteiro (JSON)</a></nav><main>''' + "\n".join(parts) + '''</main>
<script>
const slides=[...document.querySelectorAll('.slide')];let current=0;
function show(n){current=Math.max(0,Math.min(n,slides.length-1));slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.querySelector('#progress').textContent=`${current+1} / ${slides.length}`;document.querySelector('#previous').disabled=current===0;document.querySelector('#next').disabled=current===slides.length-1;}
document.querySelector('#mode').onclick=()=>{const active=document.body.classList.toggle('present');document.querySelector('#mode').textContent=active?'Ver todos':'Apresentar';show(current);};
document.querySelector('#previous').onclick=()=>show(current-1);document.querySelector('#next').onclick=()=>show(current+1);
document.addEventListener('keydown',e=>{if(!document.body.classList.contains('present'))return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1);}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1);}if(e.key==='Escape'){document.body.classList.remove('present');document.querySelector('#mode').textContent='Apresentar';}});show(0);
</script></body></html>'''
    (output / "slides.html").write_text(document, encoding="utf-8")


def insert_code_slides(slides: list[Slide], source_root: Path, required: bool) -> list[Path]:
    """Show literal current implementation, with its file and line reference."""
    specifications = [
        ("src/fraudlab/models.py", "loss.backward()", 2, "O gradiente vira uma atualização", [
            "Trecho literal do treinamento da rede neural",
            "backward calcula gradientes; step aplica o otimizador configurado",
        ], "Este é um trecho literal do código do projeto. A chamada backward calcula "
           "os gradientes da perda. A chamada step pede ao otimizador que atualize "
           "os parâmetros. O cálculo da perda e a limpeza dos gradientes acontecem "
           "nas linhas anteriores, disponíveis no arquivo completo."),
        ("src/fraudlab/experiment.py", 'oof_frame.to_csv(run / "oof.csv"', 4,
         "O resultado é salvo para inspeção", [
             "Previsões OOF e previsões do teste ficam separadas",
             "O CSV de submissão é salvo antes do envio à plataforma",
         ], "Aqui o código salva as previsões fora do treino e as previsões "
            "do teste. O arquivo de submissão é um artefato local. "
            "Só depois de validá-lo e conferir o processamento no Kaggle "
            "podemos apresentar uma pontuação pública como observada."),
    ]
    used, additions = [], []
    for relative, marker, count, title, points, narration in specifications:
        path = source_root / relative
        if not path.is_file():
            if required:
                raise ValueError(f"Falta código fonte para a apresentação final: {relative}")
            continue
        text = path.read_text(encoding="utf-8").splitlines()
        start = next((index for index, line in enumerate(text) if marker in line), None)
        if start is None:
            if required:
                raise ValueError(f"Trecho de código mudou; revise o roteiro: {relative}")
            continue
        import textwrap
        code = textwrap.dedent("\n".join(text[start:start + count]))
        additions.append(Slide("08 / Da execução à competição", title, points, narration,
                               kind="technical", code=code, code_reference=f"{relative}:{start + 1}"))
        if "models.py" in relative:
            additions.append(Slide("08 / Da execução à competição", "No dia a dia: diagnóstico e ajuste", [
                "Primeiro entendemos o que influenciou o erro",
                "Depois aplicamos uma correção segundo uma regra definida",
                "No código, essas responsabilidades aparecem em duas chamadas",
            ], "Pense em ajustar uma receita. Primeiro você identifica "
               "quais quantidades influenciaram o erro. Depois faz o ajuste "
               "seguindo uma regra. O backward calcula a sensibilidade; "
               "o step aplica a atualização. Essa separação ajuda a localizar "
               "o comportamento no código.", kind="analogy"))
        else:
            additions.append(Slide("08 / Da execução à competição", "No dia a dia: guardar os comprovantes", [
                "O caderno de ensaios fica separado da versão enviada",
                "Salvar um arquivo não significa que o destinatário o recebeu",
                "Precisamos do comprovante de processamento da plataforma",
            ], "Imagine preparar um documento para entregar. Você guarda "
               "as anotações e a versão final em lugares diferentes. "
               "Salvar a versão final no computador não prova que ela "
               "foi recebida. O processamento da submissão no Kaggle "
               "faz o papel do comprovante de recebimento.", kind="analogy"))
        used.append(path)
    index = next((i for i, slide in enumerate(slides) if slide.chapter.startswith("09")), len(slides))
    slides[index:index] = additions
    return used


def font(size: int, bold: bool = False):
    from PIL import ImageFont
    from matplotlib import font_manager

    family = "DejaVu Sans"
    path = font_manager.findfont(font_manager.FontProperties(family=family, weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def lines(text: str, face, width: int) -> list[str]:
    """Wrap on words; split a long URL if needed instead of clipping it."""
    wrapped = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            if current and face.getlength(current + " " + word) > width:
                wrapped.append(current)
                current = ""
            while face.getlength(word) > width:
                length = next(i for i in range(1, len(word) + 1) if face.getlength(word[:i]) > width) - 1
                if current:
                    wrapped.append(current)
                    current = ""
                wrapped.append(word[:length])
                word = word[length:]
            current = (current + " " + word).strip()
        if current:
            wrapped.append(current)
    return wrapped


def formula_image(slide: Slide):
    from PIL import Image
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    stream = io.BytesIO()
    with rc_context({"savefig.transparent": True}):
        math_to_image("$" + slide.formula + "$", stream, prop=FontProperties(size=30),
                      dpi=150, format="png", color=INK)
    stream.seek(0)
    image = Image.open(stream).convert("RGBA")
    image.thumbnail((WIDTH - 240, 180))
    return image


def render_slides(slides: list[Slide], output: Path, draft: bool) -> None:
    from PIL import Image, ImageDraw

    title_font, body_font, small_font = font(64, True), font(36), font(24)
    for index, slide in enumerate(slides, 1):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 8), fill=ACCENT)
        draw.text((110, 65), slide.chapter.upper(), fill=ACCENT, font=small_font)
        title_lines = lines(slide.title, title_font, WIDTH - 220)
        y = 140
        for line in title_lines:
            draw.text((110, y), line, fill=INK, font=title_font)
            y += 82
        y += 25
        body_width = WIDTH - 270
        # Dense references use a smaller type size, while remaining readable at 1080p.
        face = font(29) if slide.chapter == "Referências" else body_font
        line_height = 44 if slide.chapter == "Referências" else 52
        for point in slide.points:
            draw.ellipse((113, y + 17, 123, y + 27), fill=ACCENT)
            for line in lines(point, face, body_width):
                draw.text((150, y), line, fill=INK, font=face)
                y += line_height
            y += 22
        if slide.code:
            from PIL import ImageFont
            from matplotlib import font_manager
            code_path = font_manager.findfont("DejaVu Sans Mono")
            code_face = ImageFont.truetype(code_path, 29)
            code_lines = []
            for code_line in slide.code.splitlines():
                code_lines.extend(lines(code_line, code_face, WIDTH - 300) or [""])
            draw.rounded_rectangle((110, y, WIDTH - 110, y + len(code_lines) * 45 + 70), radius=10, fill="#e1e8df")
            y += 30
            for line in code_lines:
                draw.text((140, y), line, fill=INK, font=code_face)
                y += 45
            y += 60
            draw.text((110, y), slide.code_reference, fill=MUTED, font=small_font)
            y += 50
        if slide.visual in FLOWS:
            labels = FLOWS[slide.visual]
            gap, box_width = 58, (WIDTH - 220 - 58 * (len(labels) - 1)) // len(labels)
            for box_index, label in enumerate(labels):
                x = 110 + box_index * (box_width + gap)
                draw.rounded_rectangle((x, y + 15, x + box_width, y + 105), radius=10, outline=ACCENT, width=3)
                label_width = small_font.getlength(label)
                draw.text((x + (box_width - label_width) / 2, y + 42), label, fill=ACCENT, font=small_font)
                if box_index < len(labels) - 1:
                    draw.text((x + box_width + 17, y + 40), "→", fill=ACCENT, font=small_font)
            y += 140
        if slide.formula:
            math_image = formula_image(slide)
            # Preserve the formula as a separate image used by the HTML deck too.
            math_image.save(output / f"formula-{index:02}.png")
            if y + math_image.height > HEIGHT - 130:
                raise ValueError(f"Slide {index} não cabe: reduza o conteúdo antes de publicar.")
            x = (WIDTH - math_image.width) // 2
            image.paste(math_image, (x, y + 18), math_image)
            y += math_image.height + 18
        if y > HEIGHT - 110:
            raise ValueError(f"Slide {index} não cabe: reduza o conteúdo antes de publicar.")
        label = "RASCUNHO · EVIDÊNCIAS INCOMPLETAS" if draft else "FRAUDLAB · EVIDÊNCIAS DOCUMENTADAS"
        draw.line((110, HEIGHT - 95, WIDTH - 110, HEIGHT - 95), fill="#cbd3ca", width=2)
        draw.text((110, HEIGHT - 70), label, fill=MUTED, font=small_font)
        draw.text((WIDTH - 250, HEIGHT - 70), f"{index:02} / {len(slides):02}", fill=MUTED, font=small_font)
        image.save(output / f"slide-{index:02}.png", optimize=True)


def speech_chunks(text: str, max_chars: int = 170) -> list[str]:
    """Keep short, timed utterances so subtitle timestamps are measured, not guessed."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for sentence in sentences:
        current = ""
        for word in sentence.split():
            if current and len(current) + len(word) + 1 > max_chars:
                result.append(current)
                current = ""
            current = (current + " " + word).strip()
        if current:
            result.append(current)
    return result


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    sec, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02},{milliseconds:03}"


def word_captions(boundaries: list[dict], max_chars: int = 76, text: str = "") -> list[dict]:
    """Group service-timed words into readable cues without estimating their times."""
    captions = []
    current = None
    cursor = 0
    for boundary in boundaries:
        start = boundary["offset"] / 10_000_000
        end = start + boundary["duration"] / 10_000_000
        word = str(boundary["text"])
        position = text.find(word, cursor) if text else -1
        word_end = position + len(word)
        if position >= 0:
            while word_end < len(text) and not text[word_end].isalnum() and not text[word_end].isspace():
                word_end += 1
            word = text[position:word_end]
            cursor = word_end
        candidate = (text[current["source_start"]:word_end] if current and
                     current["source_start"] >= 0 and position >= 0 else
                     ((current["text"] + " " + word) if current else word))
        if current and (len(candidate) > max_chars or end - current["start"] > 5):
            captions.append(current)
            current = None
        if current is None:
            current = {"start": start, "end": end, "text": word, "source_start": position}
        else:
            current["text"] = candidate
            current["end"] = end
    if current:
        captions.append(current)
    for caption in captions:
        caption.pop("source_start")
    return captions


async def edge_segments(slides: list[Slide], output: Path, voice: str, rate: int) -> list[dict]:
    try:
        import aiohttp
        import edge_tts
    except ImportError as error:
        raise RuntimeError("Instale edge-tts para a voz neural ou escolha --speech-engine sapi explicitamente.") from error
    segments = []
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    for index, slide in enumerate(slides, 1):
        mp3_path = audio_dir / f"edge-{index:02}.mp3"
        metadata_path = audio_dir / f"edge-{index:02}.jsonl"
        wave_path = audio_dir / f"edge-{index:02}.wav"
        try:
            # Some Windows networks resolve Bing to IPv6 but reset the connection.
            family = socket.AF_INET if sys.platform == "win32" else socket.AF_UNSPEC
            async with aiohttp.TCPConnector(family=family) as connector:
                speech = edge_tts.Communicate(slide.narration, voice, rate=f"{rate * 10:+d}%",
                                             boundary="WordBoundary", connector=connector)
                await speech.save(str(mp3_path), str(metadata_path))
        except Exception as error:
            raise RuntimeError("A voz neural Edge falhou. Nenhum fallback foi aplicado. "
                               "Para usar a voz local, escolha --speech-engine sapi. "
                               f"Motivo: {type(error).__name__}: {error}") from error
        boundaries = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        cues = word_captions([item for item in boundaries if item.get("type") == "WordBoundary"], text=slide.narration)
        if not cues:
            raise RuntimeError("O serviço Edge não retornou tempos por palavra; revise antes de gerar legendas.")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3_path),
                        "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(wave_path)], check=True)
        segments.append({"slide": index, "path": str(wave_path), "captions": cues})
        print(f"Voz neural: slide {index}/{len(slides)}", flush=True)
    return segments


def sapi_segments(slides: list[Slide], output: Path, voice: str, rate: int) -> list[dict]:
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    utterances = []
    for slide_index, slide in enumerate(slides, 1):
        for chunk_index, text in enumerate(speech_chunks(slide.narration), 1):
            path = (audio_dir / f"{slide_index:02}-{chunk_index:02}.wav").resolve()
            utterances.append({"slide": slide_index, "text": text, "path": str(path)})
    request = output / "speech-request.json"
    write_json(request, {"voice": voice, "rate": rate, "utterances": utterances})
    helper = Path(__file__).with_name("narrate.ps1")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(helper), "-RequestPath", str(request.resolve())], check=True)
    segments = []
    for item in utterances:
        with wave.open(item["path"], "rb") as source:
            duration = source.getnframes() / source.getframerate()
        segments.append({**item, "captions": [{"start": 0, "end": duration, "text": item["text"]}]})
    return segments


def narrate(slides: list[Slide], output: Path, voice: str, rate: int, engine: str = "edge") -> list[dict]:
    if engine not in ("edge", "sapi"):
        raise ValueError("Unknown speech engine: " + engine)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg não encontrado no PATH.")
    segments = (asyncio.run(edge_segments(slides, output, voice, rate)) if engine == "edge"
                else sapi_segments(slides, output, voice, rate))
    # Slide changes follow measured PCM durations; cues follow measured utterances
    # (SAPI) or the service's word boundaries (Edge).
    captions, timeline = [], []
    elapsed = 0.0
    current_slide = 0
    with wave.open(str(output / "narration.wav"), "wb") as target:
        target.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        for item in segments:
            if item["slide"] != current_slide:
                if timeline:
                    timeline[-1]["duration"] = elapsed - timeline[-1]["start"]
                current_slide = item["slide"]
                timeline.append({"slide": current_slide, "start": elapsed})
            with wave.open(item["path"], "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 22050):
                    raise ValueError("Narration needs 22.05 kHz, mono, 16-bit PCM.")
                duration = source.getnframes() / source.getframerate()
                target.writeframes(source.readframes(source.getnframes()))
            for cue in item["captions"]:
                if cue["start"] < 0 or cue["end"] < cue["start"] or cue["end"] > duration + 0.5:
                    raise ValueError("Caption timing falls outside the rendered audio.")
                subtitle = "\n".join(speech_chunks(cue["text"], 42))
                captions.append(f"{len(captions) + 1}\n{srt_time(elapsed + cue['start'])} --> "
                                f"{srt_time(elapsed + min(duration, cue['end']))}\n{subtitle}\n")
            elapsed += duration
            silence = 0.20
            target.writeframes(b"\0\0" * round(22050 * silence))
            elapsed += silence
        if timeline:
            timeline[-1]["duration"] = elapsed - timeline[-1]["start"]
    (output / "captions.srt").write_text("\n".join(captions), encoding="utf-8")
    vtt = "WEBVTT\n\n" + "\n".join(
        re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", caption)
        for caption in captions
    )
    (output / "captions.vtt").write_text(vtt, encoding="utf-8")
    write_json(output / "timeline.json", timeline)
    return timeline


def encode_video(output: Path, timeline: list[dict]) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg não encontrado no PATH.")
    # Relative generated names avoid platform-dependent path quoting in concat files.
    concat = []
    for item in timeline:
        concat.extend((f"file 'slide-{item['slide']:02}.png'", f"duration {item['duration']:.6f}"))
    concat.append(f"file 'slide-{timeline[-1]['slide']:02}.png'")
    (output / "video-input.txt").write_text("\n".join(concat) + "\n", encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0",
        "-i", "video-input.txt", "-i", "narration.wav", "-vf", "fps=24", "-c:v", "libx264",
        "-preset", "veryfast", "-tune", "stillimage", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        "-t", f"{sum(item['duration'] for item in timeline):.6f}", "video.mp4",
    ], cwd=output, check=True)


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/public.json"))
    parser.add_argument("--metrics", type=Path, help="Selected run's metrics.json")
    parser.add_argument("--comparison", type=Path, action="append", default=[], help="Other executed runs; repeat as needed")
    parser.add_argument("--output", type=Path, default=Path("artifacts/presentation"))
    parser.add_argument("--draft", action="store_true", help="Mark incomplete evidence explicitly; never a final delivery")
    parser.add_argument("--video", action="store_true", help="Render PT-BR speech, measured captions and MP4")
    parser.add_argument("--speech-engine", choices=("edge", "sapi"), default="edge")
    parser.add_argument("--voice", help="Default: pt-BR-AntonioNeural (edge) or Microsoft Maria Desktop (sapi)")
    parser.add_argument("--rate", type=int, default=0, choices=range(-3, 4), help="Speech pace; Edge uses steps of 10 percent")
    args = parser.parse_args(argv)
    evidence = read_json(args.evidence)
    paths = ([args.metrics] if args.metrics else []) + args.comparison
    runs = [read_json(path) for path in paths]
    missing = validate_evidence(evidence, runs, args.draft)
    slides = create_slides(evidence, runs, args.draft)
    source_paths = insert_code_slides(slides, Path(__file__).resolve().parents[1], not args.draft)
    ensure_pairs(slides)
    args.output.mkdir(parents=True, exist_ok=True)
    render_slides(slides, args.output, args.draft)
    write_html(slides, args.output, args.draft)
    write_json(args.output / "script.json", {"draft": args.draft, "slides": [slide.public() for slide in slides]})
    voice = args.voice or ("pt-BR-AntonioNeural" if args.speech_engine == "edge" else "Microsoft Maria Desktop")
    timeline = narrate(slides, args.output, voice, args.rate, args.speech_engine) if args.video else []
    if args.video:
        encode_video(args.output.resolve(), timeline)
    manifest = {
        "status": "draft" if args.draft else "evidence-backed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slides": len(slides), "duration_seconds": sum(item["duration"] for item in timeline) or None,
        "video_generated": args.video, "speech_engine": args.speech_engine if args.video else None,
        "voice": voice if args.video else None,
        "subtitle_timing": ("Edge service word boundaries" if args.speech_engine == "edge"
                            else "measured utterance WAV durations") if args.video else None,
        "missing_evidence": missing,
        "inputs": {input_name(path): sha256(path) for path in [args.evidence] + paths + source_paths},
        "files": {path.name: sha256(path) for path in
                  [args.output / "slides.html", args.output / "script.json"] +
                  [args.output / f"slide-{index:02}.png" for index in range(1, len(slides) + 1)] +
                  [args.output / f"formula-{index:02}.png" for index, slide in enumerate(slides, 1) if slide.formula] +
                  ([args.output / name for name in ("video.mp4", "captions.srt", "captions.vtt")] if args.video else [])},
        "verification": {"rendered": True, "watched_in_full": False, "audio_reviewed": False},
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "slides": len(slides), "output": str(args.output),
                      "video_generated": args.video}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
