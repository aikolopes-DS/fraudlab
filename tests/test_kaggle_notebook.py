"""Public-code export contracts; no training or cloud execution is performed."""

import ast
import base64
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_notebook", ROOT / "scripts/kaggle_notebook.py")
notebook_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notebook_export)


def export_current():
    cells = []
    record, files = notebook_export.current_source_appendix(
        lambda kind, text, metadata=None: cells.append((kind, text, metadata or {}))
    )
    return cells, record, files


def test_current_code_is_pinned_and_cannot_replace_baseline_imports():
    cells, record, files = export_current()
    assert record["core_revision"] == "11b139f77206101d3f5beeb5648fb459f4cd304a"
    assert record["script_revision"] == "c95ff468619d299c94764b45e5e15af5ed9f4674"
    assert len(files) == 27
    assert all(path.startswith("r23-source/") for path in files)
    assert sum(path.startswith("r23-source/src/fraudlab/") for path in files) == 23
    assert sum(path.startswith("r23-source/scripts/") for path in files) == 2
    for path, body in files.items():
        provenance = record["source_files"][path]
        assert hashlib.sha256(body).hexdigest() == provenance["sha256"]
        assert body == notebook_export.git_bytes("show", provenance["revision"] + ":" + provenance["path"])


def test_current_appendix_only_writes_and_checks_files():
    cells, record, _ = export_current()
    allowed_calls = {"Path", "mkdir", "sha256", "read_bytes", "hexdigest", "print", "dumps", "len", "items",
                     "b64decode", "loads", "decompress", "write_bytes", "encode", "is_absolute"}
    for kind, text, metadata in cells:
        if kind != "code" or text.startswith("%%writefile "):
            continue
        assert metadata["fraudlab_scope"] == "current_source_export"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                assert name in allowed_calls, name
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert set(modules) <= {"pathlib", "hashlib", "json", "base64", "gzip"}
    assert record["observed_local_result"]["local_auc"] is None
    assert record["effective_weights"] == {"offset": 0.421875, "realmlp": 0.140625,
                                            "lightgbm": 0.1875, "external_gpt2": 0.25}


def test_v7_preserves_executable_baseline_and_has_empty_execution_history():
    recipe = json.loads((ROOT / "notebooks/hybrid-recipe.json").read_text(encoding="utf-8"))
    record = {"origin": "portable_hybrid_recipe", "phase": recipe["phase"], "recipe": recipe}
    with patch.object(notebook_export, "tokenizer_assets", return_value=[]):
        notebook, files = notebook_export.make_notebook(recipe["source_revision"], {}, recipe["data_hashes"], record)
    assert notebook["metadata"]["fraudlab"]["source_sha256"] == recipe["source_sha256"]
    assert len([name for name in files if name.startswith("fraudlab/")]) == 22
    assert notebook["metadata"]["fraudlab"]["execution_status"] == "not_executed_on_kaggle"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
    check = notebook_export.verify(notebook, files)
    assert check["exported_modules"] == 45
    assert check["exported_files"] == 49
    assert check["unexecuted"] is True
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_MLP_COMPARISON" not in text
    assert "portable_selected_cv_refit" in text
    assert "não é um comando autossuficiente para um clone vazio" in text


def test_corrupted_current_export_is_rejected():
    recipe = json.loads((ROOT / "notebooks/hybrid-recipe.json").read_text(encoding="utf-8"))
    with patch.object(notebook_export, "tokenizer_assets", return_value=[]):
        notebook, files = notebook_export.make_notebook(
            recipe["source_revision"], {}, recipe["data_hashes"], {"phase": recipe["phase"], "recipe": recipe}
        )
    corrupted = dict(files)
    corrupted["r23-source/src/fraudlab/realmlp.py"] = b"corrupted\n"
    try:
        notebook_export.verify(notebook, corrupted)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("The current-source export must match its pinned bytes.")


def test_real_tokenizer_export_and_cli_payload_fit_kaggle_limit():
    """Use committed official assets, not the empty-assets mock used in unit tests."""
    notebook = json.loads((ROOT / "notebooks/fraudlab.ipynb").read_text(encoding="utf-8"))
    tokenizer_cell = next(cell for cell in notebook["cells"]
                          if "".join(cell["source"]).startswith("TOKENIZER_ASSETS = "))
    assignment = ast.parse("".join(tokenizer_cell["source"])).body[0]
    assets = ast.literal_eval(assignment.value)
    recipe = json.loads((ROOT / "notebooks/hybrid-recipe.json").read_text(encoding="utf-8"))
    assert len(assets) == len(recipe["tokenizer"]["encoding_files"]) == 2
    for asset, official in zip(assets, recipe["tokenizer"]["encoding_files"]):
        body = lzma.decompress(base64.b64decode(asset["lzma_base64"], validate=True))
        assert len(body) > 100_000
        assert hashlib.sha256(body).hexdigest() == asset["sha256"] == official["sha256"]
        assert asset["url"] == official["url"]
    metadata = notebook["metadata"]["fraudlab"]
    sources = {name: notebook_export.git_bytes("show", metadata["source_revision"] + ":src/" + name)
               for name in metadata["source_sha256"]}
    appendix = metadata["protocol"]["current_appendix"]
    sources.update({name: notebook_export.git_bytes("show", item["revision"] + ":" + item["path"])
                    for name, item in appendix["source_files"].items()})
    report = notebook_export.verify(notebook, sources)
    assert report["exported_files"] == 49
    assert report["transmitted_bytes"] < 1_000_000


def test_source_limit_is_enforced_before_delivery():
    notebook = {"cells": [{"cell_type": "markdown", "source": ["x" * 1_000_000], "metadata": {}}]}
    try:
        notebook_export.verify(notebook, {})
    except ValueError as error:
        assert "exceeds Kaggle limit" in str(error)
    else:
        raise AssertionError("An oversized source must be rejected locally.")


def test_archive_extracts_the_exact_sources_without_importing_them(tmp_path, monkeypatch):
    notebook = json.loads((ROOT / "notebooks/fraudlab.ipynb").read_text(encoding="utf-8"))
    expected = {}
    monkeypatch.chdir(tmp_path)
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if text.startswith("%%writefile "):
            heading, body = text.split("\n", 1)
            path = Path(heading.removeprefix("%%writefile "))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body.encode("utf-8"))
            expected[str(path).replace("\\", "/")] = body.encode("utf-8")
        elif cell.get("metadata", {}).get("fraudlab_archive"):
            expected.update(notebook_export.archived_sources(cell))
            # This reviewed cell only imports the stdlib and extracts bytes; never source modules.
            exec(compile(text, "source-archive", "exec"), {})
    actual = {str(path).replace("\\", "/"): path.read_bytes()
              for path in Path(".").rglob("*") if path.is_file()}
    assert len(actual) == 49
    assert actual == expected


def test_official_tokenizer_assets_load_offline_after_lzma_unpack(tmp_path, monkeypatch):
    import tiktoken
    import tiktoken.load
    import tiktoken.registry

    notebook = json.loads((ROOT / "notebooks/fraudlab.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if "".join(cell["source"]).startswith("TOKENIZER_ASSETS = "))
    assets = ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)
    for asset in assets:
        body = lzma.decompress(base64.b64decode(asset["lzma_base64"], validate=True))
        (tmp_path / asset["cache_key"]).write_bytes(body)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})

    def refuse_network(path):
        raise AssertionError("The embedded tokenizer must not download: " + path)

    monkeypatch.setattr(tiktoken.load, "read_file", refuse_network)
    encoding = tiktoken.get_encoding("gpt2")
    assert encoding.max_token_value + 1 == 50257
    text = "FraudLab: 123.45"
    assert encoding.decode(encoding.encode(text)) == text
