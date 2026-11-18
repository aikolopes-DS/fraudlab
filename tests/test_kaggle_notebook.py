"""Public-code export contracts; no training or cloud execution is performed."""

import ast
import hashlib
import importlib.util
import json
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
    allowed_calls = {"Path", "mkdir", "sha256", "read_bytes", "hexdigest", "print", "dumps", "len", "items"}
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
                assert set(modules) <= {"pathlib", "hashlib", "json"}
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
