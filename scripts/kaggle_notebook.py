"""Export the frozen final protocol into an offline, unexecuted Kaggle notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pprint
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1fe34b540f3ea28276da652c3dd1de02726897f4"
COMMON = {
    "feature_set": "value_digits",
    "id_column": "id",
    "target": "is_fraud",
    "split": "stratified",
    "folds": 3,
    "seed": 42,
    "holdout_fraction": 0.15,
    "holdout_seed": 2026,
    "validation_note": "Synthetic IID rows without entity identifiers or chronological timestamp; recurrent hour-of-day only. No duplicate rows or train-test overlap. Independent 15 percent stratified holdout seed2026 remains untouched; all choices use development OOF.",
}
CONFIGS = {
    "glm": {
        **COMMON,
        "name": "dev-glm-values-s50",
        "model": "glm",
        "threads": 4,
        "device": "auto",
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
    },
    "lightgbm": {
        **COMMON,
        "name": "dev-lgbm-leaf4-values-s20",
        "model": "lightgbm",
        "threads": 6,
        "device": "auto",
        "iterations": 4000,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "num_leaves": 4,
        "min_child_samples": 100,
        "target_smoothing": 20.0,
        "patience": 150,
    },
    "mlp": {
        **COMMON,
        "name": "dev-mlp-te-values",
        "model": "mlp",
        "threads": 6,
        "device": "cuda",
        "target_smoothing": 20.0,
        "epochs": 20,
        "hidden_dims": [128, 64],
        "dropout": 0.1,
        "neural_learning_rate": 0.001,
        "weight_decay": 0.0001,
        "neural_numeric_transform": "quantile",
        "neural_categorical_transform": "target",
        "batch_size": 4096,
        "patience": 4,
    },
}
DATA_HASHES = {
    "train.csv": "87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5",
    "test.csv": "10f516959020357b565abaaa4fd25baae97fc2f536d0805695bb7cc15c1660f0",
    "sample_submission.csv": "13fc6ed8b73de5329884ec490f2a0b750cf29a7cda772b78cfa0cf486f6f35fa",
}


def git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True
    ).stdout


def protocol(path: Path | None) -> tuple[dict, dict, dict]:
    configs = {model: dict(config) for model, config in CONFIGS.items()}
    record = {
        "origin": "documented_local_protocol",
        "development_roc_auc": 0.9505188657556217,
        "holdout_roc_auc": 0.9485517227500961,
        "weights": [0.75, 0.25],
    }
    if path is None:
        return configs, DATA_HASHES, record
    raw = path.read_bytes()
    report = json.loads(raw)
    if (
        report.get("status") != "complete"
        or report.get("kind") != "independent_holdout"
    ):
        raise ValueError(
            "--protocol must point to a completed independent holdout report."
        )
    selection = report["selection"]
    if len(selection["sources"]) != 2 or len(selection["weights"]) != 2:
        raise ValueError("The final protocol must contain exactly GLM and LightGBM.")
    found = {}
    for source, weight in zip(selection["sources"], selection["weights"]):
        directory = Path(source.replace("\\", "/"))
        directory = directory if directory.is_absolute() else ROOT / directory
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        model = config["model"]
        if model not in ("glm", "lightgbm") or model in found:
            raise ValueError(
                "Unexpected or repeated source model in the final protocol."
            )
        active = {key: config[key] for key in CONFIGS[model]}
        if any(
            active[key] != expected
            for key, expected in COMMON.items()
            if key != "validation_note"
        ):
            raise ValueError(
                "Source run does not match the documented final split and feature protocol."
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", active["name"]):
            raise ValueError("Source run names must be portable directory names.")
        configs[model] = active
        found[model] = weight
    if any(
        not math.isfinite(found[model]) or abs(found[model] - expected) > 1e-12
        for model, expected in (("glm", 0.75), ("lightgbm", 0.25))
    ):
        raise ValueError("The final frozen weights must be 75% GLM and 25% LightGBM.")
    comparison = directory.parent / "dev-mlp-te-values" / "config.json"
    if comparison.is_file():
        config = json.loads(comparison.read_text(encoding="utf-8"))
        configs["mlp"] = {key: config[key] for key in CONFIGS["mlp"]}
    hashes = report["data_hashes"]
    if set(hashes) != set(DATA_HASHES) or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()
    ):
        raise ValueError(
            "Protocol data hashes must identify the three official CSV files."
        )
    record.update(
        origin="local_holdout_report",
        protocol_sha256=hashlib.sha256(raw).hexdigest(),
        development_roc_auc=selection["development_roc_auc"],
        holdout_roc_auc=report["roc_auc"],
    )
    return configs, hashes, record


def make_notebook(
    revision: str, configs: dict, hashes: dict, record: dict
) -> tuple[dict, dict[str, bytes]]:
    cells = []

    def cell(kind: str, text: str, metadata: dict | None = None) -> None:
        value = {
            "id": f"cell-{len(cells):02d}-{hashlib.sha256(text.encode()).hexdigest()[:8]}",
            "cell_type": kind,
            "metadata": metadata or {},
            "source": text.splitlines(True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell(
        "markdown",
        "# FraudLab · protocolo final reproduzível\n\n"
        "**Esta versão foi salva sem execução no Kaggle.** As células não contêm resultados de uma execução nesta plataforma. Os treinamentos locais e as submissões realmente processadas estão documentados no [repositório público e suas evidências](https://github.com/aikolopes-og/fraudlab). O ranking é uma observação datada e provisória.\n\n"
        "O fluxo abaixo reproduz o protocolo escolhido: desenvolver GLM e LightGBM em 85% dos rótulos, combinar com pesos congelados 75/25, avaliar uma vez os 15% reservados, refitar ambos em 100% e gerar o CSV final. A MLP com target encoding tem uma célula opcional de comparação e não entra no blend. Código MIT; dados sob os [termos da competição](https://www.kaggle.com/competitions/playground-series-s6e11/rules).",
    )
    cell(
        "markdown",
        f"## 1. Fonte fixada e execução offline\n\nRevisão [{revision}](https://github.com/aikolopes-og/fraudlab/tree/{revision}). Cada módulo abaixo é exportado com `git show` dessa revisão; os hashes conferem a integridade. Não há `pip install`, clone remoto ou importação de predições de terceiros. O notebook usa as bibliotecas já instaladas na imagem Kaggle; o preflight interrompe se forem incompatíveis. Ative GPU apenas se quiser executar a comparação neural opcional.",
    )
    cell("code", "from pathlib import Path\nPath('fraudlab').mkdir(exist_ok=True)\n")
    files = {}
    sources = (
        git_bytes("ls-tree", "-r", "--name-only", revision, "src/fraudlab")
        .decode()
        .splitlines()
    )
    for source in sources:
        if source.endswith(".py"):
            target = str(PurePosixPath(source).relative_to("src"))
            body = git_bytes("show", f"{revision}:{source}")
            files[target] = body
            cell(
                "code",
                f"%%writefile {target}\n" + body.decode("utf-8"),
                {
                    "fraudlab_source": {
                        "revision": revision,
                        "path": source,
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                },
            )
    manifest = {name: hashlib.sha256(body).hexdigest() for name, body in files.items()}
    cell(
        "code",
        "import hashlib\nimport importlib.metadata as package_metadata\nimport json\nimport re\nimport shutil\nimport subprocess\nimport sys\n\n"
        "SOURCE_SHA256 = " + pprint.pformat(manifest, width=100) + "\n"
        "for filename, expected in SOURCE_SHA256.items():\n"
        "    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename\n\n"
        "if shutil.which('git') is None:\n"
        "    raise RuntimeError('A imagem precisa do executável git para registrar a proveniência; nenhum acesso remoto será feito.')\n\n"
        "requirements = {'numpy': ((1, 24), 3), 'pandas': ((2, 0), 3),\n"
        "                'scikit-learn': ((1, 3), 2), 'lightgbm': ((4, 7), 5)}\n"
        "versions = {}\n"
        "for package, (minimum, upper_major) in requirements.items():\n"
        "    installed = package_metadata.version(package)\n"
        "    version = tuple(map(int, re.findall(r'\\d+', installed)[:2]))\n"
        "    if version < minimum or version[0] >= upper_major:\n"
        "        raise RuntimeError(f'{package} {installed} incompatível com esta revisão offline; use uma imagem Kaggle compatível.')\n"
        "    versions[package] = installed\n"
        "print(json.dumps(versions, indent=2))\n",
    )
    cell(
        "markdown",
        "## 2. Dados oficiais e partição independente\n\n"
        "Os dados são sintéticos; `time_of_day_hour` representa hora recorrente, não ordem cronológica. A inspeção local não encontrou IDs de entidade, valores ausentes, duplicatas de atributos ou sobreposição treino/teste. O pipeline registra esses testes novamente. A divisão estratificada usa 3 folds e seed 42 somente nos 85% de desenvolvimento; os 15% reservados usam seed 2026.\n\n"
        "`/kaggle/input` é somente leitura. Os CSVs são copiados para `/kaggle/temp`, pois o código congelado guarda o ledger do holdout ao lado dos dados. As cópias brutas não são gravadas em `/kaggle/working` nem incluídas no material fonte. As execuções são sequenciais para limitar RAM.",
    )
    cell(
        "code",
        "candidates = [Path('/kaggle/input/competitions/playground-series-s6e11'),\n"
        "              Path('/kaggle/input/playground-series-s6e11')]\n"
        "input_data = next((path for path in candidates if (path / 'train.csv').is_file()), None)\n"
        "if input_data is None:\n"
        "    raise FileNotFoundError('Adicione playground-series-s6e11 aos inputs do notebook.')\n"
        "data = Path('/kaggle/temp/fraudlab-s6e11/raw')\n"
        "data.mkdir(parents=True, exist_ok=True)\n"
        "EXPECTED_DATA_SHA256 = " + pprint.pformat(hashes, width=100) + "\n"
        "for filename, expected in EXPECTED_DATA_SHA256.items():\n"
        "    shutil.copyfile(input_data / filename, data / filename)\n"
        "    if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected:\n"
        "        raise ValueError(f'{filename}: versão diferente dos dados usados no protocolo documentado.')\n\n"
        "runs = Path('artifacts/runs')\n"
        "def run(*arguments):\n"
        "    subprocess.run([sys.executable, '-m', 'fraudlab', *map(str, arguments)], check=True)\n\n"
        "run('inspect', '--data', data, '--output', 'artifacts/profile.json')\n",
    )
    cell(
        "markdown",
        "## 3. Configurações usadas nos experimentos escolhidos\n\n"
        "As opções abaixo vêm dos dois devruns selecionados ou de seus padrões documentados. GLM: logística com logits de target encoding interno, one-hot e escalonamento; solver Newton-Cholesky fixado no módulo. LightGBM: 4 folhas, limite de 4.000 árvores e early stopping somente nos folds de desenvolvimento. `value_digits` cria logs, ciclos, razão e chaves numéricas/dígitos; imputação, escalonamento e codificações aprendidas são ajustados dentro de cada fold.\n\n"
        "As métricas desta nova execução podem diferir das locais conforme versões/hardware. Os registros locais documentaram OOF "
        f"{record['development_roc_auc']:.8f} e holdout {record['holdout_roc_auc']:.8f}; esses números não são outputs deste notebook salvo.",
    )
    cell(
        "code",
        "CONFIGS = " + pprint.pformat(configs, width=100, sort_dicts=False) + "\n"
        "FROZEN_WEIGHTS = [0.75, 0.25]\n"
        "def train_model(model):\n"
        "    configuration = {'data': str(data), 'output': str(runs), **CONFIGS[model]}\n"
        "    arguments = ['train']\n"
        "    for key, value in configuration.items():\n"
        "        if value is None:\n"
        "            continue\n"
        "        arguments.append('--' + key.replace('_', '-'))\n"
        "        arguments.extend(map(str, value if isinstance(value, list) else [value]))\n"
        "    run(*arguments)\n"
        "    return runs / configuration['name']\n\n"
        "glm_run = train_model('glm')\n"
        "lgbm_run = train_model('lightgbm')\n",
    )
    cell(
        "markdown",
        "## 4. Comparação neural opcional\n\n"
        "A MLP-TE foi treinada e avaliada localmente; não apresentou ganho consistente ao entrar na composição. Para reproduzir essa comparação, ative GPU e mude a opção abaixo. Ela usa as mesmas linhas de desenvolvimento, target encoding cross-fitted em 5 partições internas, transformação quantile-normal, camadas ReLU 128→64, dropout 0,1, AdamW e BCEWithLogitsLoss. O holdout permanece intocado e a MLP não altera os pesos nem entra no CSV final.",
    )
    cell(
        "code",
        "RUN_MLP_COMPARISON = False\n"
        "if RUN_MLP_COMPARISON:\n"
        "    import torch\n"
        "    if not torch.cuda.is_available():\n"
        "        raise RuntimeError('Ative uma GPU Kaggle para reproduzir a configuração neural CUDA.')\n"
        "    train_model('mlp')\n\n"
        "for model in ('glm', 'lightgbm', 'mlp'):\n"
        "    path = runs / CONFIGS[model]['name'] / 'metrics.json'\n"
        "    if path.is_file():\n"
        "        metric = json.loads(path.read_text())\n"
        "        print(model, {'development_oof_auc': metric['roc_auc'], 'fold_scores': metric['fold_scores']})\n",
    )
    cell(
        "markdown",
        "## 5. Composição congelada e holdout uma vez\n\n"
        "Os pesos 75% GLM / 25% LightGBM foram escolhidos anteriormente pela OOF de desenvolvimento; não são ajustados pelo ranking público nem pelo holdout. O comando compara IDs/folds, salva a seleção antes da primeira previsão independente e fecha o ledger por partição. Não remova o ledger para experimentar novos pesos.\n\n"
        "A avaliação independente usa a média dos 3 checkpoints de cada família: `0.75 × mean(GLM folds) + 0.25 × mean(LGB folds)`. Ela avalia esse protocolo de desenvolvimento; o refit final posterior tem 1 modelo de cada família e não recebe uma AUC independente inventada.",
    )
    cell(
        "code",
        "dev_blend = runs / 'dev-blend-selected'\n"
        "holdout_report = Path('artifacts/holdout/evaluation.json')\n"
        "run('blend', '--runs', glm_run, lgbm_run, '--weights', *FROZEN_WEIGHTS, '--name', dev_blend.name)\n"
        "run('evaluate-holdout', '--runs', glm_run, lgbm_run, '--weights', *FROZEN_WEIGHTS,\n"
        "    '--development-run', dev_blend, '--output', holdout_report, '--device', 'cpu')\n",
    )
    cell(
        "markdown",
        "## 6. Refit em 100% e arquivo final\n\n"
        "Após a avaliação, os hiperparâmetros e pesos permanecem congelados. O GLM é ajustado uma vez em todos os rótulos. O LGB usa a mediana do número de árvores escolhido nos folds de desenvolvimento, sem early stopping no holdout já avaliado. O registro local dessa regra produziu 1.438 árvores; esta reprodução registra a mediana efetivamente obtida na sua execução.\n\n"
        "O blend final exige os mesmos pesos da seleção e registra `phase=full_data_refit`, `roc_auc=null` e zero OOF. A OOF de desenvolvimento e o hash do relatório independente ficam em `selection`, separados. Nenhuma submissão é enviada automaticamente.",
    )
    cell(
        "code",
        "run('refit', '--run', glm_run, '--holdout-report', holdout_report, '--name', 'final-glm')\n"
        "run('refit', '--run', lgbm_run, '--holdout-report', holdout_report, '--name', 'final-lgbm')\n"
        "run('blend', '--runs', runs / 'final-glm', runs / 'final-lgbm', '--name', 'final-blend')\n"
        "final_csv = runs / 'final-blend' / 'submission.csv'\n"
        "run('validate-submission', '--file', final_csv, '--sample', data / 'sample_submission.csv')\n"
        "shutil.copyfile(final_csv, 'submission.csv')\n"
        "final_metadata = json.loads((runs / 'final-blend' / 'metrics.json').read_text())\n"
        "assert final_metadata['phase'] == 'full_data_refit' and final_metadata['roc_auc'] is None\n"
        "assert final_metadata['rows']['oof'] == 0\n"
        "assert hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest() == final_metadata['submission_sha256']\n"
        "print({'file': 'submission.csv', 'sha256': final_metadata['submission_sha256'],\n"
        "       'phase': final_metadata['phase'], 'selection': final_metadata['selection']})\n",
    )
    cell(
        "markdown",
        "## Evidência e limites\n\n"
        "O notebook publicado compartilha o código e o protocolo; esta versão não foi executada no Kaggle. As métricas locais, o arquivo exato submetido e os scores/posições processados são comprovados pelos registros do [repositório](https://github.com/aikolopes-og/fraudlab). Salvar um notebook, gerar um CSV e receber uma pontuação da competição são estados diferentes.\n\n"
        "Use `submission.csv` somente depois de conferir as regras e o processamento pela plataforma. O score público mede o arquivo enviado e não é a OOF nem a avaliação do holdout; a classificação final privada só existe após o encerramento. Esta reprodução não faz otimização sobre rótulos do teste, reconstrução do split público ou combinação de arquivos de terceiros.",
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "fraudlab": {
                "source_revision": revision,
                "source_sha256": manifest,
                "protocol": record,
                "execution_status": "not_executed_on_kaggle",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, files


def verify(notebook: dict, sources: dict[str, bytes]) -> dict:
    exported = {}
    compiled = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if cell["execution_count"] is not None or cell["outputs"]:
            raise ValueError("Generated notebooks must have no execution history.")
        text = "".join(cell["source"])
        if text.startswith("%%writefile "):
            heading, body = text.split("\n", 1)
            filename = heading.removeprefix("%%writefile ")
            exported[filename] = body.encode("utf-8")
            compile(body, filename, "exec")
        else:
            compile(text, cell["id"], "exec")
            compiled += 1
    if exported != sources:
        raise ValueError("Notebook module export does not match the pinned Git blobs.")
    content = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    if re.search(r"[A-Za-z]:[\\/]+Users[\\/]+", content):
        raise ValueError("A personal absolute Windows path leaked into the notebook.")
    return {
        "exported_modules": len(exported),
        "compiled_non_magic_cells": compiled,
        "unexecuted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", default=REVISION, help="Full published Git commit SHA to export"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Optional local holdout report with the selected source runs",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full Git commit SHA")
    configs, hashes, record = protocol(args.protocol)
    notebook, sources = make_notebook(args.revision, configs, hashes, record)
    checks = verify(notebook, sources)
    output = ROOT / "notebooks"
    output.mkdir(exist_ok=True)
    (output / "fraudlab.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata_path = output / "kernel-metadata.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.is_file()
        else {
            "id": "aikoog/fraudlab-reproducible-neural-fraud-detection",
            "title": "FraudLab - Reproducible Neural Fraud Detection",
            "is_private": False,
            "enable_gpu": False,
        }
    )
    metadata.update(
        code_file="fraudlab.ipynb",
        language="python",
        kernel_type="notebook",
        enable_internet=False,
        dataset_sources=[],
        competition_sources=["playground-series-s6e11"],
        kernel_sources=[],
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "revision": args.revision,
                "notebook": "notebooks/fraudlab.ipynb",
                **checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
