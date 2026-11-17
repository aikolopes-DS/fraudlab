"""Export the frozen final protocol into an offline, unexecuted Kaggle notebook."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import pprint
import re
import subprocess
import tempfile
from textwrap import dedent
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


def tokenizer_assets(recipe: dict) -> list[dict]:
    """Embed verified official assets already cached locally; never download here."""
    cache = Path(
        os.environ.get("TIKTOKEN_CACHE_DIR")
        or os.environ.get("DATA_GYM_CACHE_DIR")
        or str(Path(tempfile.gettempdir()) / "data-gym-cache")
    )
    assets = []
    for spec in recipe["tokenizer"]["encoding_files"]:
        key = hashlib.sha1(spec["url"].encode()).hexdigest()
        body = (cache / key).read_bytes()
        if hashlib.sha256(body).hexdigest() != spec["sha256"]:
            raise ValueError("Cached official tokenizer asset differs: " + spec["url"])
        assets.append(
            {
                **spec,
                "cache_key": key,
                "gzip_base64": base64.b64encode(gzip.compress(body, mtime=0)).decode(),
            }
        )
    return assets


def tokenizer_cell(recipe: dict) -> str:
    return (
        "TOKENIZER_ASSETS = "
        + pprint.pformat(tokenizer_assets(recipe), width=100)
        + "\n"
        + dedent("""
        import base64
        import gzip
        import os

        for package, expected in {'tiktoken': '0.14.0', 'regex': '2026.9.29'}.items():
            installed = package_metadata.version(package)
            if installed != expected:
                raise RuntimeError(f'{package}=={expected} is required offline; found {installed}.')
        tokenizer_cache = Path('/kaggle/temp/fraudlab-tokenizer-cache')
        tokenizer_cache.mkdir(parents=True, exist_ok=True)
        for asset in TOKENIZER_ASSETS:
            body = gzip.decompress(base64.b64decode(asset['gzip_base64']))
            assert hashlib.sha256(body).hexdigest() == asset['sha256']
            assert hashlib.sha1(asset['url'].encode()).hexdigest() == asset['cache_key']
            (tokenizer_cache / asset['cache_key']).write_bytes(body)
        os.environ['TIKTOKEN_CACHE_DIR'] = str(tokenizer_cache.resolve())
        from fraudlab.bpe import _encoding
        encoding = _encoding()
        assert encoding.name == 'gpt2' and encoding.max_token_value + 1 == 50257
        print({'offline_tokenizer': encoding.name, 'verified_assets': len(TOKENIZER_ASSETS)})
        """)
    )


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
    hybrid = record.get("phase") == "exploratory_full_data_cv"

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
        (
            "# FraudLab · R18 em CV e refit completo R20\n\n"
            if hybrid
            else "# FraudLab · protocolo próprio reproduzível\n\n"
        )
        + "**Esta versão foi salva sem execução no Kaggle.** As células não contêm resultados de uma execução nesta plataforma. Os treinamentos locais e as submissões realmente processadas estão documentados no [repositório público e suas evidências](https://github.com/aikolopes-DS/fraudlab). O ranking é uma observação datada e provisória.\n\n"
        + (
            "O fluxo reproduz a receita selecionada R18: offset próprio com BPE, frequência, pooling e oito interações, além do LightGBM próprio. Executa CV estratificada 5/42 e, depois, dois refits em todos os rótulos conforme a R20. Pesos fixos em probabilidades: **56,25% offset / 18,75% LGB / 25% GPT-2 de Masaya Kawamata**. O GPT-2 foi treinado pelo autor externo. A R20 não fabrica OOF nem AUC de refit; a reserva antiga não é independente destes protocolos.\n\nCódigo próprio MIT; previsões externas v2 CC BY 4.0. Tokenizador e vocabulários oficiais GPT-2: OpenAI. Dados sob os termos da competição."
            if hybrid
            else "O fluxo abaixo reproduz o protocolo escolhido: desenvolver GLM e LightGBM em 85% dos rótulos, combinar com pesos congelados 75/25, avaliar uma vez os 15% reservados, refitar ambos em 100% e gerar o CSV final. A MLP com target encoding tem uma célula opcional de comparação e não entra no blend. Código MIT; dados sob os termos da competição."
        ),
    )
    cell(
        "markdown",
        f"## 1. Fonte fixada e execução offline\n\nRevisão [{revision}](https://github.com/aikolopes-DS/fraudlab/tree/{revision}). Cada módulo abaixo é exportado com `git show` dessa revisão; os hashes conferem a integridade. Não há `pip install`, clone remoto ou execução de código de treinamento de terceiros. O notebook usa as bibliotecas já instaladas na imagem Kaggle; o preflight interrompe se forem incompatíveis. "
        + (
            "Os 22 módulos vêm do core 8a1c7fd. Os dois vocabulários oficiais estão incorporados, comprimidos e fixados por SHA-256; o cache é preparado antes da tokenização, sem rede. tiktoken 0.14.0 e regex 2026.9.29 são obrigatórios. As previsões externas vêm de um input v2 com hashes fixados. Não é necessária GPU."
            if hybrid
            else "Ative GPU apenas se quiser executar a comparação neural opcional."
        ),
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
    if hybrid and (
        revision != record["recipe"]["source_revision"]
        or manifest != record["recipe"]["source_sha256"]
        or len(manifest) != 22
    ):
        raise ValueError(
            "The canonical R18/R20 notebook requires all 22 pinned core modules."
        )
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
    if hybrid:
        if "fraudlab/external.py" not in files:
            raise ValueError(
                "The hybrid notebook requires a published source revision containing external.py."
            )
        cell("code", tokenizer_cell(record["recipe"]))
        hybrid_cells(cell, record["recipe"])
        return document(cells, revision, manifest, record), files
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
        "O notebook publicado compartilha o código e o protocolo; esta versão não foi executada no Kaggle. As métricas locais, o arquivo exato submetido e os scores/posições processados são comprovados pelos registros do [repositório](https://github.com/aikolopes-DS/fraudlab). Salvar um notebook, gerar um CSV e receber uma pontuação da competição são estados diferentes.\n\n"
        "Use `submission.csv` somente depois de conferir as regras e o processamento pela plataforma. O score público mede o arquivo enviado e não é a OOF nem a avaliação do holdout; a classificação final privada só existe após o encerramento. Esta reprodução não faz otimização sobre rótulos do teste, reconstrução do split público ou combinação de arquivos de terceiros.",
    )
    return document(cells, revision, manifest, record), files


def document(cells: list, revision: str, manifest: dict, record: dict) -> dict:
    return {
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


def hybrid_cells(cell, recipe: dict) -> None:
    public = recipe["public_component"]
    cell(
        "markdown",
        "## 2. Inputs e créditos\n\n"
        "Adicione a competição S6E11 e o dataset GPT-2 **versão 2** aos inputs. "
        "Os dados e arrays são verificados por SHA-256; não há download durante a execução. "
        f"[Masaya Kawamata]({public['predictions_source_url']}) treinou o GPT-2 externo; "
        f"previsões CC BY 4.0 e [código MIT]({public['training_code_url']}) "
        f"no commit `{public['training_code_commit']}`. "
        "Método target-first: Chris Deotte; GReaT: Borisov et al.; dados originais: uditjain13. "
        "Os arrays são logits: no teste o autor fez a média dos cinco logits antes da sigmoid. "
        "O importador aprovado conserva conversão, autoria e hashes.\n\n"
        "BPE e pooling usam o vocabulário OpenAI, sem treinar a rede GPT-2. Pooling: "
        "crédito conceitual Megayak/Goodpjw2008; implementação própria e gerador não verificado.",
    )
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cell(
        "code",
        "RECIPE = " + pprint.pformat(recipe, width=100, sort_dicts=False) + "\n"
        f"RECIPE_SHA256 = {recipe_hash!r}\n"
        "EXPORT = "
        + pprint.pformat(
            {
                "core_revision": recipe["source_revision"],
                "generator_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "origin": "portable R18/R20 notebook cells; not the local artifact-audit script",
                "checkpoint_byte_identity_to_local_runs_claimed": False,
            },
            width=100,
        )
        + "\n"
        + dedent(
            """
        candidates = [Path('/kaggle/input/competitions/playground-series-s6e11'),
                      Path('/kaggle/input/playground-series-s6e11')]
        data = next((p for p in candidates if (p / 'train.csv').is_file()), None)
        if data is None:
            raise FileNotFoundError('Adicione playground-series-s6e11 aos inputs.')
        manifest_hash = RECIPE['public_component']['manifest_sha256']
        public_source = next((p.parent for p in Path('/kaggle/input').rglob('manifest.json')
                              if hashlib.sha256(p.read_bytes()).hexdigest() == manifest_hash), None)
        if public_source is None:
            raise FileNotFoundError('Adicione o dataset GPT-2 versão 2 aprovado.')

        def digest(path):
            result = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    result.update(block)
            return result.hexdigest()

        def check_inputs():
            current = hashlib.sha256(json.dumps(
                RECIPE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert current == RECIPE_SHA256, 'Recipe changed'
            for filename, expected in SOURCE_SHA256.items():
                assert digest(filename) == expected, filename
            for filename, expected in RECIPE['data_hashes'].items():
                assert digest(data / filename) == expected, filename
            assert digest(public_source / 'manifest.json') == manifest_hash
            for filename, expected in RECIPE['public_component']['array_sha256'].items():
                assert digest(public_source / filename) == expected, filename
            for asset in TOKENIZER_ASSETS:
                assert digest(tokenizer_cache / asset['cache_key']) == asset['sha256']

        check_inputs()
        import fraudlab
        assert Path(fraudlab.__file__).resolve() == Path('fraudlab/__init__.py').resolve()
        runs = Path('artifacts/runs')
        def run(*arguments):
            subprocess.run([sys.executable, '-m', 'fraudlab', *map(str, arguments)], check=True)

        run('inspect', '--data', data, '--output', 'artifacts/profile.json')
        public_run = runs / RECIPE['public_component']['run']
        run('import-public-gpt2', '--source', public_source, '--data', data,
            '--output', runs, '--name', public_run.name)
        """
        ),
    )
    cell(
        "markdown",
        "## 3. R18: configurações fixas e validação dos argumentos\n\n"
        "Offset: `joint_bpe` + frequência do treino + pooling, 92 colunas e 223 dimensões "
        "codificadas. GLM Newton-Cholesky C=0,3, TE interno 5/50 e pooling 50/2.000/200. "
        "Residual: 1.000 árvores, quatro folhas, taxa 0,03, L2=5 e mínimo 100 por folha. "
        "LightGBM separado: `value_digits`, 1.438 árvores, quatro folhas, taxa 0,04 e TE20. "
        "CV estratificada 5/42 em todos os rótulos, sem holdout independente, dados originais "
        "adicionais, frequência transdutiva ou early stopping externo. "
        "O preflight abaixo valida a CLI antes dos treinos.",
    )
    cell(
        "code",
        dedent("""
        from fraudlab.__main__ import parser, validate_training_config

        def training_arguments(role):
            configuration = {'data': str(data), 'output': str(runs),
                             **RECIPE['common'], **RECIPE['models'][role]}
            arguments = ['train']
            for key, value in configuration.items():
                if value is None or value is False:
                    continue
                arguments.append('--' + key.replace('_', '-'))
                if value is not True:
                    arguments.extend(map(str, value if isinstance(value, list) else [value]))
            return arguments

        for role in ('offset', 'lightgbm'):
            config = vars(parser().parse_args(training_arguments(role)))
            config.pop('command')
            if config['patience'] is None:
                config['patience'] = 100
            validate_training_config(config)
            assert config['fixed_training'] and config['folds'] == 5
            assert config['seed'] == 42 and config['holdout_fraction'] == 0
            assert not config.get('original_data') and not config.get('glm_transductive_frequency')
        assert [RECIPE['effective_weights'][r] for r in ('offset', 'lightgbm', 'external_gpt2')] == [0.5625, 0.1875, 0.25]
        print({'validated_models': list(RECIPE['models']), 'fit_performed_by_this_cell': False})
        """),
    )
    cell(
        "markdown",
        "## 4. Executar a CV R18 e conservar seu CSV\n\n"
        "Os dez ajustes são sequenciais. Cada OOF vem do treino que excluiu seu fold; "
        "o teste usa a média de cinco probabilidades por família. A composição é direta: "
        "**56,25% offset / 18,75% LGB / 25% Masaya**, em probabilidades. "
        "A OOF local histórica selecionada foi 0,9516524825255244; ela não é output deste "
        "notebook salvo. Esta execução produzirá seus próprios artefatos e hashes.",
    )
    cell(
        "code",
        dedent("""
        check_inputs()
        run(*training_arguments('offset'))
        run(*training_arguments('lightgbm'))
        offset_run = runs / RECIPE['models']['offset']['name']
        lgbm_run = runs / RECIPE['models']['lightgbm']['name']
        hybrid_run = runs / RECIPE['cv_hybrid_name']
        weights = [RECIPE['effective_weights'][role]
                   for role in ('offset', 'lightgbm', 'external_gpt2')]
        run('blend', '--runs', offset_run, lgbm_run, public_run,
            '--weights', *weights, '--output', runs, '--name', hybrid_run.name)
        run('validate-submission', '--file', hybrid_run / 'submission.csv',
            '--sample', data / 'sample_submission.csv')
        shutil.copyfile(hybrid_run / 'submission.csv', 'submission-cv.csv')
        cv_metric = json.loads((hybrid_run / 'metrics.json').read_text())
        assert cv_metric['phase'] == 'exploratory_full_data_cv'
        assert cv_metric['validation']['independent_holdout'] is False
        assert cv_metric['rows']['oof'] == RECIPE['rows']['train']
        component = cv_metric['model_details']['external_components']
        assert len(component) == 1 and component[0]['trained_by_us'] is False
        assert component[0]['weight'] == 0.25
        assert component[0]['predictions_source_version'] == 2
        check_inputs()
        CV_ARTIFACT_SHA256 = {
            str(p / name): digest(p / name)
            for p in (offset_run, lgbm_run, public_run, hybrid_run)
            for name in ('config.json', 'metrics.json', 'oof.csv', 'predictions.csv')}
        print({'cv_file': 'submission-cv.csv', 'sha256': digest('submission-cv.csv'),
               'this_execution_oof_auc': cv_metric['roc_auc']})
        """),
    )
    cell(
        "markdown",
        "## 5. R20 portátil: refits em 100%, sem nova OOF\n\n"
        "Dois novos ajustes usam todos os 699.960 rótulos, seed42 e os mesmos parâmetros. "
        "O meta não existe neste protocolo: R21 é outra pesquisa e não é executada aqui. "
        "A validação passada ao fit contém uma linha de teste e `validation_y=None`; "
        "não seleciona árvores. Encodings supervisionados permanecem cruzados no treino.\n\n"
        "Esta implementação portátil usa `fit_predict` do core8a1 e guarda os hashes "
        "**desta execução**, sem exigir ou alegar identidade com os checkpoints locais. "
        "O script local auditado usa outro contrato de proveniência. Cada checkpoint novo "
        "é recarregado em 1.024 testes; falha de convergência interrompe, sem retry. "
        "Metadados: `full_data_refit`, AUC nula, folds vazios, zero OOF. "
        "A AUC da CV acima aparece somente como referência de desenvolvimento.",
    )
    cell(
        "code",
        dedent("""
        import gc
        import warnings
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.exceptions import ConvergenceWarning
        from fraudlab.data import load_data, submission, validate_submission, write_json
        from fraudlab.experiment import create_run, source_provenance, versions
        from fraudlab.features import transform_features
        from fraudlab.models import fit_predict, predict_saved

        raw, raw_test, sample = load_data(data, 'id', 'is_fraud')
        assert len(raw) == RECIPE['rows']['train'] and len(raw_test) == RECIPE['rows']['test']
        labels = raw['is_fraud'].to_numpy()
        reference = json.loads((hybrid_run / 'metrics.json').read_text())
        assert reference['status'] == 'complete' and reference['data_hashes'] == RECIPE['data_hashes']
        assert reference['model_details']['weights'] == weights
        full_common = {
            'phase': 'full_data_refit', 'roc_auc': None, 'fold_scores': [], 'fold_std': None,
            'rows': {'train': len(raw), 'test': len(raw_test), 'oof': 0},
            'data_hashes': RECIPE['data_hashes'], 'seed': 42, 'versions': versions(),
            'development_roc_auc': reference['roc_auc'],
            'development_roc_auc_scope': 'This execution R18 exploratory CV; not an AUC of refits',
            'reference_cv_run': hybrid_run.name,
            'reference_cv_metrics_sha256': digest(hybrid_run / 'metrics.json'),
            'notebook_export': {**EXPORT, 'recipe_sha256': RECIPE_SHA256},
        }

        def full_refit(role, source):
            check_inputs()
            for filename in ('config.json', 'metrics.json'):
                assert digest(source / filename) == CV_ARTIFACT_SHA256[str(source / filename)]
            original = json.loads((source / 'config.json').read_text())
            source_metric = json.loads((source / 'metrics.json').read_text())
            assert original['model'] == role and original['fixed_training']
            assert original['phase'] == 'exploratory_full_data_cv'
            assert not original.get('original_data') and not original.get('glm_transductive_frequency')
            name = RECIPE['refit']['names'][role]
            config = {**original, 'name': name, 'phase': 'full_data_refit', 'refit': True,
                      'validation_note': 'Portable fixed all-label refit; no new OOF or holdout'}
            execution_keys = {'name', 'phase', 'refit', 'validation_note'}
            assert {k: v for k, v in config.items() if k not in execution_keys} == {
                k: v for k, v in original.items() if k not in execution_keys}
            directory = create_run(runs, name)
            write_json(directory / 'status.json', {'status': 'running'})
            write_json(directory / 'config.json', config)
            try:
                training = transform_features(raw.drop(columns=['id', 'is_fraud']),
                                              config['feature_set'])
                testing = transform_features(raw_test.drop(columns='id'), config['feature_set'])
                assert training.columns.tolist() == testing.columns.tolist() == source_metric['features']['names']
                stem = directory / 'model'
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always', ConvergenceWarning)
                    _, prediction, detail = fit_predict(
                        role, training, labels, testing.iloc[:1], None, testing, config, 42, stem)
                for warning in captured:
                    warnings.warn(warning.message, warning.category)
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    raise RuntimeError('Refit did not converge; no automatic retry')
                assert detail['best_iteration'] == config['iterations']
                checkpoint_hash = digest(stem.with_suffix('.joblib'))
                if role == 'offset':
                    classifier = joblib.load(stem.with_suffix('.joblib')).glm.named_steps['classifier']
                    assert int(classifier.n_iter_[0]) < classifier.max_iter
                    assert detail['encoded_features'] == 223
                replay = predict_saved(role, stem, testing.iloc[:1024], device='cpu')
                error = float(np.max(np.abs(replay - prediction[:1024])))
                assert np.allclose(replay, prediction[:1024], rtol=0, atol=1e-12)
                assert digest(stem.with_suffix('.joblib')) == checkpoint_hash
                final = submission(raw_test['id'], prediction, sample, 'is_fraud')
                for filename in ('predictions.csv', 'submission.csv'):
                    final.to_csv(directory / filename, index=False)
                validate_submission(pd.read_csv(directory / 'submission.csv'), sample, 'id', 'is_fraud')
                check_inputs()
                for filename in ('config.json', 'metrics.json'):
                    assert digest(source / filename) == CV_ARTIFACT_SHA256[str(source / filename)]
                metric = {**full_common, **source_provenance(config['feature_set']),
                          'status': 'complete', 'model': role,
                          'features': source_metric['features'], 'model_details': detail,
                          'source_cv_run': source.name,
                          'source_cv_config_sha256': digest(source / 'config.json'),
                          'config_sha256': digest(directory / 'config.json'),
                          'checkpoint_sha256': checkpoint_hash, 'reload_rows': len(replay),
                          'reload_max_abs_error': error,
                          'external_components': [],
                          'submission_sha256': digest(directory / 'submission.csv')}
                assert metric['roc_auc'] is None and metric['fold_scores'] == []
                assert metric['rows']['oof'] == 0 and not (directory / 'oof.csv').exists()
                write_json(directory / 'metrics.json', metric)
                write_json(directory / 'status.json', {'status': 'complete'})
                return directory
            except Exception as error:
                write_json(directory / 'status.json', {'status': 'failed', 'error': str(error)})
                raise
        """),
    )
    cell(
        "code",
        dedent("""
        refit_offset = full_refit('offset', offset_run)
        gc.collect()
        refit_lgbm = full_refit('lightgbm', lgbm_run)
        gc.collect()
        """),
    )
    cell(
        "markdown",
        "## 6. R20: composição final e manifesto desta execução\n\n"
        "O blend geral continua recusando fases mistas. Esta composição especializada "
        "combina os dois novos refits completos com as previsões externas aprovadas, "
        "valida os IDs e probabilidades e mantém os pesos 56,25/18,75/25. "
        "Gera `submission.csv` e `notebook-execution-manifest.json`, sem envio automático "
        "e sem nova AUC de refit. O arquivo CV permanece separado.",
    )
    cell(
        "code",
        dedent("""
        paths = [refit_offset, refit_lgbm, public_run]
        inputs = {str(p / name): digest(p / name) for p in paths
                  for name in ('config.json', 'metrics.json', 'predictions.csv')}
        inputs.update(CV_ARTIFACT_SHA256)
        inputs.update({str(p / 'model.joblib'): digest(p / 'model.joblib') for p in paths[:2]})
        for path, expected in inputs.items():
            assert digest(path) == expected, path
        metrics = [json.loads((p / 'metrics.json').read_text()) for p in paths]
        assert all(m['status'] == 'complete' and m['data_hashes'] == RECIPE['data_hashes']
                   for m in metrics)
        assert all(m['phase'] == 'full_data_refit' and m['roc_auc'] is None
                   and m['fold_scores'] == [] and m['rows']['oof'] == 0 for m in metrics[:2])
        assert metrics[2]['phase'] == 'exploratory_full_data_cv'
        for path, metric in zip(paths[:2], metrics[:2]):
            assert digest(path / 'model.joblib') == metric['checkpoint_sha256']
        external = [{**c, 'weight': 0.25}
                    for c in metrics[2]['model_details']['external_components']]
        assert len(external) == 1 and external[0]['trained_by_us'] is False
        assert external[0]['predictions_source_version'] == 2
        assert external[0]['predictions_license'] == 'CC BY 4.0'
        frames = [pd.read_csv(p / 'predictions.csv') for p in paths]
        for frame in frames:
            validate_submission(frame, sample, 'id', 'is_fraud')
        probability = sum(w * frame['is_fraud'].to_numpy() for w, frame in zip(weights, frames))
        final = submission(raw_test['id'], probability, sample, 'is_fraud')
        directory = create_run(runs, RECIPE['refit']['hybrid_name'])
        config = {'operation': 'portable_selected_cv_refit_composition',
                  'sources': list(map(str, paths)), 'weights': weights,
                  'inputs_sha256': inputs, 'notebook_export': full_common['notebook_export']}
        write_json(directory / 'config.json', config)
        for filename in ('predictions.csv', 'submission.csv'):
            final.to_csv(directory / filename, index=False)
        written = pd.read_csv(directory / 'submission.csv')
        validate_submission(written, sample, 'id', 'is_fraud')
        assert np.allclose(written['is_fraud'], probability, rtol=0, atol=1e-15)
        check_inputs()
        for path, expected in inputs.items():
            assert digest(path) == expected, path
        final_metric = {
            **full_common, **source_provenance(), 'status': 'complete', 'model': 'blend',
            'features': metrics[0]['features'],
            'config_sha256': digest(directory / 'config.json'),
            'model_details': {'method': 'weighted_probability_average',
                              'sources': list(map(str, paths)), 'weights': weights,
                              'external_components': external},
            'external_components': external,
            'submission_sha256': digest(directory / 'submission.csv'),
        }
        assert final_metric['roc_auc'] is None and final_metric['rows']['oof'] == 0
        assert not (directory / 'oof.csv').exists()
        write_json(directory / 'metrics.json', final_metric)
        write_json(directory / 'status.json', {'status': 'complete'})
        shutil.copyfile(directory / 'submission.csv', 'submission.csv')
        outputs = {str(p): digest(p) for folder in [refit_offset, refit_lgbm, directory]
                   for p in folder.iterdir() if p.is_file()}
        outputs['submission.csv'] = digest('submission.csv')
        assert outputs['submission.csv'] == final_metric['submission_sha256']
        execution_manifest = {
            'origin': EXPORT, 'source_sha256': SOURCE_SHA256, 'recipe_sha256': RECIPE_SHA256,
            'data_sha256': RECIPE['data_hashes'], 'input_artifacts_sha256': inputs,
            'output_artifacts_sha256': outputs,
            'phase': 'full_data_refit', 'roc_auc': None, 'fold_scores': [], 'oof_rows': 0,
            'development_roc_auc': reference['roc_auc'],
            'development_roc_auc_scope': full_common['development_roc_auc_scope'],
            'external_training_executed_here': False, 'external_probability_weight': 0.25,
            'kaggle_submission_sent': False,
        }
        write_json(Path('notebook-execution-manifest.json'), execution_manifest)
        print({'file': 'submission.csv', 'sha256': outputs['submission.csv'],
               'phase': 'full_data_refit', 'roc_auc': None, 'oof_rows': 0})
        """),
    )
    observed = recipe["observed_local_results"]
    cell(
        "markdown",
        "## Resultados locais registrados e limites\n\n"
        f"R18 CV híbrida local: **{observed['hybrid_oof_auc']:.10f}**, sem holdout independente. "
        f"R20 local: submissão **{observed['refit_public_submission']}**, score público "
        f"**{observed['refit_public_auc']:.5f}**, posição **{observed['ranking']}/{observed['total_teams']}** "
        f"em **{observed['observed_at']}**. A referência CV pertence à R18; não é AUC do refit. "
        "Esses fatos vêm das evidências do repositório e não de execução deste notebook salvo. "
        "Os módulos são os mesmos blobs 8a1, mas bibliotecas, caminhos e ambiente podem mudar "
        "os bytes dos novos modelos. Não se alega identidade com os checkpoints locais.\n\n"
        "A reserva histórica de 15% não é independente desta seleção e não é reavaliada. "
        "Gerar CSV, salvar notebook e obter score são estados distintos. Nenhuma submissão "
        "é automática; ranking é provisório, primeiras posições não foram alcançadas e "
        "a classificação privada final depende do encerramento.",
    )


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
        "--revision",
        help="Published source SHA; the canonical R18/R20 recipe requires its exact pinned revision",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Regenerate the historical own-model protocol from its local holdout report",
    )
    args = parser.parse_args()
    recipe_path = ROOT / "notebooks" / "hybrid-recipe.json"
    recipe = (
        None if args.protocol else json.loads(recipe_path.read_text(encoding="utf-8"))
    )
    revision = args.revision or (
        REVISION if args.protocol else recipe.get("source_revision")
    )
    if revision is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("revision must be a full Git commit SHA")
    if recipe is None:
        configs, hashes, record = protocol(args.protocol)
        dataset_sources = []
    else:
        if revision != recipe["source_revision"]:
            parser.error("The R18/R20 recipe requires its pinned source revision")
        configs, hashes = {}, recipe["data_hashes"]
        record = {
            "origin": "portable_hybrid_recipe",
            "phase": recipe["phase"],
            "recipe": recipe,
        }
        dataset_sources = [recipe["public_component"]["predictions_download_ref"]]
    notebook, sources = make_notebook(revision, configs, hashes, record)
    checks = verify(notebook, sources)
    output = ROOT / "notebooks"
    output.mkdir(exist_ok=True)
    if recipe is not None:
        recipe_path.write_bytes(
            (json.dumps(recipe, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
    (output / "fraudlab.ipynb").write_bytes(
        (json.dumps(notebook, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
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
        dataset_sources=dataset_sources,
        competition_sources=["playground-series-s6e11"],
        kernel_sources=[],
    )
    metadata_path.write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    print(
        json.dumps(
            {
                "revision": revision,
                "notebook": "notebooks/fraudlab.ipynb",
                **checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
