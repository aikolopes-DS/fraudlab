"""Generate the public Kaggle notebook from a pinned source revision."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Published Git commit to install")
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("revision must be a full Git commit SHA")
    cells = []

    def cell(kind: str, text: str) -> None:
        value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", "# FraudLab · experimentos reproduzíveis\n\n"
         "Código público da participação no Playground S6E11. Este notebook compartilha "
         "o pipeline; uma versão salva sem execução não comprova treinamento no Kaggle.\n\n"
         "[Repositório e evidências](https://github.com/aikolopes-og/fraudlab) · "
         "[Regras da competição](https://www.kaggle.com/competitions/playground-series-s6e11/rules)\n\n"
         "Os experimentos locais, seus scores e as submissões observadas são documentados "
         "no repositório. O ranking público é provisório. Código MIT; dados sob os termos da competição.")
    cell("code", f'%pip install -q "fraudlab[neural] @ git+https://github.com/aikolopes-og/fraudlab.git@{args.revision}"\n')
    cell("code", "from pathlib import Path\nimport subprocess\nimport sys\n\n"
         "candidates = [Path('/kaggle/input/competitions/playground-series-s6e11'),\n"
         "              Path('/kaggle/input/playground-series-s6e11')]\n"
         "data = next(path for path in candidates if (path / 'train.csv').is_file())\n"
         "subprocess.run([sys.executable, '-m', 'fraudlab', 'inspect', '--data', str(data)], check=True)\n")
    cell("markdown", "## Treinamento explícito\n\n"
         "Ative GPU nas configurações para acelerar a rede. A execução abaixo cria "
         "OOF, checkpoints e um CSV validado; não envia nenhuma submissão automaticamente. "
         "Para seleção, reserve o holdout antes das comparações e siga a operação documentada no repositório. "
         "A hora do dia não é um timestamp cronológico e não há identificador de cliente nos arquivos inspecionados.")
    cell("code", "subprocess.run([\n"
         "    sys.executable, '-m', 'fraudlab', 'train', '--data', str(data),\n"
         "    '--model', 'mlp', '--name', 'kaggle-mlp', '--folds', '3', '--device', 'auto',\n"
         "    '--validation-note',\n"
         "    'Synthetic transactions; no chronological timestamp or entity identifier; no duplicate feature rows in the inspected version.',\n"
         "], check=True)\n")
    output = Path("notebooks")
    output.mkdir(exist_ok=True)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (output / "fraudlab.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {"id": "aikoog/fraudlab-reproducible-neural-fraud-detection", "title": "FraudLab - Reproducible Neural Fraud Detection", "code_file": "fraudlab.ipynb", "language": "python", "kernel_type": "notebook", "is_private": False, "enable_gpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": ["playground-series-s6e11"], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
