# Executar o projeto

Os comandos partem da raiz do repositório e usam PowerShell. São receitas de execução; os resultados realmente obtidos ficam registrados no README, nos experimentos e em `evidence/public.json`.

## Ambiente

O núcleo aceita Python 3.10 ou superior. O ambiente local foi iniciado com 3.10; a CI usa 3.11. Em um clone novo:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,presentation]"
```

Sem ativação, substitua `python` por `.\.venv\Scripts\python.exe`. No Linux, crie o ambiente com `python3.11 -m venv .venv` e ative com `source .venv/bin/activate`; os demais comandos Python são iguais.

Essa instalação do PyTorch usa CPU. Para CUDA, siga o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) e confirme `torch.cuda.is_available()` antes de usar `--device cuda`. As versões efetivamente utilizadas ficam em cada `metrics.json`.

## Dados e regras

Confirme as regras e a inscrição na [competição S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/rules). Os arquivos esperados são `data/raw/train.csv`, `test.csv` e `sample_submission.csv`. Dados, credenciais e artefatos gerados são ignorados pelo Git.

A CLI Kaggle 2.2.5 utilizada neste projeto exige Python 3.11 e usa ambiente separado. Com a autenticação configurada pelo mecanismo oficial da CLI:

```powershell
py -3.11 -m venv .local/kaggle-cli
.\.local\kaggle-cli\Scripts\python.exe -m pip install kaggle==2.2.5 six==1.17.0
.\.local\kaggle-cli\Scripts\kaggle.exe competitions download playground-series-s6e11 -p data/raw --unzip
python -m fraudlab inspect --data data/raw --output artifacts/profile.json
```

A inspeção grava agregados, tipos, ausências, duplicatas e hashes dos arquivos. Leia o perfil antes de decidir a divisão: uma coluna de tempo, uma entidade repetida ou duplicatas podem exigir outra estratégia.

## Comparar modelos

O exemplo abaixo serve para dados cuja hipótese de linhas independentes sustente validação estratificada. A justificativa deve refletir a inspeção real. Grupo e tempo são alternativas explícitas: `--split group --group-column NOME` e `--split time --time-column NOME`.

```powershell
$splitReason = "Divisao estratificada sob hipotese IID, justificada pela inspecao dos dados."
python -m fraudlab train --name logistic-dev --model logistic --folds 3 --holdout-fraction 0.15 --holdout-seed 2026 --validation-note "$splitReason"
python -m fraudlab train --name catboost-dev --model catboost --folds 3 --holdout-fraction 0.15 --holdout-seed 2026 --validation-note "$splitReason"
python -m fraudlab train --name mlp-dev --model mlp --folds 3 --holdout-fraction 0.15 --holdout-seed 2026 --validation-note "$splitReason" --device cpu
```

Cada nome cria uma execução imutável em `artifacts/runs/`. A pasta contém configuração, status, modelos por fold, métricas, previsões OOF, previsões de teste e `submission.csv`. O ID e o alvo são excluídos das entradas. Imputação, escala e codificação são ajustadas dentro de cada fold de treino.

O holdout é reservado antes da validação cruzada e não participa do treino nem da parada antecipada. A opção atual suporta apenas a estratégia estratificada. Para grupo ou tempo, defina um protocolo independente coerente com essa estrutura; o programa rejeita combinar essas estratégias com o holdout aleatório.

Uma combinação só deve ser adotada quando os experimentos sustentarem a escolha. Os pesos iguais abaixo exemplificam o comando; não representam uma configuração vencedora comprovada:

```powershell
python -m fraudlab blend --runs artifacts/runs/catboost-dev artifacts/runs/mlp-dev --weights 1 1 --name blend-dev
```

O programa exige os mesmos dados, IDs, divisões e rótulos OOF. Compare a ROC-AUC e a variação entre folds. Usar repetidamente os mesmos resultados para escolher modelos não produz uma avaliação final independente.

Depois de congelar a seleção e os pesos, avalie uma vez no holdout reservado:

```powershell
python -m fraudlab evaluate-holdout --runs artifacts/runs/catboost-dev artifacts/runs/mlp-dev --weights 1 1 --output artifacts/holdout/evaluation.json --device cpu
```

O comando registra a seleção antes da avaliação e rejeita sobrescrever a saída existente. Usar esse resultado para reajustar modelos transforma o holdout em parte da seleção. Ao encerrar essa fase, a configuração escolhida pode ser treinada novamente com todos os dados, em novos nomes de execução e com `--holdout-fraction 0`.

## Validar e submeter

```powershell
python -m fraudlab validate-submission --file artifacts/runs/blend-dev/submission.csv --sample data/raw/sample_submission.csv
```

A validação exige colunas exatas, IDs únicos na ordem esperada e probabilidades finitas entre zero e um. Isso verifica o arquivo, não seu desempenho. Para enviar o arquivo real após a seleção, respeitando as regras e os limites da conta:

```powershell
.\.local\kaggle-cli\Scripts\kaggle.exe competitions submit playground-series-s6e11 -f artifacts/runs/blend-dev/submission.csv -m "FraudLab: blend-dev"
```

Confirme o processamento na plataforma. Registre em `evidence/public.json` o identificador, a pontuação pública, a posição, o total de equipes e a data da observação. Flags de verificação só devem ser ativadas após a conferência correspondente. O ranking público é provisório.

## Apresentação e vídeo

Para revisar o conteúdo antes de ter todas as evidências:

```powershell
python -m presentation.build --draft --output artifacts/presentation-draft
```

A prévia é marcada como rascunho e não substitui os experimentos. A versão final exige dados e treino verificados, métricas de uma MLP realmente treinada, submissão e ranking documentados:

```powershell
python -m presentation.build --metrics artifacts/runs/blend-dev/metrics.json --comparison artifacts/runs/mlp-dev/metrics.json --comparison artifacts/runs/catboost-dev/metrics.json --comparison artifacts/runs/logistic-dev/metrics.json --output artifacts/presentation --video
```

`--video` usa por padrão a voz online `pt-BR-AntonioNeural`, pelo pacote `edge-tts` do extra `presentation`, e requer FFmpeg no `PATH`. A alternativa local no Windows é `--speech-engine sapi --voice "Microsoft Maria Desktop"`. O comando gera slides HTML editáveis, imagens, roteiro, vídeo MP4, legendas SRT e WebVTT e um manifest com hashes. Não há duração fixa: cada slide técnico é seguido imediatamente por explicação simples com exemplo cotidiano. O vídeo exportado precisa de revisão visual e auditiva; geração bem-sucedida não marca essa revisão como realizada.

## Site

```powershell
python site/build.py --presentation-dir artifacts/presentation-draft
python -m http.server 8000 --directory site/dist
```

Abra `http://localhost:8000`. Para a versão com resultados, use `--run-dir artifacts/runs/blend-dev --presentation-dir artifacts/presentation`. O build usa `evidence/public.json`, valida os hashes da apresentação e inclui somente arquivos públicos previstos. Métricas locais não são convertidas em score público ou ranking. A hospedagem estática recebe `site/dist/`; o servidor acima é apenas local, e o build não realiza deploy.

## Testes e seus limites

```powershell
python -m pytest tests presentation/test_presentation.py site/test_build.py -q
node --check site/app.js
```

Os testes cobrem contratos, divisões, pequenos treinos sintéticos, persistência, combinação de previsões, barreiras de evidência e empacotamento público. A CI instala PyTorch CPU para executar também o teste neural. Ela não acessa a conta Kaggle, não usa os dados da competição, não submete arquivos e não valida a síntese de voz, a reprodução integral do vídeo ou a publicação.

A verificação visual local é separada:

```powershell
python -m pip install playwright
python -m playwright install chromium
python site/check_browser.py
```

Esse comando verifica o site local em desktop e mobile, exercita a navegação e registra capturas em `artifacts/qa/site/`. Não comprova o funcionamento da competição.

As ações da CI estão fixadas nos commits oficiais das versões [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) e [setup-python v7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0), conferidos pela API do GitHub em 16/11/2026.
