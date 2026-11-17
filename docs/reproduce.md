# Executar o projeto

Os comandos partem da raiz do repositório e usam PowerShell. A receita operacional é a [híbrida 5/42](../notebooks/hybrid-recipe.json): modelos próprios fixos mais previsões públicas GPT-2 versão 2. Os resultados observados e o acompanhamento ficam no [README](../README.md) e em `evidence/public.json`.

Use um clone novo ou novos nomes de execução: `artifacts/runs/` não sobrescreve runs existentes. Versões e hardware podem alterar resultados; reproduzir o código não transfere uma pontuação já processada pelo Kaggle para outro arquivo.

## Ambiente

O núcleo aceita Python 3.10 ou superior; a CI usa 3.11. A instalação abaixo inclui apresentação, testes e PyTorch CPU para as comparações neurais próprias:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,presentation]"
```

Sem ativação, substitua `python` por `.\.venv\Scripts\python.exe`. No Linux, use `python3.11 -m venv .venv` e `source .venv/bin/activate`. A receita híbrida roda em CPU e não treina o GPT-2. Para treinar a MLP própria em CUDA, instale PyTorch pelo [seletor oficial](https://pytorch.org/get-started/locally/) e confirme `torch.cuda.is_available()`.

Para experimentar `--feature-set bpe_digits`, instale `python -m pip install -e ".[bpe]"`.
Essa opção usa `tiktoken==0.14.0` e seis chaves numéricas; não chama uma API paga
nem carrega pesos de uma rede. A primeira inicialização baixa os vocabulários
oficiais e verifica seus hashes. Para executar depois sem internet, preserve o
diretório configurado por `TIKTOKEN_CACHE_DIR`. O manifesto registra a versão,
os hashes, o arredondamento e o descarte dos tokens após o segundo.

`--model glm --glm-value-one-hot` acrescenta efeitos por valor com one-hot
esparso e agrupamento de contagens menores que cinco, ajustados dentro de cada
treino. Essa opção usa L-BFGS com limite de 2.000 iterações e recusa ajustes sem
convergência. O manifesto do modelo registra solver, limite e iterações reais.

## Dados oficiais e componente público

Confira inscrição e [regras da S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/rules). Configure a autenticação pelo mecanismo oficial da CLI; credenciais, dados e artefatos locais ficam fora do Git. A CLI 2.2.5 utilizada tem um ambiente Python 3.11 separado:

```powershell
py -3.11 -m venv .local/kaggle-cli
.\.local\kaggle-cli\Scripts\python.exe -m pip install kaggle==2.2.5 six==1.17.0
$fraudKaggle = ".\.local\kaggle-cli\Scripts\kaggle.exe"
& $fraudKaggle competitions download playground-series-s6e11 -p data/raw --unzip
& $fraudKaggle datasets download masayakawamata/s6e11-gpt2-target-first-oof/2 -p .local/research/public-models/gpt2 --unzip
python -m fraudlab inspect --data data/raw --output artifacts/profile.json
```

Os CSVs esperados são `train.csv`, `test.csv` e `sample_submission.csv`. O perfil registra tipos, ausências, duplicatas e hashes. Nesta versão inspecionada, os dados são sintéticos e não têm IDs de entidade ou timestamp cronológico; a hora do dia é recorrente. Essa inspeção sustenta a divisão estratificada usada na receita.

As previsões [GPT-2 de Masaya Kawamata](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof/versions/2) são CC BY 4.0. Seu [treinamento público MIT](https://github.com/Matapanino/s6e11-gpt2-target-first/tree/8446fb73c11583df4f7006aa08fa34472249b318) foi realizado pelo autor externo. Créditos e auditoria estão em [external-gpt2.json](../evidence/external-gpt2.json) e na receita, incluindo Chris Deotte, Borisov et al./GReaT e uditjain13.

```powershell
python -m fraudlab import-public-gpt2 --source .local/research/public-models/gpt2 --data data/raw --name explore5-public-gpt2-v2
```

O importador não executa código externo. Confere manifesto fixado, hashes dos arrays e CSVs, formas, tipos, valores finitos, ordem dos IDs e folds exatos 5/42. Os arrays contêm logits: aplica sigmoid e grava o contrato OOF/probabilidades com autoria, versão e licenças. No teste público, o autor já fez a média dos cinco logits; essa média recebe a sigmoid na importação.

## Receita híbrida congelada

Todas as 699.960 linhas entram na CV de cinco folds, seed 42. Cada OOF exclui a linha de seu treino; o teste recebe a média das probabilidades dos cinco modelos de cada família própria. Não há reserva independente nesta etapa, nem early stopping ou escolha nova do número de árvores.

Execute os treinos sequencialmente para limitar RAM:

```powershell
$fraudRecipe = Get-Content notebooks/hybrid-recipe.json -Raw -Encoding UTF8 | ConvertFrom-Json
$fraudCvArgs = @(
    "--data", "data/raw", "--phase", "exploratory_full_data_cv",
    "--feature-set", "value_digits", "--split", "stratified",
    "--folds", "5", "--seed", "42", "--holdout-fraction", "0",
    "--fixed-training", "--validation-note", $fraudRecipe.common.validation_note
)
python -m fraudlab train @fraudCvArgs --model glm --name explore5-glm --threads 4 --logistic-c 0.3 --target-smoothing 50
python -m fraudlab train @fraudCvArgs --model lightgbm --name explore5-lgbm --threads 6 --iterations 1438 --num-leaves 4 --learning-rate 0.04 --l2-leaf-reg 5 --min-child-samples 100 --target-smoothing 20 --patience 150
python -m fraudlab blend --runs artifacts/runs/explore5-glm artifacts/runs/explore5-lgbm --weights 0.75 0.25 --name explore5-own
python -m fraudlab blend --runs artifacts/runs/explore5-own artifacts/runs/explore5-public-gpt2-v2 --weights 0.75 0.25 --name explore5-public25
python -m fraudlab validate-submission --file artifacts/runs/explore5-public25/submission.csv --sample data/raw/sample_submission.csv
```

O LGB usa 1.438 árvores fixas; `--fixed-training` desativa early stopping, portanto `patience` não seleciona árvores nesta execução. O GLM usa C=0,3, target encoding com suavização 50 e Newton-Cholesky. O LGB usa suavização 20. Transformações aprendidas são ajustadas somente dentro do treino de cada fold; o target encoding usa mais cinco divisões internas.

A média final é em **probabilidades**: 75% da composição própria mais 25% GPT-2, equivalentes a **56,25% GLM / 18,75% LGB / 25% GPT-2 externo**. A grade local anterior comparou 10/25/50/75% de GPT-2 por OOF compatível; esta receita reproduz a escolha congelada, sem repetir seleção pelo leaderboard. `phase=exploratory_full_data_cv` e `independent_holdout=false` permanecem nos metadados.

Cada run salva configuração, status, métricas, OOF, previsões do teste e submissão; treinos próprios também salvam os modelos. O blend exige hashes dos dados, IDs, folds e rótulos OOF compatíveis e propaga créditos e peso efetivo do componente externo.

## Notebook público

```powershell
python scripts/kaggle_notebook.py --revision 30dc7eb723e4cf31291bb075ed6c0d943557264c
```

O gerador cria um único `notebooks/fraudlab.ipynb`, incorpora a receita e exporta os módulos por `git show` desse SHA, com hashes. A metadata anexa a competição e `masayakawamata/s6e11-gpt2-target-first-oof/2`, com internet desativada. Não há pip remoto nem treinamento de GPT-2 no notebook; a imagem precisa fornecer as bibliotecas compatíveis, incluindo LightGBM >=4.7.

A versão 3 foi publicada por **Save sem execução**. Sintaxe, comandos e exportação foram conferidos; isso não comprova execução do notebook no ambiente Kaggle. Treinos locais e avaliação dos CSVs pela competição têm registros separados. Ao executá-lo futuramente, os inputs precisam estar anexados na versão indicada.

## Histórico próprio: reserva e refits

A primeira etapa usou três folds nos 85% de desenvolvimento e reservou 15% com seed 2026. Após congelar a composição própria 75/25, a reserva foi avaliada uma vez pelas médias dos três checkpoints de cada família. Em seguida, `final-glm` e `final-lgbm` foram refitados em todos os rótulos, sem nova OOF ou early stopping; a regra congelada definiu 1.438 árvores no LGB. Esses refits compõem a submissão própria anterior **58437018**.

A híbrida **58438450** usa as médias de teste das novas CVs `explore5-glm`/`explore5-lgbm` e GPT-2 v2. Ela não usa os refits únicos anteriores. Como a nova etapa e o componente externo incluem os antigos rótulos reservados, aquele holdout é apenas histórico para a híbrida.

O gerador aceita `--protocol artifacts/holdout/evaluation.json` para reconstruir o notebook próprio anterior, desde que o relatório e os configs de origem estejam disponíveis localmente. Ele regrava o mesmo notebook; o comando híbrido acima restaura a versão canônica. Esses artefatos locais não são incluídos no clone público.

A MLP-TE própria permanece uma comparação histórica real, fora da composição final. Para reproduzir suas métricas de desenvolvimento necessárias à apresentação:

```powershell
$fraudHistoricalNote = "Comparacao historica: 3 folds nos 85% de desenvolvimento, seed 42 e reserva 15% seed 2026; nao e holdout independente da receita hibrida."
python -m fraudlab train --model mlp --name dev-mlp-te-values --feature-set value_digits --target-smoothing 20 --neural-categorical-transform target --neural-numeric-transform quantile --folds 3 --seed 42 --holdout-fraction 0.15 --holdout-seed 2026 --epochs 20 --patience 4 --threads 6 --device auto --validation-note "$fraudHistoricalNote"
```

A execução original usou CUDA; `auto` permite CPU quando CUDA não estiver disponível. Isso não reabre nem repete a avaliação histórica da reserva.

## Submissão e evidência

```powershell
& $fraudKaggle competitions submit playground-series-s6e11 -f artifacts/runs/explore5-public25/submission.csv -m "FraudLab: hybrid5fold-public25"
```

Enviar é uma ação explícita, sujeita às regras e limites da conta. Confirme o processamento e registre identificador, score, posição, total de equipes e data. [record_submission.py](../scripts/record_submission.py) confere o arquivo e a resposta da plataforma; seus argumentos estão em `python scripts/record_submission.py --help`. Para a híbrida, não forneça o relatório antigo como holdout independente. O ranking público é provisório; a meta da liderança não foi alcançada.

## Apresentação, vídeo e site

Com evidências correspondentes à submissão, as métricas híbridas e a comparação MLP histórica disponíveis:

```powershell
python -m presentation.build --evidence evidence/public.json --metrics artifacts/runs/explore5-public25/metrics.json --comparison artifacts/runs/explore5-glm/metrics.json --comparison artifacts/runs/explore5-lgbm/metrics.json --comparison artifacts/runs/dev-mlp-te-values/metrics.json --historical-holdout artifacts/holdout/evaluation.json --output artifacts/presentation-hybrid --video
python -m presentation.review --presentation-dir artifacts/presentation-hybrid
python site/build.py --run-dir artifacts/runs/explore5-public25 --presentation-dir artifacts/presentation-hybrid
python -m http.server 8000 --directory site/dist
```

O relatório histórico é opcional: omita `--historical-holdout` se não o possuir; a apresentação híbrida não exige uma reserva independente inexistente. Para revisar um rascunho sem todas as evidências, use `--draft` e outra pasta de saída.

`--video` usa a voz online `pt-BR-AntonioNeural` via `edge-tts` e requer FFmpeg no PATH. No Windows, há a alternativa `--speech-engine sapi --voice "Microsoft Maria Desktop"`. São gerados slides HTML, imagens, roteiro, MP4, legendas SRT/WebVTT e manifest de hashes. Cada slide complexo é seguido por explicação simples e exemplo cotidiano; a cobertura determina a duração.

`presentation.review` verifica formato, timeline e legendas e extrai quadros. A transcrição integral automática é opcional: instale `pip install -e ".[review]"` e acrescente `--asr`; o modelo ASR pode precisar de download inicial. Nenhuma dessas verificações declara que uma pessoa assistiu ou ouviu o vídeo inteiro.

O site estático é montado em `site/dist/`, com evidência pública e arquivos permitidos pelo manifest; datasets, pesos e caminhos locais não são publicados. O servidor acima é apenas preview local. Hospedagem e acesso público são verificados separadamente; o build não faz deploy.

O mirror da hospedagem Sites limita cada arquivo a 32 MiB; o MP4 desta versão tem cerca de 49 MiB. Para usar o vídeo publicado na release GitHub e manter slides/legendas no site:

```powershell
python site/build.py --run-dir artifacts/runs/explore5-public25 --presentation-dir artifacts/presentation --video-url https://github.com/aikolopes-og/fraudlab/releases/download/v0.1.0/video.mp4
```

O build ainda confere o hash do MP4 local, registra o hash e a URL externa no manifesto público e exclui o binário de `site/dist/`. A igualdade dos bytes publicados e a reprodução anônima precisam ser verificadas após o upload.

## Testes e limites

A continuação de 17/11 é reproduzida em um clone com os mesmos CSVs e novos
nomes de execução. Todos os candidatos abaixo usam cinco folds/seed42,
pré-processamento local ao treino e nenhuma reserva independente:

```powershell
$fraudRoundArgs = @("--phase", "exploratory_full_data_cv", "--folds", "5", "--seed", "42", "--holdout-fraction", "0", "--fixed-training", "--threads", "4", "--logistic-c", "0.3", "--target-smoothing", "50", "--validation-note", "Exploratory full-data CV5/42, no independent holdout; fixed candidates and fold-local preprocessing.")
python -m fraudlab train @fraudRoundArgs --model offset --feature-set value_digits --name round2-offset1000 --iterations 1000 --num-leaves 4 --learning-rate 0.03 --l2-leaf-reg 5 --min-child-samples 100
python -m fraudlab train @fraudRoundArgs --model glm --feature-set value_digits --name round2-glm-splines12 --glm-spline-knots 12
python -m fraudlab train @fraudRoundArgs --model glm --feature-set paired_digits --name round3-glm-pairs
python -m fraudlab train @fraudRoundArgs --model glm --feature-set paired_digits --name round3-glm-pairs-frequency --glm-frequency
```

Os planos das rodadas 2/3 descrevem as combinações de probabilidades e seus pesos.
A rodada 4 contém a fórmula de duas operações fixas em log-odds; como elas não
atingiram o critério, não foi acrescentada uma API de produção para essa operação.
Os resultados negativos não alteram a receita selecionada nem enviam arquivos
automaticamente ao Kaggle. Não reexecute a avaliação histórica da reserva.

```powershell
python -m pytest tests presentation site/test_build.py -q
node --check site/app.js
```

A CI usa o mesmo comando com Python 3.11 e PyTorch CPU. Os testes exercitam contratos, treinos sintéticos pequenos, persistência, importação simulada, mistura, evidências e empacotamento. Não acessam a conta Kaggle, não treinam na competição, não submetem CSVs e não substituem revisão audiovisual ou verificação da publicação.

Para a verificação visual local separada:

```powershell
python -m pip install playwright
python -m playwright install chromium
python site/check_browser.py
```

O comando verifica desktop/mobile e registra capturas em `artifacts/qa/site/`. A qualidade competitiva exige os experimentos reais e o resultado processado na plataforma.
