# Executar o projeto

Os comandos partem da raiz do repositório e usam PowerShell. A receita operacional atual é a [R10](../evidence/round10-plan.json): 56,25% offset com BPE e frequência, 18,75% LightGBM e 25% das previsões públicas GPT-2 v2. Sua fonte de ML é o commit [169f3ee](https://github.com/aikolopes-DS/fraudlab/tree/169f3ee5cdfbb2afe16c942a7af0efb8b12a403f). A [receita v0.1](../notebooks/hybrid-recipe.json) e os refits anteriores são históricos. O acompanhamento único fica no [README](../README.md); os fatos publicados estão em `evidence/public.json`.

Use um clone novo ou novos nomes de execução: `artifacts/runs/` não sobrescreve runs existentes. Versões e hardware podem alterar resultados; reproduzir o código não transfere uma pontuação já processada pelo Kaggle para outro arquivo.

## Ambiente

O núcleo aceita Python 3.10 ou superior; a CI usa 3.11. A instalação abaixo inclui apresentação, testes e PyTorch CPU para as comparações neurais próprias:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,presentation,bpe]"
```

Sem ativação, substitua `python` por `.\.venv\Scripts\python.exe`. No Linux, use `python3.11 -m venv .venv` e `source .venv/bin/activate`. A receita híbrida roda em CPU e não treina o GPT-2. Para treinar a MLP própria em CUDA, instale PyTorch pelo [seletor oficial](https://pytorch.org/get-started/locally/) e confirme `torch.cuda.is_available()`.

A R10 usa `--feature-set bpe_digits`, incluído na instalação acima.
Essa representação usa `tiktoken==0.14.0` e seis chaves numéricas; não chama uma API paga
nem carrega pesos de uma rede. A primeira inicialização baixa os vocabulários
oficiais e verifica seus hashes. Para executar depois sem internet, preserve o
diretório configurado por `TIKTOKEN_CACHE_DIR`. O manifesto registra a versão,
os hashes, o arredondamento e o descarte dos tokens após o segundo.

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

## Receita atual: R10

Todas as 699.960 linhas entram na CV de cinco folds, seed 42. Cada OOF exclui a linha de seu treino; o teste recebe a média das probabilidades dos cinco modelos de cada família própria. Não há reserva independente nesta etapa, nem early stopping ou escolha nova do número de árvores.

Execute os treinos sequencialmente para limitar RAM:

```powershell
$fraudR10Plan = Get-Content evidence/round10-plan.json -Raw -Encoding UTF8 | ConvertFrom-Json
$fraudCvArgs = @(
    "--data", "data/raw", "--phase", "exploratory_full_data_cv",
    "--split", "stratified",
    "--folds", "5", "--seed", "42", "--holdout-fraction", "0",
    "--fixed-training", "--validation-note", $fraudR10Plan.validation
)
python -m fraudlab train @fraudCvArgs --model offset --name round10-offset-bpe-frequency --feature-set bpe_digits --glm-frequency --glm-inner-folds 5 --glm-spline-knots 0 --logistic-c 0.3 --target-smoothing 50 --iterations 1000 --num-leaves 4 --learning-rate 0.03 --l2-leaf-reg 5 --min-child-samples 100 --threads 4 --device cpu
python -m fraudlab train @fraudCvArgs --model lightgbm --name explore5-lgbm --feature-set value_digits --threads 6 --iterations 1438 --num-leaves 4 --learning-rate 0.04 --l2-leaf-reg 5 --min-child-samples 100 --target-smoothing 20 --patience 150
python -m fraudlab blend --runs artifacts/runs/round10-offset-bpe-frequency artifacts/runs/explore5-lgbm artifacts/runs/explore5-public-gpt2-v2 --weights 0.5625 0.1875 0.25 --name round10-offset-bpe-frequency-public25
python -m fraudlab validate-submission --file artifacts/runs/round10-offset-bpe-frequency-public25/submission.csv --sample data/raw/sample_submission.csv
```

O offset usa GLM Newton-Cholesky com C=0,3 e target encoding interno 5/50, mais 1.000 árvores residuais de quatro folhas, taxa 0,03, L2=5 e mínimo de 100 exemplos por folha. O GLM e as árvores usam a mesma matriz de treino com encodings cruzados; a inferência soma suas margens antes da sigmoid. O LightGBM separado mantém 1.438 árvores, taxa 0,04 e suavização 20. `--fixed-training` desativa early stopping; `patience` não seleciona árvores nessas execuções.

A mistura final é em **probabilidades**: **56,25% offset / 18,75% LGB / 25% GPT-2 externo**. O [resultado R10](../evidence/round10-results.json) registra OOF **0,9514921382152453** e melhora nos cinco folds sobre a baseline original. Pesos e parâmetros foram declarados antes do treino e do envio. A OOF foi usada na seleção; `phase=exploratory_full_data_cv` e `independent_holdout=false` permanecem nos metadados. A [R11](../evidence/round11-results.json), com 2.000 árvores residuais, ficou abaixo da R10 e não foi submetida.

Cada run salva configuração, status, métricas, OOF, previsões do teste e submissão; treinos próprios também salvam os modelos. O blend exige hashes dos dados, IDs, folds e rótulos OOF compatíveis e propaga créditos e peso efetivo do componente externo.

## Notebook V5: baseline executável histórica

O [notebook V5 publicado](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) mantém as células executáveis da baseline v0.1 e acrescenta Markdown sobre a R10, com links para sua fonte. Foi salvo **sem execução na nuvem**: as 22 células remotas conferem com as locais, sem outputs ou contagens de execução. A [verificação](../evidence/verification.json) não comprova treino, previsões ou submissão da R10 naquele notebook. A reprodução atual é a CLI da seção anterior.

O exportador abaixo reconstrói a baseline v0.1, não a R10 nem o Markdown atualizado da V5. Ele regrava `notebooks/fraudlab.ipynb` e a receita; use um clone separado para preservar a versão publicada:

```powershell
python scripts/kaggle_notebook.py --revision 30dc7eb723e4cf31291bb075ed6c0d943557264c
```

O gerador incorpora a receita histórica e exporta os módulos por `git show` desse SHA, com hashes. A metadata anexa a competição e `masayakawamata/s6e11-gpt2-target-first-oof/2`, com internet desativada. Não há pip remoto nem treinamento de GPT-2 no notebook; a imagem precisa fornecer bibliotecas compatíveis, incluindo LightGBM >=4.7. Os treinos locais e a avaliação dos CSVs pelo Kaggle têm registros separados.

## Histórico próprio: reserva e refits

A primeira etapa usou três folds nos 85% de desenvolvimento e reservou 15% com seed 2026. Após congelar a composição própria 75/25, a reserva foi avaliada uma vez pelas médias dos três checkpoints de cada família. Em seguida, `final-glm` e `final-lgbm` foram refitados em todos os rótulos, sem nova OOF ou early stopping; a regra congelada definiu 1.438 árvores no LGB. Esses refits compõem a submissão própria anterior **58437018**.

A híbrida v0.1 **58438450** usava as médias de teste das CVs `explore5-glm`/`explore5-lgbm` e GPT-2 v2. Ela não usava os refits únicos anteriores. A R10 substitui seu GLM pelo offset com BPE e frequência. Como essas CVs e o componente externo incluem os antigos rótulos reservados, aquele holdout é apenas histórico para a composição atual.

Uma variante fixa reaproveita os refits completos já auditados:

```powershell
python scripts/compose_refit_hybrid.py --plan evidence/refit-hybrid-plan.json
```

O comando exige os artefatos locais cujos hashes estão no plano, incluindo os
checkpoints originais. Combina 75% de `final-blend` e 25% do GPT-2 aprovado.
Gera apenas inferência: AUC nula, nenhuma OOF e nenhuma submissão automática.
Essa variante é histórica e não compõe a R10. Sua referência CV pertence à receita anterior. Os hashes protegem os inputs
históricos; recriar numericamente os modelos em outro ambiente pode produzir
outros bytes e exige registrar um novo plano, sem herdar o score anterior.

O gerador aceita `--protocol artifacts/holdout/evaluation.json` para reconstruir o notebook próprio anterior, desde que o relatório e os configs de origem estejam disponíveis localmente. Ele também regrava o notebook; use um clone separado. Esses artefatos locais não são incluídos no clone público.

A MLP-TE própria permanece uma comparação histórica real, fora da composição final. Para reproduzir suas métricas de desenvolvimento necessárias à apresentação:

```powershell
$fraudHistoricalNote = "Comparacao historica: 3 folds nos 85% de desenvolvimento, seed 42 e reserva 15% seed 2026; nao e holdout independente da receita hibrida."
python -m fraudlab train --model mlp --name dev-mlp-te-values --feature-set value_digits --target-smoothing 20 --neural-categorical-transform target --neural-numeric-transform quantile --folds 3 --seed 42 --holdout-fraction 0.15 --holdout-seed 2026 --epochs 20 --patience 4 --threads 6 --device auto --validation-note "$fraudHistoricalNote"
```

A execução original usou CUDA; `auto` permite CPU quando CUDA não estiver disponível. Isso não reabre nem repete a avaliação histórica da reserva.

## Submissão e evidência

```powershell
& $fraudKaggle competitions submit playground-series-s6e11 -f artifacts/runs/round10-offset-bpe-frequency-public25/submission.csv -m "FraudLab: round10-offset-bpe-frequency-public25"
```

Enviar é uma ação explícita, sujeita às regras e limites da conta. Confirme o processamento e registre identificador, score, posição, total de equipes e data. [record_submission.py](../scripts/record_submission.py) confere o arquivo e a resposta da plataforma; seus argumentos estão em `python scripts/record_submission.py --help`. Para a híbrida, não forneça o relatório antigo como holdout independente. O ranking público é provisório; a meta da liderança não foi alcançada.

A R10 observada é a submissão **58470267**, score **0,95134**, posição **712/2.089** em **17/11/2026 às 13h34 UTC**. Esses números pertencem ao arquivo identificado por hash em [round10-public.json](../evidence/round10-public.json); uma reprodução não herda seu score. O envio R9 permanece uma [exceção exploratória declarada](../evidence/public-evaluation-results.json), sem reclassificar seu critério original como cumprido.

## Apresentação, vídeo e site

A versão publicada de referência continua sendo a [v0.2](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.2.0), com 134 slides e 51min23s. Uma nova geração em implementação só deve substituir esse estado depois de revisão e publicação verificadas.

Com evidências correspondentes à submissão, as métricas híbridas e a comparação MLP histórica disponíveis:

```powershell
$fraudComparisons = @("round10-offset-bpe-frequency", "round9-glm-bpe-frequency", "round9-bpe-frequency-public25", "explore5-glm", "explore5-lgbm", "explore5-public-gpt2-v2", "explore5-public25", "round8-tabm", "round8-tabm10", "dev-mlp-te-values", "dev-logistic-raw", "dev-mlp-standard-raw", "dev-mlp-quantile-raw", "dev-lgbm-leaf4-raw", "dev-lgbm-leaf4-digits")
$fraudPresentationArgs = @()
foreach ($fraudRun in $fraudComparisons) { $fraudPresentationArgs += @("--comparison", "artifacts/runs/$fraudRun/metrics.json") }
python -m presentation.build --evidence evidence/public.json --metrics artifacts/runs/round10-offset-bpe-frequency-public25/metrics.json @fraudPresentationArgs --historical-holdout artifacts/holdout/evaluation.json --output artifacts/presentation-v02 --video
python -m presentation.review --presentation-dir artifacts/presentation-v02 --asr
python site/build.py --run-dir artifacts/runs/round10-offset-bpe-frequency-public25 --presentation-dir artifacts/presentation-v02
python -m http.server 8000 --directory site/dist
```

O relatório histórico é opcional: omita `--historical-holdout` se não o possuir; a apresentação híbrida não exige uma reserva independente inexistente. Para revisar um rascunho sem todas as evidências, use `--draft` e outra pasta de saída.

`--video` usa a voz online `pt-BR-AntonioNeural` via `edge-tts` e requer FFmpeg no PATH. No Windows, há a alternativa `--speech-engine sapi --voice "Microsoft Maria Desktop"`. São gerados slides HTML, imagens, roteiro, MP4, legendas SRT/WebVTT e manifest de hashes. Cada slide complexo é seguido por explicação simples e exemplo cotidiano; a cobertura determina a duração.

Para reaproveitar narrações de uma exportação anterior, acrescente
`--reuse-audio-dir artifacts/presentation/audio`. Apenas texto, voz e ritmo
idênticos reutilizam o áudio; a pasta original permanece intacta. Congele uma
cópia das evidências antes de exportar. O manifest registra os hashes de
entrada e recusa mudanças durante a geração. A reprodução exata do snapshot
de evidências publicado usa `evidence-snapshot.json` do ZIP da respectiva release; evidências
atuais podem registrar uma classificação posterior.

`presentation.review` verifica formato, timeline e legendas e extrai quadros. A transcrição integral automática é opcional: instale `pip install -e ".[review]"` e acrescente `--asr`; o modelo ASR pode precisar de download inicial. Nenhuma dessas verificações declara que uma pessoa assistiu ou ouviu o vídeo inteiro.

O site estático é montado em `site/dist/`, com evidência pública e arquivos permitidos pelo manifest; datasets, pesos e caminhos locais não são publicados. O servidor acima é apenas preview local. Hospedagem e acesso público são verificados separadamente; o build não faz deploy.

O mirror da hospedagem Sites limita cada arquivo a 32 MiB. Para usar o vídeo publicado na release GitHub e manter slides/legendas no site:

```powershell
python site/build.py --run-dir artifacts/runs/round10-offset-bpe-frequency-public25 --presentation-dir artifacts/presentation-v02 --video-url https://github.com/aikolopes-DS/fraudlab/releases/download/v0.2.0/video.mp4
```

O build ainda confere o hash do MP4 local, registra o hash e a URL externa no manifesto público e exclui o binário de `site/dist/`. A igualdade dos bytes publicados e a reprodução anônima precisam ser verificadas após o upload.

## Pacote único da AED

`scripts/package_delivery.py` reúne vídeo, slides, legendas, código de um commit
Git explícito e provas. Antes de empacotar, confere os hashes dos arquivos e
exige revisão audiovisual, ASR e cobertura visual de todos os slides. Dados,
credenciais, checkpoints e links simbólicos no código são recusados.

```powershell
$fraudDeliverySource = git rev-parse HEAD
python scripts/package_delivery.py --presentation-dir artifacts/presentation-v03 --snapshot artifacts/presentation-input-v03/evidence.json --source-ref $fraudDeliverySource --output artifacts/delivery/AED-FraudLab-entrega.zip
```

Extraia o pacote e abra `ENTREGA.html`. O vídeo e os slides abrem localmente;
o código e as provas acompanham a entrega. `MANIFEST.json` lista os hashes de
cada arquivo. Saídas existentes não são sobrescritas. Os testes de empacotamento
usam arquivos fabricados e não substituem a revisão do vídeo real.

## Experimentos históricos e testes

As ablações históricas de 17/11 podem ser reproduzidas em um clone com os mesmos CSVs e novos
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

`--model glm --glm-value-one-hot` é outra ablação histórica, fora da R10: usa
one-hot esparso com suporte mínimo de cinco, L-BFGS até 2.000 iterações e rejeita
ajustes sem convergência. Não ativa o offset nem representa a receita escolhida.

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

## TabM — rodada 8 descartada

O [plano R8](../evidence/round8-plan.json) foi registrado antes do treino.
O [resultado R8](../evidence/round8-results.json) não justificou incluir a rede.
Para reproduzir o experimento, instale as dependências opcionais oficiais:

```powershell
python -m pip install -e ".[tabm]"
```

Somente depois de congelar e revisar o código e o plano:

```powershell
python -m fraudlab train --name round8-tabm --model tabm --feature-set value_digits --phase exploratory_full_data_cv --split stratified --folds 5 --seed 42 --holdout-fraction 0 --epochs 40 --patience 6 --batch-size 1024 --neural-learning-rate 0.001 --weight-decay 0.0001 --target-smoothing 20 --threads 2 --device cuda --hidden-dims 256 256 --dropout 0.1 --neural-numeric-transform standard --neural-categorical-transform target --validation-note "Prospective TabM with inner 15% epoch selection and fresh fixed-epoch outer refits; CV 5/42 reused for selection, no independent holdout."
```

O padrão de dados e saídas é o da CLI; `--data` e `--output` aceitam caminhos
absolutos para uma worktree isolada. A arquitetura fixa é `2×256`, `k=16`,
embeddings lineares por partes versão B, dimensão 12 e até 48 intervalos. Os
demais parâmetros neurais são lidos da CLI e registrados. Opções
incompatíveis são recusadas: `--fixed-training`, outro `--hidden-dims` ou
codificação categórica por embeddings. Não há protocolo de refit em 100%
para este candidato; o teste usa a média dos cinco checkpoints dos folds externos.

Cada fold salva `.pt` com pesos, intervalos, configuração e época escolhida,
além de `.preprocessing.joblib` com pré-processador e máscara de constantes.
`predict_saved("tabm", ...)` reconstrói a mesma previsão. Os testes em CPU
usam dados fabricados e não comprovam desempenho competitivo.
