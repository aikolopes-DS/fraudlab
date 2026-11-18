# Reproduzir o FraudLab

Os comandos usam PowerShell na raiz do repositório. A receita selecionada atual é a [R23](../evidence/round23-results.json): **42,1875% offset próprio, 14,0625% RealMLP própria, 18,75% LightGBM próprio e 25% GPT-2 externo aprovado**, em probabilidades. A RealMLP foi refeita em todos os 699.960 rótulos; os dois modelos completos da R20 são reutilizados. A fonte neural é [11b139f](https://github.com/aikolopes-DS/fraudlab/tree/11b139f77206101d3f5beeb5648fb459f4cd304a) e o script de refit é [c95ff46](https://github.com/aikolopes-DS/fraudlab/tree/c95ff468619d299c94764b45e5e15af5ed9f4674). Não há nova OOF desse refit. O acompanhamento único fica no [README](../README.md); a evidência pública fica em [public.json](../evidence/public.json).

Este guia preserva também o **baseline histórico R18/R20**, executável no notebook. Seus três componentes e pesos diferem da R23; não confunda um CSV desse baseline com o arquivo atual da competição.

Use um clone novo ou nomes de execução novos. Os comandos não sobrescrevem `artifacts/runs/`. Uma reprodução em outro ambiente pode produzir bytes diferentes; ela precisa de verificação própria e não herda o score de outro arquivo.

## Ambiente e fonte

O núcleo aceita Python 3.10 ou superior. A CI usa 3.11; as execuções locais registram suas versões nos artefatos.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,bpe]"
```

Sem ativação, use `.\.venv\Scripts\python.exe`. No Linux, crie o ambiente com `python3.11 -m venv .venv` e ative com `source .venv/bin/activate`.

Para o baseline R18/R20, importe o núcleo histórico congelado sem retirar do checkout principal os scripts posteriores. Crie esta worktree em um clone novo:

```powershell
git worktree add --detach .local/worktrees/joints-counts 8a1c7fd141b43e41b0d43e3a07b0c5ad0bf41ae5
git -C .local/worktrees/joints-counts rev-parse HEAD
git -C .local/worktrees/joints-counts status --porcelain
$env:PYTHONPATH = (Resolve-Path .local/worktrees/joints-counts/src).Path
```

Se ela já existir, confira o SHA e a ausência de alterações em vez de criá-la novamente. Os comandos de treino abaixo importam esse núcleo por `PYTHONPATH`; dados e saídas continuam relativos à raiz principal.

A receita R18 roda em CPU. `joint_bpe` usa `tiktoken==0.14.0` e `regex==2026.9.29` para chaves numéricas. A primeira inicialização baixa os vocabulários oficiais e verifica seus hashes; preserve `TIKTOKEN_CACHE_DIR` para executar depois sem internet. Essa tokenização não treina uma rede GPT-2 nem chama uma API paga.

## Dados e autenticação

Inscreva-se na competição e aceite as [regras da S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/rules) na própria plataforma. A CLI oficial utilizada tem ambiente separado e login pelo navegador:

```powershell
py -3.11 -m venv .local/kaggle-cli
.\.local\kaggle-cli\Scripts\python.exe -m pip install kaggle==2.2.5 six==1.17.0
$fraudKaggle = ".\.local\kaggle-cli\Scripts\kaggle.exe"
& $fraudKaggle auth login
& $fraudKaggle competitions download playground-series-s6e11 -p data/raw --unzip
& $fraudKaggle datasets download masayakawamata/s6e11-gpt2-target-first-oof/2 -p .local/research/public-models/gpt2 --unzip
python -m fraudlab inspect --data data/raw --output artifacts/profile.json
```

O login só é necessário quando a CLI ainda não estiver autenticada. Credenciais ficam no mecanismo privado da CLI, fora do código e do Git. Também ficam fora do Git os dados, modelos e saídas locais.

São esperados `train.csv` (699.960 linhas rotuladas), `test.csv` (299.984 linhas sem alvo) e `sample_submission.csv`. O perfil registra tipos, ausências, duplicatas e SHA-256. A reprodução auditada exige os mesmos bytes; trocar o CSV ou sua serialização pode invalidar os planos fixados.

As [previsões GPT-2 de Masaya Kawamata, versão 2](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof/versions/2) são CC BY 4.0. O [treinamento público MIT](https://github.com/Matapanino/s6e11-gpt2-target-first/tree/8446fb73c11583df4f7006aa08fa34472249b318) foi realizado pelo autor externo. Créditos e verificação estão em [external-gpt2.json](../evidence/external-gpt2.json).

```powershell
python -m fraudlab import-public-gpt2 --source .local/research/public-models/gpt2 --data data/raw --name explore5-public-gpt2-v2
```

O importador confere o manifesto aprovado, hashes dos arrays e dados, formas, finitude, IDs e folds 5/42; não executa o código externo. Converte logits em probabilidades por sigmoid. No teste público, o autor já fez a média dos cinco logits antes dessa conversão.

## Receita atual: R22 em CV e refit R23

A [R22](../evidence/round22-results.json) completou cinco folds próprios em todos os rótulos, com seleção de E em 15% internos de cada treino externo. Todo estado interno é descartado antes do ajuste novo do fold. Sua RealMLP usa PyTabKit 1.8.0, 53 vistas (20 numéricas e 33 categóricas) dos 24 atributos brutos, 16 membros e camadas 512→256→128. Os E observados foram [3,2,3,3,3]. A híbrida escolhida nessa rodada tem OOF exploratória **0,9517074903582017**; esses folds participaram da seleção, sem reserva independente.

Prepare uma fonte isolada para essa rede, distinta do core histórico do notebook:

```powershell
git worktree add --detach .local/worktrees/realmlp-cv 11b139f77206101d3f5beeb5648fb459f4cd304a
$env:PYTHONPATH = (Resolve-Path .local/worktrees/realmlp-cv/src).Path
python -m pip install -e ".[bpe,realmlp]"
```

Para iniciar uma **nova CV real** pela CLI, use os argumentos públicos originais, sem copiar manualmente dezenas de padrões. O nome da execução deve estar livre:

```powershell
$fraudR22Plan = Get-Content evidence/round22-plan.json -Raw -Encoding UTF8 | ConvertFrom-Json
$fraudR22Args = @($fraudR22Plan.cli_argv)
python -m fraudlab @fraudR22Args
```

Esse comando exige CUDA, conforme o plano; não há fallback silencioso para CPU. Os bundles guardam transformadores e pesos, e a média do teste combina probabilidades dos cinco folds. Mudar hardware ou versões requer nova auditoria e pode mudar bytes e métricas.

Na [R23](../evidence/round23-plan.json), a mediana E=3 foi congelada antes de treinar uma rede nova em todos os rótulos, sem validação nem seleção adicional. Ela contém 16 membros e **6.128.752 parâmetros** no bundle efetivamente ajustado. Offset e LightGBM completos da R20 e previsões GPT-2 v2 foram reutilizados. A [auditoria](../evidence/round23-verification.json) verificou a recarga da rede, os modelos reutilizados e a mistura das 299.984 linhas. `roc_auc=null`, `fold_scores=[]` e `rows.oof=0` são obrigatórios no novo refit. A OOF citada continua pertencendo à R22.

O script auditado é um **replay condicionado aos artefatos exatos do plano**, não uma reprodução autossuficiente em um clone vazio:

```powershell
git show c95ff468619d299c94764b45e5e15af5ed9f4674:scripts/refit_neural_cv.py
python scripts/refit_neural_cv.py --plan evidence/round23-plan.json --workspace . --core-worktree .local/worktrees/realmlp-cv
```

O segundo comando só deve ser executado após reconstruir e auditar todos os inputs de `inputs` e `proofs` do plano, incluindo a CV R22, os dois refits R20, os arrays públicos e os recibos locais ignorados pelo Git. Ele confere hashes antes de ajustar, recusa saídas existentes e exige a worktree limpa pinada. Checkpoints e recibos privados não estão incorporados no repositório ou notebook. Uma nova execução com bytes diferentes precisa de plano e recibos próprios; não edite os registros históricos para simular compatibilidade. O script não faz submissão.

As versões exatas usadas nesse replay estão em `dependency_versions` do plano: NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, torch 2.11.0+cu128, PyTabKit 1.8.0, pytorch-lightning 2.6.6, torchmetrics 1.9.0, lightning-utilities 0.15.3, cloudpickle 3.1.2, tiktoken 0.14.0, regex 2026.9.29 e LightGBM 4.7.0. O extra `realmlp` fixa a biblioteca e seus adaptadores; a instalação CUDA e a compatibilidade do equipamento devem ser preparadas separadamente. Este texto não declara execução desse ambiente no Kaggle.

## Baseline histórico: seleção R18 e refit R20

Antes dos comandos deste baseline, volte ao seu core próprio pinado:

```powershell
$env:PYTHONPATH = (Resolve-Path .local/worktrees/joints-counts/src).Path
```

A CV estratificada usa cinco folds, seed 42, sobre todos os 699.960 rótulos. Cada previsão OOF exclui o próprio fold do ajuste supervisionado; o teste recebe a média das cinco probabilidades por família própria. Essa OOF foi reutilizada para seleção: não existe holdout independente nesta etapa.

Execute os treinos sequencialmente para limitar RAM:

```powershell
$fraudR18Plan = Get-Content evidence/round18-plan.json -Raw -Encoding UTF8 | ConvertFrom-Json
$fraudCvArgs = @(
    "--data", "data/raw", "--phase", "exploratory_full_data_cv",
    "--split", "stratified", "--folds", "5", "--seed", "42",
    "--holdout-fraction", "0", "--fixed-training",
    "--validation-note", $fraudR18Plan.validation
)
python -m fraudlab train @fraudCvArgs --model offset --name round18-joint-bpe-pooled-offset --feature-set joint_bpe --glm-frequency --glm-pooled-tokens --glm-inner-folds 5 --glm-spline-knots 0 --logistic-c 0.3 --target-smoothing 50 --iterations 1000 --num-leaves 4 --learning-rate 0.03 --l2-leaf-reg 5 --min-child-samples 100 --threads 4 --device cpu
python -m fraudlab train @fraudCvArgs --model lightgbm --name explore5-lgbm --feature-set value_digits --threads 6 --device auto --iterations 1438 --num-leaves 4 --learning-rate 0.04 --l2-leaf-reg 5 --min-child-samples 100 --target-smoothing 20 --patience 150
python -m fraudlab blend --runs artifacts/runs/round18-joint-bpe-pooled-offset artifacts/runs/explore5-lgbm artifacts/runs/explore5-public-gpt2-v2 --weights 0.5625 0.1875 0.25 --name round18-pooled-share100-public25
python -m fraudlab validate-submission --file artifacts/runs/round18-pooled-share100-public25/submission.csv --sample data/raw/sample_submission.csv
```

`joint_bpe` preserva a base BPE e acrescenta oito interações fixas. Suas **92 colunas** (24 numéricas e 68 categóricas) produzem **223 dimensões codificadas** no offset, incluindo 16 logits de pooling. O GLM usa Newton-Cholesky, C=0,3 e target encoding interno de cinco folds com suavização 50. O pooling usa suavizações 50/2.000/200 e prior do treino interno. Encodings supervisionados são cruzados; validação e teste usam apenas os mapas do treino externo. A frequência da R18 também usa somente esse treino, sem contagens transdutivas ou dados originais adicionais.

O GLM e as 1.000 árvores residuais usam a mesma matriz cruzada; a inferência soma as margens e aplica sigmoid. As árvores têm quatro folhas, taxa 0,03, L2=5 e mínimo de 100 exemplos por folha. O LightGBM separado mantém 1.438 árvores, taxa 0,04 e suavização 20; nesse modelo a implementação usa CPU, inclusive com o `device=auto` histórico. `--fixed-training` impede seleção do número de árvores pela validação externa.

A mistura é em **probabilidades**, com pesos **0,5625/0,1875/0,25**. O [resultado R18](../evidence/round18-results.json) registra OOF híbrida **0,9516524825255244**, delta **+0,0001603443102791** e melhora em cinco folds sobre a R10. Os três shares prospectivos e a [auditoria](../evidence/round18-verification.json) permanecem registrados; reproduzir a escolhida não cria uma nova evidência independente.

Cada run salva configuração, status, métricas, OOF, previsões e submissão; treinos próprios salvam checkpoints. `blend` exige compatibilidade dos hashes dos dados, IDs, ordem, folds e rótulos, e propaga os créditos externos. Os scripts seguintes exigem ainda os hashes exatos dos artefatos originais: um novo ajuste com bytes diferentes precisa de um novo registro prospectivo, sem substituir silenciosamente esses pinos.

## R20: refit em todos os rótulos

O [plano R20](../evidence/round20-plan.json) define dois novos ajustes sequenciais: offset R18 e LightGBM selecionado, cada um com os 699.960 rótulos, seed 42 e parâmetros congelados. A previsão passa de uma média de cinco modelos de 80% para um modelo de 100% por família. O componente GPT-2 v2 e os pesos 0,5625/0,1875/0,25 permanecem fixos.

Após revisão e autorização da execução, com os inputs auditados disponíveis:

```powershell
python scripts/refit_selected_cv.py --plan evidence/round20-plan.json --root . --source-worktree .local/worktrees/joints-counts
```

O script especializado verifica fonte limpa 8a1c7fd, hashes do plano, código, dados, seleção e checkpoints, e recarrega os novos modelos. A saída composta prevista é `round20-refit-joint-public25`. O contrato é **phase=full_data_refit, roc_auc=null, fold_scores=[], rows.oof=0**, sem fabricar `oof.csv`. A métrica anterior entra somente como `development_roc_auc`, com escopo histórico explícito; não mede esses refits. Não há seleção por validação externa nem envio automático.

A CLI comum `fraudlab refit` continua ligada ao seu contrato de holdout independente. A R20 usa o script acima porque sua seleção veio da CV exploratória. O ajuste em mais linhas não constitui uma nova estimativa local de desempenho. Sua [auditoria](../evidence/round20-verification.json) e [avaliação pública](../evidence/round20-public.json) têm registros separados.

## Inferência R20 com modelos salvos

Os dois refits auditados podem prever o lote oficial sem repetir treino ou aprender novos mapas. Execute a API abaixo numa sessão Python do ambiente com o núcleo 8a1c7fd. Os checkpoints locais precisam coincidir com a prova publicada; os CSVs públicos de GPT-2 correspondem somente aos IDs desse lote.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fraudlab.data import fingerprint, submission, validate_submission
from fraudlab.features import transform_features
from fraudlab.models import predict_saved

plan = json.loads(Path("evidence/round20-plan.json").read_text(encoding="utf-8"))
proof = json.loads(Path("evidence/round20-verification.json").read_text(encoding="utf-8"))
model_hashes = {item["role"]: item["model_sha256"] for item in proof["checkpoints"]}
for name in ("test.csv", "sample_submission.csv"):
    assert fingerprint(Path("data/raw") / name) == plan["data_hashes"][name]
test = pd.read_csv("data/raw/test.csv")
sample = pd.read_csv("data/raw/sample_submission.csv")
scores = []
for role, feature_set in (("offset", "joint_bpe"), ("lightgbm", "value_digits")):
    checkpoint = Path("artifacts/runs") / plan["refit_names"][role] / "model"
    assert fingerprint(checkpoint.with_suffix(".joblib")) == model_hashes[role]
    features = transform_features(test.drop(columns="id"), feature_set)
    score = predict_saved(role, checkpoint, features, "cpu")
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    scores.append(score)
    del features
external = Path(plan["inputs"]["external"]["path"]) / "predictions.csv"
assert fingerprint(external) == plan["inputs"]["external"]["sha256"]["predictions.csv"]
public = pd.read_csv(external)
validate_submission(public, sample, "id", "is_fraud")
probability = 0.5625 * scores[0] + 0.1875 * scores[1] + 0.25 * public["is_fraud"].to_numpy()
result = submission(test["id"], probability, sample, "is_fraud")
validate_submission(result, sample, "id", "is_fraud")
output = Path(".local/replayed-r20.csv")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8", newline="") as stream:
    result.to_csv(stream, index=False)
print({"file": str(output), "sha256": fingerprint(output)})
```

Esse fluxo só faz inferência; não gera OOF, não calcula uma nova AUC e não submete o arquivo. A sigmoid do offset já inclui GLM e residual: não acrescente o `init_score` novamente. Mantenha os modelos e o componente externo auditados da mesma receita.

## R21: mistura aprendida com validação aninhada

O [plano R21](../evidence/round21-plan.json) fixa três bases próprias: `round18-joint-bpe-pooled-offset`, `round19-transductive-pooled-offset` e `explore5-lgbm`. O método é `nested_owned_logit_holdout_stack`; o plano e os resultados efetivamente registrados determinam seu estado, sem score presumido nesta documentação.

Em cada treino externo de 559.968 linhas, uma divisão estratificada seed 2026 separa 419.976 linhas (75%) para **novos ajustes das três bases** e 139.992 (25%) para ajustar o meta. O meta recebe apenas seus três logits, limitados por clip 1e-6, e usa LogisticRegression C=0,01, Newton-Cholesky, tolerância 1e-7 e máximo de 200 iterações. Não recebe features brutas, IDs, rótulos externos ou GPT-2.

Para prever a validação externa e o teste, aplica-se o meta aos **checkpoints externos já existentes do mesmo fold**, treinados em 559.968 linhas. Eles não são refitados nessa aplicação. A R19 mantém suas contagens globais de features sem rótulos; suas demais transformações supervisionadas continuam locais ao treino correspondente. Há uma diferença declarada de distribuição: bases do treino meta veem 60% dos rótulos totais; bases da inferência veem 80%.

Somente após revisão, publicação do plano e liberação de memória/execução:

```powershell
python scripts/train_nested_stack.py --plan evidence/round21-plan.json --source-tree .local/worktrees/joints-counts --output artifacts/runs
```

O protocolo prevê 15 novos fits-base e cinco metas, persistidos com hashes. A previsão própria de teste é a média de cinco probabilidades meta. Só depois entra o GPT-2: **75% stack próprio + 25% Masaya aprovado**. Os coeficientes aprendidos nos logits não são percentuais de probabilidade nem precisam somar um. Convergência inadequada invalida a execução, sem retry automático.

O gate prospectivo exige híbrida acima da R18, delta estritamente maior que 0,0001 sobre a R10 e pelo menos quatro folds melhores que a R18. Não há novo holdout independente; a OOF continua exploratória. O script não submete ao Kaggle.

## R22: inferência da comparação RealMLP

A [R22](../evidence/round22-plan.json) introduziu a RealMLP posteriormente incluída na R23. Seu adaptador usa a biblioteca oficial PyTabKit 1.8.0: aprende 53 vistas dos 24 atributos brutos, escolhe E em 15% internos e descarta esse modelo antes de um ajuste novo no treino externo. O horizonte dos schedules permanece em três épocas mesmo quando E=1. Os mapas, vocabulários e pesos entram no bundle local `.pkl`; valores categóricos novos usam o índice zero.

Para usar esse suporte, instale `python -m pip install -e ".[realmlp]"` no ambiente e volte à worktree neural com `$env:PYTHONPATH = (Resolve-Path .local/worktrees/realmlp-cv/src).Path`. O extra fixa as dependências oficiais e exige scikit-learn >=1.7. A inferência CPU não exige GPU.

Após completar e auditar os cinco checkpoints do run, a API aceita apenas os 24 atributos brutos, sem `id` ou alvo. Em uma sessão Python com `test` carregado do CSV oficial:

```python
from pathlib import Path
from fraudlab.models import predict_saved

run = Path("artifacts/runs/round22-realmlp53")
raw_features = test.drop(columns="id")
probability = sum(
    predict_saved("realmlp", run / f"fold-{fold}", raw_features, "cpu")
    for fold in range(5)
) / 5
```

Cada checkpoint reconstrói os mapas armazenados e a média das probabilidades de seus 16 membros; a média acima combina os cinco folds da R22. Não ajuste transformadores durante a inferência nem passe dados por `joint_bpe` nesse modelo. Esse exemplo reproduz a família CV, enquanto a R23 usa o bundle único `round23-refit-realmlp/model.pkl` treinado em todos os rótulos. Para ele, use `predict_saved("realmlp", Path("artifacts/runs/round23-refit-realmlp/model"), raw_features, "cpu")`, sem a média de cinco folds. A receita atual ainda requer as outras três contribuições e seus pesos registrados.

## Resultado público e submissão

O arquivo atual R23 é a submissão **58490653**, score público **0,95183**, posição **626/2.135** em **18/11/2026 às 02h12min21s UTC**. O [snapshot R23](../evidence/round23-public.json) comprova que o CSV remoto e local têm SHA-256 `937dc4dd355229bc49fda67c3099ea97867a23fb9c9bbe97a2bf57fae6cf24de`. A referência OOF 0,9517074903582017 pertence à seleção R22; o refit não tem OOF própria. A R20 anterior obteve 0,95162 no [envio 58483010](../evidence/round20-public.json). Ranking é provisório e primeiras posições ainda não foram alcançadas.

A submissão CV anterior R18, **58481744**, recebeu **0,95141**, com posição **714/2.125** em **17/11/2026 às 20h53min43s UTC**; sua prova permanece em [round18-public.json](../evidence/round18-public.json).

Depois da auditoria do CSV e de uma decisão explícita de envio:

```powershell
& $fraudKaggle competitions submit playground-series-s6e11 -f artifacts/runs/round23-refit-realmlp-offset025-public25/submission.csv -m "FraudLab: round23-refit-realmlp-offset025-public25"
```

O comando faz um envio real, sujeito às regras e limites da conta. A confirmação exige processamento na plataforma; um CSV válido não comprova score. [record_submission.py](../scripts/record_submission.py) registra arquivo, resposta e snapshot: consulte seus argumentos em `python scripts/record_submission.py --help`. Uma reprodução não herda a pontuação já processada.

## Histórico e notebook

A [R10](../evidence/round10-results.json), fonte 169f3ee, é a receita anterior: OOF 0,9514921382152453 e submissão 58470267/0,95134. A R18 preserva seu mecanismo de offset e acrescenta pooling e interações. A [R9](../evidence/public-evaluation-results.json) permanece uma exceção pública exploratória declarada; seu critério original não foi reclassificado como cumprido. Os demais planos/resultados documentam comparações próprias, incluindo MLPs, TabM, CatBoost, tokens, dados originais e frequência transdutiva; resultados negativos não promovem uma receita.

O [notebook V7 público](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) preserva o baseline executável R18/R20 (22 módulos de `8a1c7fd` e vocabulários GPT-2 oficiais) e acrescenta um apêndice de código atual R23. O gerador `python scripts/kaggle_notebook.py` produz esses dois escopos. O apêndice exporta em `r23-source/` os 23 módulos de `11b139f`, dois scripts de `c95ff46` e dois planos, com SHA-256. Para caber no limite de fonte do Kaggle, os vocabulários oficiais usam LZMA/base64 e parte do apêndice usa gzip/base64; descompactar preserva os bytes e os hashes originais. `realmlp.py` e `refit_neural_cv.py` continuam diretamente legíveis em células próprias. O verificador reconstrói os 49 arquivos e recusa o JSON transmitido com tamanho igual ou maior que 1.000.000 bytes. Ele só grava e confere arquivos; não importa esse core, não treina nem faz inferência. O baseline continua produzindo a composição histórica de três componentes, sem se fazer passar pela receita atual de quatro componentes.

A [prova V7](../evidence/notebook-v7.json) registra o save público real e o readback integral: 45 células, todas iguais às locais após normalização documentada de strings e de um timestamp, com contagens de execução e outputs vazios. Os 49 arquivos reconstituídos coincidiram com as fontes pinadas. O payload transmitido tem 991.610 bytes. **A V7 foi salva sem executar e sem treinamento na nuvem.** A [prova V6](../evidence/notebook-v6.json) permanece histórica. O preflight local do tokenizador e os testes de exportação verificam integridade e contratos; não comprovam treinamento numérico no Kaggle. O notebook não instala pacotes nem submete automaticamente.

A reserva inicial de 15% seed 2026 é histórica: esses rótulos já participam das CVs posteriores. Não a reapresente como avaliação independente atual. Os refits `final-glm`/`final-lgbm` e `compose_refit_hybrid.py` pertencem à receita antiga e não são os refits R20.

## Testes, apresentação e entrega

Para a suíte com as comparações neurais em CPU e os geradores:

```powershell
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,bpe,tabm,realmlp,presentation]"
python -m pytest tests presentation site/test_build.py
```

Testes usam dados fabricados; não medem qualidade competitiva nem substituem as auditorias dos checkpoints reais. CUDA é uma instalação opcional separada para reproduzir redes próprias; siga o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

A release atual [v0.4.0](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.4.0) publica a apresentação da R23: **198 slides, 78 pares técnico/exemplo, 62 fórmulas, 13 trechos de código, 87min58s e 1.181 legendas**. O [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) oferece player e materiais; os quatro assets da release foram baixados sem autenticação e conferidos por hash, e a entrega offline teve 471 arquivos verificados. A fonte incluída no ZIP é `17d43d70641f6cda4d09f2a797f633450dba42ec`. A [v0.3 histórica](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0) preserva a R10, com 139 slides e 54min51s.

O fluxo de geração V04, com a lista completa de comparações R22/RealMLP e os refits R23, está em [presentation/README.md](../presentation/README.md). A [prova de mídia](../evidence/media.json) registra decodificação e ASR integrais e revisão visual dos 198 slides; a [prova de publicação](../evidence/publication.json) separa release e site. Isso não declara escuta humana nem reprodução contínua integral. No HTML a 390px, fórmulas longas podem exigir paisagem ou zoom.

O empacotador exige uma exportação final com manifest, snapshot, revisão audiovisual/visual e um commit explícito de código. Exemplo V04 com os artefatos finais locais, sem sobrescrever a entrega publicada:

```powershell
python scripts/package_delivery.py --presentation-dir artifacts/presentation-v04 --snapshot artifacts/presentation-input-v04/evidence.json --source-ref 17d43d70641f6cda4d09f2a797f633450dba42ec --output artifacts/delivery/AED-FraudLab-v04-reproduzida.zip
```

Exemplo histórico V03, apenas para seus próprios artefatos:

```powershell
python scripts/package_delivery.py --presentation-dir artifacts/presentation-v03 --snapshot artifacts/presentation-input-v03/evidence.json --source-ref ef07bdc9f0e07a34d5082a04df956332fbe7df84 --output artifacts/delivery/AED-FraudLab-v03-reproduzida.zip
```

Extraia o ZIP e abra `ENTREGA.html`; o manifest lista hashes. Exportações novas precisam de seus próprios artefatos, snapshot, revisão e fonte correspondentes. Dados, credenciais e checkpoints não entram no pacote. Gerar o site ou ZIP local não publica conteúdo nem envia a atividade ao portal da faculdade.
