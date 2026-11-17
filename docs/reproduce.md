# Reproduzir o FraudLab

Os comandos usam PowerShell na raiz do repositório. A receita selecionada é a [R18](../evidence/round18-plan.json), com núcleo de ML no commit [8a1c7fd](https://github.com/aikolopes-DS/fraudlab/tree/8a1c7fd141b43e41b0d43e3a07b0c5ad0bf41ae5): **56,25% offset próprio com BPE, frequência, pooling e interações, 18,75% LightGBM próprio e 25% GPT-2 externo aprovado**. O arquivo público atual usa seus [refits R20 em 100% dos rótulos](../evidence/round20-plan.json), sem nova OOF. O acompanhamento único fica no [README](../README.md); a evidência pública fica em [public.json](../evidence/public.json).

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

Para importar o núcleo congelado sem retirar do checkout principal os scripts posteriores, crie esta worktree em um clone novo:

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

## Receita selecionada: R18

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

## Resultado público e submissão

O arquivo atual R20 é a submissão **58483010**, score público **0,95162**, posição **657/2.126** em **17/11/2026 às 21h59min16s UTC**. O snapshot e o SHA conferido com o arquivo remoto estão em [round20-public.json](../evidence/round20-public.json) e [public.json](../evidence/public.json). Sua referência OOF 0,9516524825255244 pertence à seleção R18; o refit não tem OOF própria. Ranking é provisório e primeiras posições ainda não foram alcançadas.

A submissão CV anterior R18, **58481744**, recebeu **0,95141**, com posição **714/2.125** em **17/11/2026 às 20h53min43s UTC**; sua prova permanece em [round18-public.json](../evidence/round18-public.json).

Depois da auditoria do CSV e de uma decisão explícita de envio:

```powershell
& $fraudKaggle competitions submit playground-series-s6e11 -f artifacts/runs/round20-refit-joint-public25/submission.csv -m "FraudLab: round20-refit-joint-public25"
```

O comando faz um envio real, sujeito às regras e limites da conta. A confirmação exige processamento na plataforma; um CSV válido não comprova score. [record_submission.py](../scripts/record_submission.py) registra arquivo, resposta e snapshot: consulte seus argumentos em `python scripts/record_submission.py --help`. Uma reprodução não herda a pontuação já processada.

## Histórico e notebook

A [R10](../evidence/round10-results.json), fonte 169f3ee, é a receita anterior: OOF 0,9514921382152453 e submissão 58470267/0,95134. A R18 preserva seu mecanismo de offset e acrescenta pooling e interações. A [R9](../evidence/public-evaluation-results.json) permanece uma exceção pública exploratória declarada; seu critério original não foi reclassificado como cumprido. Os demais planos/resultados documentam comparações próprias, incluindo MLPs, TabM, CatBoost, tokens, dados originais e frequência transdutiva; resultados negativos não promovem uma receita.

O [notebook V6](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) incorpora os 22 módulos do core `8a1c7fd`, os vocabulários GPT-2 oficiais com hashes e as células para CV R18 e refit completo R20. Foi salvo **sem execução na nuvem**: suas 39 células foram [conferidas com a versão remota](../evidence/notebook-v6.json), com outputs e contagens de execução vazios. O preflight local do tokenizador passou sem rede; contratos com callbacks simulados não comprovam execução numérica no Kaggle. O notebook exige as versões de dependências declaradas e não instala pacotes nem submete automaticamente.

A reserva inicial de 15% seed 2026 é histórica: esses rótulos já participam das CVs posteriores. Não a reapresente como avaliação independente atual. Os refits `final-glm`/`final-lgbm` e `compose_refit_hybrid.py` pertencem à receita antiga e não são os refits R20.

## Testes, apresentação e entrega

Para a suíte com as comparações neurais em CPU e os geradores:

```powershell
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,bpe,tabm,presentation]"
python -m pytest tests presentation site/test_build.py
```

Testes usam dados fabricados; não medem qualidade competitiva nem substituem as auditorias dos checkpoints reais. CUDA é uma instalação opcional separada para reproduzir redes próprias; siga o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

A release publicada [v0.3](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0) preserva a apresentação da R10, com 139 slides e 54min51s. A atualização de fontes para R18 não equivale a um novo vídeo publicado. O fluxo de geração, comparações necessárias, vídeo, revisão e site está em [presentation/README.md](../presentation/README.md). `evidence/media.json` e `evidence/publication.json` registram publicação e QA reais; ASR e quadros não significam escuta humana integral.

O empacotador exige uma exportação final com manifest, snapshot, revisão audiovisual/visual e um commit explícito de código. Exemplo para os artefatos históricos v0.3 disponíveis localmente:

```powershell
python scripts/package_delivery.py --presentation-dir artifacts/presentation-v03 --snapshot artifacts/presentation-input-v03/evidence.json --source-ref ef07bdc9f0e07a34d5082a04df956332fbe7df84 --output artifacts/delivery/AED-FraudLab-v03-reproduzida.zip
```

Extraia o ZIP e abra `ENTREGA.html`; o manifest lista hashes. Exportações novas precisam de seus próprios artefatos, snapshot, revisão e fonte correspondentes. Dados, credenciais e checkpoints não entram no pacote. Gerar o site ou ZIP local não publica conteúdo nem envia a atividade ao portal da faculdade.
