# Requisitos e arquitetura

## Objetivo e contratos

Participar de uma competição aberta com intenção de vencer, construir uma solução explicável e publicar código, vídeo e estudo de caso como portfólio. A atividade de Redes Neurais é parte desse objetivo. A liderança continua uma meta não alcançada.

O projeto exige regras conferidas, treinamento neural próprio real, comparação de alternativas, previsões submetidas de fato e posição observada. Scores locais, score público e ranking final são registros distintos. Nenhuma flag de evidência é inferida de um build, teste sintético ou arquivo gerado.

A apresentação cobre problema, proposta, construção, matemática, arquiteturas, experimentos, códigos e ranking. Cada slide complexo deve ser seguido imediatamente por explicação simples com exemplo cotidiano. Narração e legendas são em português, com duração determinada pela cobertura.

## Protocolo atual: seleção R22 e refit R23

A [R22](../evidence/round22-plan.json) acrescentou uma família neural própria às bases já auditadas. A [R23](../evidence/round23-results.json) refez essa rede em todos os 699.960 rótulos, mantendo os dois modelos completos da R20 e o GPT-2 público v2. A soma ocorre em probabilidades:

| Componente atual | Origem e execução | Peso efetivo |
| --- | --- | --- |
| Offset próprio | GLM com BPE, frequência, pooling e oito interações; árvores residuais; checkpoint completo R20 reutilizado | 42,1875% |
| RealMLP própria | PyTabKit 1.8.0; novo ajuste em todos os rótulos, 16 membros, camadas 512→256→128 e E=3 | 14,0625% |
| LightGBM próprio | Checkpoint completo R20 reutilizado, 1.438 árvores fixas | 18,75% |
| GPT-2 externo | Previsões de Masaya Kawamata v2, sem novo treino nosso | 25% |

O core neural está fixado em [11b139f](https://github.com/aikolopes-DS/fraudlab/tree/11b139f77206101d3f5beeb5648fb459f4cd304a); os scripts de refit estão em [c95ff46](https://github.com/aikolopes-DS/fraudlab/tree/c95ff468619d299c94764b45e5e15af5ed9f4674). A RealMLP transforma os 24 atributos brutos em 53 vistas, com 20 numéricas e 33 categóricas. Medianas, quantis, vocabulários e target encoding interno pertencem somente ao treino de cada etapa. A arquitetura usa bases numéricas periódicas PBLD, embeddings categóricos, SiLU paramétrica, Adam e schedules registrados. Seu bundle completo tem 6.128.752 parâmetros e guarda transformadores e os 16 membros treinados.

Na CV R22, cada treino externo separou 15% internos para escolher E em até três épocas, descartou todo esse estado e ajustou novamente em todo o treino externo. Os cinco E foram [3,2,3,3,3]. A R23 usou a mediana 3, congelada antes do novo ajuste integral, sem conjunto de validação. A mistura selecionada tem referência OOF **0,9517074903582017**, atribuída à R22 exploratória; o refit R23 registra `roc_auc=null`, `fold_scores=[]` e zero linhas OOF. A seleção reutilizou folds, sem novo holdout independente.

A [auditoria R23](../evidence/round23-verification.json) conferiu a recarga neural, os dois checkpoints reutilizados e a mistura de todas as 299.984 linhas de teste. O [envio 58490653](../evidence/round23-public.json) recebeu **0,95183**, posição **626/2.135**, observada em **18/11/2026 às 02h12min21s UTC**. O CSV remoto e local compartilham SHA-256 `937dc4dd355229bc49fda67c3099ea97867a23fb9c9bbe97a2bf57fae6cf24de`. Essa é uma melhora pública de 0,00021 sobre a R20, sem inferência de vitória ou posição final privada.

```mermaid
flowchart LR
    A[CSVs oficiais e hashes] --> B[CV R22 5 folds e seleção interna de E]
    B --> C[Mediana E3 congelada]
    C --> D[Nova RealMLP integral: 16 membros]
    E[Offset e LGB completos R20 auditados] --> H[Soma em probabilidades: pesos R23]
    D --> H
    F[GPT-2 público v2 com autoria e hashes] --> H
    H --> I[Validação e recálculo do CSV inteiro]
    I --> J[Submissão e snapshot da plataforma]
```

O [guia de reprodução](reproduce.md) distingue uma nova CV pela CLI do replay auditado R23. Este último exige os artefatos exatos dos planos, incluindo recibos locais fora do Git. Não é um comando autossuficiente para um clone vazio e não permite que uma nova reprodução herde hashes ou scores de outra execução.

## Baseline histórico: seleção R18 e refit R20

O [plano R18](../evidence/round18-plan.json) fixa CV estratificada com cinco folds, shuffle e seed 42 sobre as 699.960 linhas. Cada treino próprio usa quatro folds; a OOF é calculada somente no fold excluído. No teste, cada família própria usa a média das probabilidades dos cinco modelos. Target encoding e pooling têm cinco partições internas; mapas, frequência e escalas são ajustados no treino externo. Fonte: commit [8a1c7fd](https://github.com/aikolopes-DS/fraudlab/tree/8a1c7fd141b43e41b0d43e3a07b0c5ad0bf41ae5).

| Componente | Implementação e parâmetros usados | Peso efetivo |
| --- | --- | --- |
| Offset próprio | GLM Newton-Cholesky, C=0,3, target encoding 5/50, one-hot e escala; `joint_bpe`, frequência por fold e 16 taxas hierárquicas. Resíduo: 1.000 árvores, 4 folhas, taxa 0,03, L2=5, mínimo de 100 exemplos por folha | 56,25% |
| LightGBM próprio | `value_digits`, 1.438 árvores fixas, 4 folhas, taxa 0,04, L2=5, mínimo de 100 exemplos por folha, suavização 20; sem early stopping | 18,75% |
| GPT-2 target-first público | Previsões da versão 2, importadas e convertidas de logits para probabilidades; treinamento do autor externo | 25% |

`bpe_digits` estende `value_digits` — entradas originais, logs, ciclos, razão e chaves de valores/dígitos — com seis chaves do tokenizer GPT-2 para valor e saldo. `joint_bpe` acrescenta oito interações fixas. O pooling compartilha taxas entre campos relacionados com suavizações 50/2.000/200; cada linha recebe estatísticas de outros subconjuntos internos. São 92 atributos antes do pré-processamento e 223 entradas codificadas. Testamos padrões sintéticos, sem afirmar regras causais bancárias. ID e alvo ficam excluídos.

A composição é `0.5625 × offset + 0.1875 × LGB + 0.25 × GPT2`, em probabilidades. Dentro do offset: `sigmoid(margem GLM + margem residual)`. Os dois termos usam a mesma matriz de treino com codificação supervisionada internamente cruzada e são salvos juntos. A seleção R18 não usa early stopping externo. No [refit R20](../evidence/round20-plan.json), ambos os componentes próprios são reaprendidos com todos os 699.960 rótulos, preservando configurações e pesos da mistura. O GPT-2 externo permanece igual. Esses refits não produzem uma nova OOF; a nota local continua atribuída à seleção R18.

A [OOF R18](../evidence/round18-results.json) é **0,9516524825255244**, ganho de 0,0001603443 sobre R10 e melhora nos cinco folds. Planos, pesos e critério precederam o treino; a [auditoria independente](../evidence/round18-verification.json) recalculou OOF, misturas e checkpoints. A CV participou da seleção e não é avaliação independente. A [submissão 58481744](../evidence/round18-public.json) obteve score público **0,95141**, posição **714/2.125**, observada em **17/11/2026 às 20h53min43s UTC**. A R10 anterior havia obtido 0,95134.

O arquivo histórico [R20, submissão 58483010](../evidence/round20-public.json), recebeu **0,95162**, posição **657/2.126** em **17/11/2026 às 21h59min16s UTC**. A [auditoria R20](../evidence/round20-verification.json) reproduziu 1.024 previsões de cada checkpoint e verificou a composição nas 299.984 linhas de teste; o CSV remoto completo tem o mesmo SHA-256 do local. A classificação continua provisória e as primeiras posições não foram alcançadas.

```mermaid
flowchart LR
    A[CSVs oficiais] --> B[Contratos e hashes]
    B --> C[CV própria 5/42 e transformações por fold]
    C --> D[GLM com BPE, pooling e interações; resíduo e LGB fixos]
    D --> E[OOF e médias de teste próprias]
    F[GPT-2 público v2] --> G[Manifesto, arrays, ordem e folds]
    B --> G
    G --> H[Sigmoid dos logits públicos]
    E --> I[Mistura em probabilidades congelada]
    H --> I
    I --> J[CSV validado e submissão explícita]
    J --> K[Score e posição observados]
    I --> L[Métricas exploratórias e proveniência]
    K --> M[Evidência pública]
    L --> M
    M --> N[Apresentação narrada, site e GitHub]
```

## Origem externa e limites de atribuição

As [previsões GPT-2 v2 de Masaya Kawamata](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof/versions/2) são CC BY 4.0. O [código de treino MIT](https://github.com/Matapanino/s6e11-gpt2-target-first/tree/8446fb73c11583df4f7006aa08fa34472249b318) e os créditos a Chris Deotte, Borisov et al./GReaT e uditjain13 são preservados. A [auditoria pública](../evidence/external-gpt2.json) e a receita registram versão, hashes e proveniência.

O importador não executa código externo: valida a release aprovada, bytes dos CSVs/arrays, formas, tipos, valores finitos, ordem original e fold_ids exatos de cinco folds com seed 42. O manifesto declara 10 ordens de atributos, log-sum-exp por rótulo e prior do treino do fold. Cada OOF pública vem do modelo que excluiu aquele fold; no teste, o autor faz média de cinco logits, seguida de sigmoid na importação. O blend posterior usa probabilidades.

`external_components` registra autoria, licenças, URLs, commit, versão 2, conversão, peso efetivo e `trained_by_us=false`. A rede pública não é apresentada como treinamento nosso. A RealMLP atual foi treinada por nós e integra a R23; MLPs anteriores e TabM próprias foram treinadas e comparadas, sem justificar sua inclusão na composição selecionada.

## Histórico separado: holdout e refits próprios

A primeira etapa reservou 15% dos rótulos com seed 2026 e usou três folds nos 85% de desenvolvimento. Depois de congelar a composição própria 75/25, a avaliação única da reserva usou a média dos três checkpoints de cada família. Esse resultado pertence ao protocolo de desenvolvimento histórico.

Em seguida, os refits únicos `final-glm` e `final-lgbm` utilizaram todos os 699.960 rótulos, sem nova OOF ou early stopping. A mediana das árvores escolhidas no desenvolvimento fixou 1.438 árvores. A própria `final-blend`, ligada à submissão 58437018, registra `phase=full_data_refit`, `roc_auc=null` e a referência de desenvolvimento separada em `selection`.

A híbrida v0.1, submissão 58438450, usava `explore5-glm` e `explore5-lgbm` mais GPT-2 v2, com OOF 0,95125863. Sua [receita histórica congelada](https://github.com/aikolopes-DS/fraudlab/blob/1418f08/notebooks/hybrid-recipe.json) permanece no histórico Git. A R10 substituiu seu GLM pelo offset com BPE e frequência, mantendo o LightGBM, o componente externo e os pesos efetivos. A receita do baseline do notebook preserva a seleção R18 e os refits R20; a receita selecionada atual é a R23 de quatro componentes.

Todos os rótulos, inclusive a antiga reserva, participam da CV atual: `phase=exploratory_full_data_cv`, `holdout_fraction=0` e `independent_holdout=false`. O holdout anterior não é reavaliado nem atribuído às rodadas atuais. R12–R17 e R19 falharam em seus critérios de promoção. Após falta de memória, a recuperação R17 preservou três checkpoints completos e treinou apenas os dois folds restantes, sem mudar o modelo; o tempo original completo permanece indisponível.

## Estrutura e rastreabilidade

| Local | Responsabilidade |
| --- | --- |
| `src/fraudlab/data.py`, `splits.py`, `features.py`, `bpe.py` | Contratos, hashes, divisões e transformações determinísticas |
| `models.py`, `experiment.py` | Modelos, ajustes por fold, OOF, checkpoints, médias e metadados |
| `offset.py`, `frequency.py` | Componente R10: árvores residuais sobre o GLM e contagens por categoria ajustadas no treino |
| `tabm.py` e MLP em `models.py` | Redes próprias treinadas como comparações |
| `realmlp.py`, `scripts/refit_neural_cv.py` | Rede própria R22 e ajuste integral R23; transformadores/checkpoints, versões, épocas e contratos auditados |
| `external.py` | Importação da única release pública aprovada e atribuição |
| `holdout.py`, `refit.py` | Avaliação reservada e refits do protocolo próprio histórico |
| `notebooks/hybrid-recipe.json`, `scripts/kaggle_notebook.py` | Baseline R18/R20 com 22 módulos e vocabulários; apêndice R23 isolado com 23 módulos, dois scripts, dois planos e hashes |
| `evidence/`, `presentation/`, `site/` | Fatos públicos, conteúdo didático, mídia e estudo de caso |

O núcleo é uma CLI Python sobre arquivos locais. Não há banco de dados ou servidor de treinamento. `evidence/public.json` concentra os fatos publicados; o README mantém o acompanhamento único do trabalho. Dados, credenciais e artefatos privados ficam fora do Git.

Os planos e resultados das ablações ficam em `evidence/`. Splines, pares, one-hot esparso e TabM não foram promovidos. Frequência e BPE foram integrados à R10 com o offset após a comparação prospectiva. A R9 teve [envio exploratório em exceção declarada](../evidence/public-evaluation-results.json): melhorou apenas três folds e não passou no critério original. A R10 passou no seu critério sem essa exceção. A [R11](../evidence/round11-results.json), com 2.000 árvores residuais, ficou abaixo da R10 na OOF e não foi submetida.

O [notebook público](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) mantém o baseline R18/R20. A [prova V6](../evidence/notebook-v6.json) comprova suas 39 células e 22 módulos salvos sem execução. O gerador V7 preserva esse fluxo e acrescenta o código atual em `r23-source/`, com módulos de `11b139f` e scripts de `c95ff46`. O apêndice só grava e verifica fontes, sem importar a nova rede ou executar treino/inferência. Gerar a V7 local não comprova publicação; save/readback da versão publicada precisa de recibo próprio. Os testes de exportação e preflight não validam o runtime numérico do Kaggle. A fonte atual está no [GitHub](https://github.com/aikolopes-DS/fraudlab). A [mídia v0.3, 139 slides e 54min51s](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0), documenta o marco R10. O estado da nova mídia e sua publicação deve ser conferido em `evidence/media.json` e `evidence/publication.json`.

## Critérios de verificação

- Dados e arquivo: alvo binário; IDs únicos, completos e na ordem exigida; probabilidades finitas entre 0 e 1.
- Modelagem: exclusão do fold avaliado, transformações aprendidas só no treino, alinhamento exato entre OOFs, parâmetros/versões/hashes registrados e escopo da validação explícito.
- Submissão: arquivo exato identificado por SHA256, processamento e posição conferidos na plataforma. Ranking público é provisório.
- Apresentação: fórmulas e símbolos legíveis, pares complexo/simples imediatos, código real, autoria externa explícita e narração/legendas sincronizadas.
- Publicação: arquivos públicos permitidos, site responsivo, links reais e verificação do ambiente hospedado separada do build local.

O comando de testes é `python -m pytest tests presentation site/test_build.py -q`. Testes sintéticos e checagens automáticas validam software; não comprovam competitividade, publicação ou que uma pessoa assistiu ao vídeo inteiro.
