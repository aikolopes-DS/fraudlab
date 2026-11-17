# Requisitos e arquitetura

## Objetivo e contratos

Participar de uma competição aberta com intenção de vencer, construir uma solução explicável e publicar código, vídeo e estudo de caso como portfólio. A atividade de Redes Neurais é parte desse objetivo. A liderança continua uma meta não alcançada.

O projeto exige regras conferidas, treinamento neural próprio real, comparação de alternativas, previsões submetidas de fato e posição observada. Scores locais, score público e ranking final são registros distintos. Nenhuma flag de evidência é inferida de um build, teste sintético ou arquivo gerado.

A apresentação cobre problema, proposta, construção, matemática, arquiteturas, experimentos, códigos e ranking. Cada slide complexo deve ser seguido imediatamente por explicação simples com exemplo cotidiano. Narração e legendas são em português, com duração determinada pela cobertura.

## Protocolo operacional atual: R10

O [plano R10](../evidence/round10-plan.json) fixa CV estratificada com cinco folds, shuffle e seed 42 sobre as 699.960 linhas. Cada treino próprio usa quatro folds; a OOF é calculada somente no fold excluído. No teste, cada família própria usa a média das probabilidades dos cinco modelos. O pré-processamento aprendido fica dentro de cada treino; target encoding usa cinco partições internas. A fonte do treino é o commit [169f3ee](https://github.com/aikolopes-DS/fraudlab/tree/169f3ee5cdfbb2afe16c942a7af0efb8b12a403f).

| Componente | Implementação e parâmetros usados | Peso efetivo |
| --- | --- | --- |
| Offset próprio | GLM Newton-Cholesky, C=0,3, target encoding 5/50, one-hot e escala; `bpe_digits` e frequência por fold. Resíduo: 1.000 árvores, 4 folhas, taxa 0,03, L2=5, mínimo de 100 exemplos por folha | 56,25% |
| LightGBM próprio | `value_digits`, 1.438 árvores fixas, 4 folhas, taxa 0,04, L2=5, mínimo de 100 exemplos por folha, suavização 20; sem early stopping | 18,75% |
| GPT-2 target-first público | Previsões da versão 2, importadas e convertidas de logits para probabilidades; treinamento do autor externo | 25% |

`bpe_digits` estende `value_digits` — entradas originais, logs, ciclos, razão e chaves de valores/dígitos — com seis chaves do tokenizer GPT-2 para valor e saldo. A frequência usa contagens aprendidas somente no treino de cada fold. Essas representações não treinam um GPT-2 próprio e são hipóteses sobre os dados sintéticos, não regras causais de fraude bancária. ID e alvo são excluídos das entradas.

A composição é `0.5625 × offset + 0.1875 × LGB + 0.25 × GPT2`, em probabilidades. Dentro do offset, a previsão é `sigmoid(margem GLM + margem das árvores residuais)`. O GLM e o resíduo usam a mesma matriz de treino com target encoding internamente cruzado; os dois termos ficam no checkpoint. A R10 não usa early stopping externo nem refit único com 100% dos rótulos.

A [OOF R10](../evidence/round10-results.json) é **0,9514921382152453**, com melhora nos cinco folds sobre a baseline `explore5-public25` (0,9512586315) e acima da R9. Os parâmetros e pesos foram declarados antes do treino e do envio. Essa OOF participou da seleção e não é avaliação independente. A [submissão 58470267](../evidence/round10-public.json) obteve score público **0,95134**; a posição **712/2.089** foi observada em **17/11/2026 às 13h34min51s UTC** e permanece um snapshot provisório.

```mermaid
flowchart LR
    A[CSVs oficiais] --> B[Contratos e hashes]
    B --> C[CV própria 5/42 e transformações por fold]
    C --> D[GLM com BPE e frequência, resíduo e LGB fixos]
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

`external_components` registra autoria, licenças, URLs, commit, versão 2, conversão, peso efetivo e `trained_by_us=false`. A rede pública não é apresentada como treinamento nosso. MLPs e TabM próprias foram treinadas e comparadas; não justificaram sua inclusão na composição selecionada e permanecem experimentos reais documentados.

## Histórico separado: holdout e refits próprios

A primeira etapa reservou 15% dos rótulos com seed 2026 e usou três folds nos 85% de desenvolvimento. Depois de congelar a composição própria 75/25, a avaliação única da reserva usou a média dos três checkpoints de cada família. Esse resultado pertence ao protocolo de desenvolvimento histórico.

Em seguida, os refits únicos `final-glm` e `final-lgbm` utilizaram todos os 699.960 rótulos, sem nova OOF ou early stopping. A mediana das árvores escolhidas no desenvolvimento fixou 1.438 árvores. A própria `final-blend`, ligada à submissão 58437018, registra `phase=full_data_refit`, `roc_auc=null` e a referência de desenvolvimento separada em `selection`.

A híbrida v0.1, submissão 58438450, usava `explore5-glm` e `explore5-lgbm` mais GPT-2 v2, com OOF 0,95125863. Sua [receita](../notebooks/hybrid-recipe.json) permanece como baseline histórica. A R10 substitui seu GLM pelo offset com BPE e frequência, mantendo o LightGBM, o componente externo e os pesos efetivos. Ambas usam médias de cinco modelos próprios, sem os refits únicos anteriores.

Todos os rótulos, inclusive a antiga reserva, participam da CV atual: `phase=exploratory_full_data_cv`, `holdout_fraction=0` e `independent_holdout=false`. O holdout anterior não é reavaliado nem atribuído à R10. Seu relatório, ledger e artefatos foram preservados.

## Estrutura e rastreabilidade

| Local | Responsabilidade |
| --- | --- |
| `src/fraudlab/data.py`, `splits.py`, `features.py`, `bpe.py` | Contratos, hashes, divisões e transformações determinísticas |
| `models.py`, `experiment.py` | Modelos, ajustes por fold, OOF, checkpoints, médias e metadados |
| `offset.py`, `frequency.py` | Componente R10: árvores residuais sobre o GLM e contagens por categoria ajustadas no treino |
| `tabm.py` e MLP em `models.py` | Redes próprias treinadas como comparações |
| `external.py` | Importação da única release pública aprovada e atribuição |
| `holdout.py`, `refit.py` | Avaliação reservada e refits do protocolo próprio histórico |
| `notebooks/hybrid-recipe.json`, `scripts/kaggle_notebook.py` | Receita e exportação da baseline v0.1; o notebook V5 contextualiza a R10 em Markdown |
| `evidence/`, `presentation/`, `site/` | Fatos públicos, conteúdo didático, mídia e estudo de caso |

O núcleo é uma CLI Python sobre arquivos locais. Não há banco de dados ou servidor de treinamento. `evidence/public.json` concentra os fatos publicados; o README mantém o acompanhamento único do trabalho. Dados, credenciais e artefatos privados ficam fora do Git.

Os planos e resultados das ablações ficam em `evidence/`. Splines, pares, one-hot esparso e TabM não foram promovidos. Frequência e BPE foram integrados à R10 com o offset após a comparação prospectiva. A R9 teve [envio exploratório em exceção declarada](../evidence/public-evaluation-results.json): melhorou apenas três folds e não passou no critério original. A R10 passou no seu critério sem essa exceção. A [R11](../evidence/round11-results.json), com 2.000 árvores residuais, ficou abaixo da R10 na OOF e não foi submetida.

O [notebook V5](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) mantém as células executáveis da baseline v0.1, exportadas do SHA `30dc7eb723e4cf31291bb075ed6c0d943557264c`, e acrescenta Markdown sobre a solução atual. Foi salvo **sem execução na nuvem**; as 22 células remotas conferem com as locais, sem contagens de execução ou outputs. Ele não executa a R10. A fonte da R10 está no [GitHub](https://github.com/aikolopes-DS/fraudlab) e a apresentação atual é a [v0.2, com 134 slides e 51min23s](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.2.0). A [verificação](../evidence/verification.json) separa publicação do notebook, treino local e avaliação dos CSVs pelo Kaggle.

## Critérios de verificação

- Dados e arquivo: alvo binário; IDs únicos, completos e na ordem exigida; probabilidades finitas entre 0 e 1.
- Modelagem: exclusão do fold avaliado, transformações aprendidas só no treino, alinhamento exato entre OOFs, parâmetros/versões/hashes registrados e escopo da validação explícito.
- Submissão: arquivo exato identificado por SHA256, processamento e posição conferidos na plataforma. Ranking público é provisório.
- Apresentação: fórmulas e símbolos legíveis, pares complexo/simples imediatos, código real, autoria externa explícita e narração/legendas sincronizadas.
- Publicação: arquivos públicos permitidos, site responsivo, links reais e verificação do ambiente hospedado separada do build local.

O comando de testes é `python -m pytest tests presentation site/test_build.py -q`. Testes sintéticos e checagens automáticas validam software; não comprovam competitividade, publicação ou que uma pessoa assistiu ao vídeo inteiro.
