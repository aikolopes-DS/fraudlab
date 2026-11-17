# Pesquisa e atribuição

Consulta realizada em 16/11/2026. Os experimentos deste repositório usam os
arquivos da competição S6E11. Resultados de outros participantes servem como
referência de pesquisa e não são resultados deste projeto.

## Ideias investigadas

Os notebooks públicos [S6E11 — Beginner → Expert](https://www.kaggle.com/code/yashteotia/s6e11-beginner-expert-lb-0-95306),
de yashteotia, e [S6E11 — 100% Self-Trained Stack](https://www.kaggle.com/code/sidhaarthshree/s6e11-100-self-trained-stack-lb-0-95154),
de sidhaarthshree, foram consultados pela API do Kaggle. A pesquisa motivou
hipóteses sobre representações de valores e dígitos, codificação supervisionada
com suavização, árvores pequenas e regressão logística sobre logits de risco.
Essas hipóteses são implementadas e avaliadas neste projeto com sua própria
validação, artefatos e ablações. Os números presentes nos títulos dos notebooks
pertencem aos respectivos autores.

As licenças desses dois notebooks não foram confirmadas. Não incorporamos seu
código, pesos ou arquivos de previsões. Atribuir uma ideia não equivale a declarar
uma licença de redistribuição de código de terceiros.

## Fonte neural externa auditada

Também foi auditada a fonte [GPT-2 target-first de Masaya Kawamata](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof):
previsões **CC BY 4.0**, [código MIT](https://github.com/Matapanino/s6e11-gpt2-target-first/tree/8446fb73c11583df4f7006aa08fa34472249b318),
commit `8446fb73c11583df4f7006aa08fa34472249b318`. O autor declara adaptação de
GPT-2 ao dataset original e treinamento em cinco folds da competição. **Esse
treinamento não foi executado pelo FraudLab.**

O artefato é a versão **2**, referência imutável de download
`masayakawamata/s6e11-gpt2-target-first-oof/2`; hashes, formas e verificações estão
no [registro público da auditoria](../evidence/external-gpt2.json).

O método target-first é creditado a [Chris Deotte](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754683);
a adaptação de linguagem a tabelas, a [Borisov et al./GReaT](https://arxiv.org/abs/2210.06280);
e o original sintético, a uditjain13. A reutilização das previsões mantém a
[licença CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) e registra a
conversão por sigmoid e a eventual combinação como operações do FraudLab.

A auditoria conferiu integridade dos arrays, hashes dos CSVs, ordem das linhas e
folds estratificados 5/42. A conversão dos log-odds por sigmoid reproduziu AUC OOF
`0.9500272037932734`. A release combina dez ordens dos campos por log-sum-exp em
cada classe; o teste usa a média dos log-odds dos cinco folds. Esses resultados
pertencem ao componente externo; o ganho de uma combinação precisa ser medido
com os modelos próprios no mesmo protocolo. A antiga reserva de 15% deixa de ser
independente para essa extensão e permanece apenas como registro histórico.

A explicação de [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2),
[atenção](https://arxiv.org/abs/1706.03762) e
[arquitetura autoregressiva](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
segue fontes primárias e o código auditado. Não inferimos números de camadas,
cabeças ou parâmetros do artefato sem uma configuração confirmada.

## Referências de implementação

- [TargetEncoder — scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.TargetEncoder.html):
  médias de alvo por categoria, suavização em direção à média do treino,
  categorias desconhecidas e codificação de treino por cross-fitting interno.
  `fit_transform(X, y)` e `fit(X, y).transform(X)` têm comportamentos distintos.
- [QuantileTransformer — scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.QuantileTransformer.html):
  estimativa da distribuição acumulada, mapeamento por quantis e transformação
  para distribuição normal, ajustados apenas no treino de cada divisão.
- [LGBMClassifier — LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html):
  boosting binário, limite de folhas por árvore, regularização e early stopping.
- [LogisticRegression — scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html):
  regularização L2, controle inverso de intensidade por `C` e solver
  Newton-Cholesky no GLM; o baseline logístico usa L-BFGS.
- [BCEWithLogitsLoss — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
  e [AdamW — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html):
  perda binária numericamente estável e atualização da rede neural.

## Origem e limite dos dados

Os metadados do [dataset original de uditjain13](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
informam geração sintética com NumPy e licença CC0-1.0. O CSV original não foi
carregado nos treinos executados pelo FraudLab; a fonte externa GPT-2 declara
adaptação nesse original. A licença do original não substitui as regras dos
arquivos da competição nem a licença das previsões externas.

Padrões de dígitos são hipóteses sobre esta distribuição sintética, e não regras
universais de fraude bancária. Um ganho local só é atribuído a uma mudança quando
as métricas das execuções comparadas sustentam essa conclusão. AUC local,
avaliação da reserva, pontuação pública e posição final são evidências distintas.

## Segunda rodada: 17/11/2026

O [plano registrado antes dos novos treinos](../evidence/round2-plan.json) limita
esta rodada a duas mudanças próprias, na mesma CV 5/42. Os resíduos OOF mostram
complementaridade entre GLM e árvores, enquanto algumas codificações por valor
contínuo têm pouco suporte por categoria. Isso motiva testar:

- **Curvas suaves no GLM:** bases cúbicas do
  [SplineTransformer](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.SplineTransformer.html),
  com 12 nós por quantis ajustados apenas no treino de cada fold. As demais
  representações do GLM são mantidas. Essa mudança testa formas não lineares
  suaves sem presumir uma distribuição geradora conhecida.
- **Árvores residuais sobre o GLM:** a margem logística inicializa árvores
  pequenas com `init_score`; na inferência, a margem do GLM é somada ao
  `raw_score` das árvores antes da sigmoid. O treino usa exatamente a matriz
  com target encoding internamente cross-fitted que ajustou o GLM. A ideia foi
  investigada nas seções 6.2–6.3 de
  [goodpjw2008 — Part 12](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298)
  e implementada aqui sem copiar código externo. O candidato próprio não
  seleciona pares usando rótulos de validação nem usa early stopping externo.

A ficha do original cita `gen.py`, mas a listagem oficial disponibiliza somente
o CSV. O [notebook do próprio autor](https://www.kaggle.com/code/uditjain13/credit-card-fraud-2026-interpretable-eda)
contém EDA, sem uma equação geradora verificável. Por isso não reconstruímos
coeficientes ou distribuições como se fossem fatos conhecidos. Nenhum CSV
externo foi acrescentado aos treinos desta rodada.

Esses candidatos foram medidos pelas OOF compatíveis. O resultado e a decisão
estão registrados abaixo; a antiga reserva continua apenas histórica.

### Resultado da continuação

As quatro configurações próprias foram treinadas nos cinco folds. Curvas e
árvores residuais, quatro pares predefinidos e frequências de categoria não
produziram uma melhoria que atendesse ao plano. A melhor combinação nova teve
OOF 0,95132140 contra 0,95125863 da composição publicada, com somente três folds
melhores. Duas combinações fixas em log-odds também ficaram abaixo do critério.
Não houve novo envio ao Kaggle. Planos, folds e decisões estão nas evidências
das [rodadas 2](../evidence/round2-results.json),
[3](../evidence/round3-results.json) e [4](../evidence/round4-results.json).

As frequências usam `log(1 + contagem no treino)`; os quatro pares são
codificados pelo mesmo target encoder internamente cross-fitted. Não houve
seleção de pares por estatísticas calculadas com os rótulos do fold avaliado.
O ganho da frequência foi medido na comparação própria, mas não bastou para
substituir a composição final pelo critério previamente registrado.

Uma segunda fonte neural [Megayak v4](https://www.kaggle.com/datasets/megayak/s6e11-target-first-lm/versions/4)
foi auditada e não incorporada. O código visível exclui folds, mas faltam os
vínculos que associam a execução aos vetores finais. Além disso, o OOF é descrito
como log-odds e o teste como uma combinação de ranks. Isso impede tratar os dois
como probabilidades equivalentes. A [auditoria](../evidence/external-megayak-audit.json)
registra evidências positivas, lacunas e hashes; não afirma vazamento pelo autor.

## Ablações e tokens numéricos

A [rodada 5](../evidence/round5-results.json) isolou frequência sem os pares e
dez partições internas de target encoding. A primeira mistura chegou a
0,95132941, com três folds melhores; a segunda a 0,95127072, com quatro.
Nenhuma superou simultaneamente o ganho mínimo de 0,0001 e a consistência em
quatro folds. Não houve submissão dessas variantes.

O [plano da rodada 6](../evidence/round6-plan.json) acrescenta seis chaves ao
GLM: os dois primeiros tokens do inteiro do valor e do saldo, além da combinação
do segundo token com os centavos. A ideia vem de
[azizdzhon](https://www.kaggle.com/code/azizdzhon/how-great-tokenizes-numbers-bpe-token-target)
e foi investigada no Shallow-Tree Stack v31. A implementação própria usa o
[tiktoken oficial da OpenAI](https://github.com/openai/tiktoken), sob MIT,
sem copiar tabelas ou previsões desses notebooks. Os IDs são categorias, com
target encoding ajustado somente no treino de cada fold. O eventual ganho é
uma hipótese sobre estes dados sintéticos; não comprova como foram gerados.

O [resultado dos tokens](../evidence/round6-results.json) foi OOF 0,95052405 no
GLM e 0,95133407 na mistura fixa. O ganho de 0,00007544, com três folds melhores,
não atendeu ao critério. As previsões não foram submetidas.

O [plano da rodada 7](../evidence/round7-plan.json) testa coeficientes por valor
com one-hot esparso em todas as categorias do GLM, mantendo seus logits de
target encoding. Categorias com menos de cinco ocorrências no treino são
agrupadas; desconhecidas recebem zero no bloco one-hot e a média de treino no
target encoder. Essa hipótese veio da seção *The joint GLM* do notebook
[100% Self-Trained Stack](https://www.kaggle.com/code/sidhaarthshree/s6e11-100-self-trained-stack-lb-0-95154),
sem copiar sua implementação. O solver L-BFGS evita armazenar a Hessiana densa
dos cerca de 51 mil coeficientes. Um ajuste que não converge é recusado antes
de salvar o checkpoint; a regularização e a regra de seleção permanecem fixas.

O primeiro ajuste [atingiu 500 iterações sem convergir](../evidence/round7-results.json),
antes de produzir checkpoint ou previsão. Não há AUC dessa execução. O
[plano 7b](../evidence/round7b-plan.json) aumenta somente o orçamento do
otimizador para 2.000 iterações. A função objetivo, `C`, tolerância, atributos,
folds e regra de seleção são os mesmos; a falha numérica não é um resultado
de desempenho competitivo.
