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

## Hipótese neural TabM — rodada 8

O [plano prospectivo R8](../evidence/round8-plan.json) propõe uma única
configuração de [TabM, ICLR 2025](https://arxiv.org/html/2410.24210v3), com
implementação oficial como dependência e treino próprio. A hipótese é que seus
erros complementem a composição atual; nenhum ganho competitivo foi medido
nesta preparação. Atribuímos a arquitetura aos autores, não ao FraudLab.

O modelo reúne 16 MLPs implícitas de dois blocos de 256 unidades. Em uma camada,
o membro `i` calcula `s_i ⊙ (W (r_i ⊙ x_i)) + b_i`: a matriz grande `W` é
compartilhada, enquanto vetores próprios alteram a visão de cada membro. Antes
dos blocos, embeddings lineares por partes aprendem uma representação de 12
dimensões por entrada, usando até 48 intervalos calculados no treino. Usamos
`value_digits` e o `NeuralPreprocessor` existente, incluindo target encoding
internamente cross-fitted em cinco partições.

Em linguagem cotidiana, 16 analistas usam o mesmo manual e cada um realça
detalhes de forma própria. Cada analista recebe sua avaliação durante o estudo;
na decisão final, fazemos a média de suas probabilidades. O treino minimiza
`sum(BCEWithLogits(z_bi, y_b)) / (B × 16)`. A previsão é
`mean(sigmoid(z_bi))`, como orienta a
[implementação oficial](https://github.com/yandex-research/tabm).

Dentro de cada fold externo, 15% do treino escolhem a época `E`. Imputação,
escalas, target encoding, máscara de constantes e intervalos são ajustados
somente nos outros 85%. Depois tudo é descartado: outro pré-processador e
outro modelo são ajustados no treino externo completo por exatamente `E`
épocas. O fold externo só recebe previsões. A mistura pré-declarada usa 10%
TabM e 90% da composição `explore5-public25`, mantendo o mesmo critério de
ganho superior a 0,0001 e melhora em pelo menos quatro folds. Continua sendo
CV reutilizada para seleção, sem reserva independente.

## Novas hipóteses após a R10 — 17/11/2026

A referência desta etapa é a R10: OOF 0,95149214 e score público 0,95134.
A R11, com mais árvores residuais, ficou abaixo dela e foi descartada.
Os planos [R12](../evidence/round12-plan.json), [R13](../evidence/round13-plan.json)
e [R14](../evidence/round14-plan.json) foram publicados antes dos respectivos
treinos. Pesos, folds e critérios são fixados no plano; todas as comparações
serão registradas, inclusive as negativas. A CV continua exploratória.

A R12 testa CatBoost com categorias de valores, dígitos e BPE, usando GPU e
6.000 árvores fixas. O smoke sintético encontrou e permitiu corrigir o registro
de `best_iteration=None` quando não há conjunto de avaliação. Esse smoke
comprova a execução pequena; não mede qualidade nem recursos do treino completo.

A R13 testa 16 taxas hierárquicas de decimais e tokens. Para um token `t`,
`p_pool=(sum_y+50*p_prior)/(count+50)` agrega campos relacionados;
`p_field=(sum_y_field+2000*p_pool)/(count_field+2000)` adapta a taxa ao campo;
a combinação do último token inteiro com a fração usa
`p_pair=(sum_y_pair+200*p_field)/(count_pair+200)`. Cada linha de treino recebe
estatísticas de outros quatro quintos do treino interno. O prior também vem
somente desse subconjunto. Os mapas completos do treino externo são usados
apenas na inferência. Em termos cotidianos, uma loja nova começa com a taxa da
rede, ajusta a estimativa com seus próprios clientes e depois aprende grupos
mais específicos quando reúne exemplos suficientes.

A ideia vem do pooling de megayak, descrito no
[Part 12 de goodpjw2008, versão 16](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298).
A implementação é própria. O script externo foi lido, sem execução. Seu
resultado declarado usa seleção global de pares, contagens de treino+teste e
early stopping externo; nossos planos excluem esses passos e não tratam aquele
score como validação do FraudLab. A R14 amplia separadamente a representação
para tokens do número completo nas seis colunas contínuas, com 42 novas chaves.

O [original público foi auditado](../evidence/original-data-audit.json): 20 mil
linhas, 339 fraudes e nenhuma sobreposição exata das 24 features entre original,
treino e teste. O gerador não foi verificado; não há mapeamento de IDs que
justifique transferir rótulos. Nenhum modelo adicional foi treinado no original
nesta auditoria. Essa fonte já faz parte da adaptação declarada pelo GPT-2 de
Masaya usado na composição atual.

A [R13 terminou](../evidence/round13-results.json) com OOF própria 0,95093272,
acima dos 0,95077054 da R10. As misturas com 25%, 50% e 100% de substituição do
offset chegaram a 0,95152100, 0,95154286 e 0,95156442. As duas primeiras melhoraram
os cinco folds, mas não superaram o ganho mínimo; a última melhorou três folds.
Nenhuma foi submetida. A [auditoria independente](../evidence/round13-verification.json)
recalculou as métricas, fórmulas em todas as linhas, hashes e cinco recargas reais.

A [R15](../evidence/round15-results.json) manteve os três modelos e pesos da R10.
As médias por logits, ranks normalizados e logits desses ranks chegaram a
0,95150283, 0,95150269 e 0,95151381; nenhuma passou no gate. Para ranks, o
round-trip CSV alterou uma AUC em 1,864e-10 ao desfazer empates numéricos. Isso
foi quantificado no relatório, sem mudar pesos, clipping ou regra de combinação.
Coeficientes nessas escalas não são percentuais de contribuição em probabilidade.

Os planos [R16](../evidence/round16-plan.json) e [R17](../evidence/round17-plan.json)
separam duas hipóteses seguintes: mais rótulos públicos originais somente no
treino de cada fold, e uma rede que aprende uma correção da margem do GLM.
A segunda seleciona épocas em uma divisão interna, descarta todos os modelos
dessa seleção e refaz o treino externo completo. Os planos não afirmam ganho.
