# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 16/11/2026. O projeto está em experimentação. A primeira submissão foi processada, mas o resultado inicial ainda está distante das primeiras posições.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | Logistic e CatBoost treinados; rede neural e otimização em andamento | [Núcleo](src/fraudlab), perfis e métricas locais; holdout de 15% reservado |
| Concorrer de verdade | Submissão processada, score e posição com data | Referência inicial: score 0,94240; posição pública 1.837/1.996 em 16/11/2026 às 14h49 UTC | [evidence/public.json](evidence/public.json), submissão `58434290` |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | Motor e amostra de voz neural testados; vídeo completo pendente | [Apresentação](presentation/), testes e amostras locais identificadas |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | GitHub publicado; site local testado, deploy pendente; notebook Kaggle recusado com HTTP 403, em investigação | [GitHub](https://github.com/aikolopes-og/fraudlab), `site/` |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | Suíte integrada local passou, incluindo treino neural sintético; revisão cruzada realizada; CI remota pendente | [Testes](tests/), [CI](.github/workflows/check.yml) |

## Princípios

- Poucas peças, responsabilidades claras e dependências justificadas.
- Dados, credenciais e checkpoints não entram no Git. Código público sob MIT; a licença dos dados é independente.
- Pré-processamento ajustado apenas no treino de cada divisão. IDs não viram preditores por padrão.
- Arquivo submetido e experimento ligados por hashes, configuração e registro de métricas.
- Sem inventar resultados: validação local, teste simulado, score público e classificação final têm estados próprios.
- Nenhuma duração fixa para o vídeo. Todo conceito complexo recebe uma explicação simples com exemplo cotidiano no slide imediatamente seguinte.

## Estrutura

```text
src/fraudlab/       inspeção, treinamento, avaliação e submissão
tests/             contratos dos dados e da validação
presentation/      slides, roteiro, narração e exportação
site/              estudo de caso estático
evidence/          fatos públicos verificados, sem dados brutos
docs/              requisitos, arquitetura e operação
data/raw/          CSVs do Kaggle, ignorados pelo Git
artifacts/         experimentos e mídia gerada, ignorados pelo Git
```

Os comandos de instalação, treino, submissão, apresentação e site estão em [Executar o projeto](docs/reproduce.md). Requisitos e decisões estão em [Arquitetura](docs/design.md).
