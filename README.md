# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 16/11/2026. O projeto está em implementação; ainda não há resultado competitivo comprovado.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Prazo e métrica confirmados; regras e sessão pendentes | [Overview oficial](https://www.kaggle.com/competitions/playground-series-s6e11/overview) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | Em implementação; nenhum treino real | `src/fraudlab/`, `tests/` quando integrados |
| Concorrer de verdade | Submissão processada, score e posição com data | Não realizado | [evidence/public.json](evidence/public.json), valores nulos |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | Motor em implementação; nenhum vídeo final | `presentation/` quando integrado |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | Conta GitHub confirmada; publicação pendente | `aikolopes-og` via GitHub CLI |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | Pendente de integração | CI e testes serão vinculados após execução |

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

Os comandos de instalação, treino e publicação serão documentados depois de verificados no projeto integrado.
