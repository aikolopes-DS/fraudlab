# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 16/11/2026. Três submissões próprias foram processadas. O resultado ainda está distante das primeiras posições; uma combinação complementar com uma rede neural pública está em avaliação, com licença e origem explícitas.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | Redes próprias e alternativas treinadas; composição GLM/LGB refitada em todos os rótulos. Nova etapa exploratória 5/42 em andamento | [Experimentos agregados](evidence/experiments.json), [núcleo](src/fraudlab) |
| Concorrer de verdade | Submissão processada, score e posição com data | Terceiro envio: 0,95065; posição pública 881/2.003 em 16/11/2026 às 16h21 UTC. A meta das primeiras posições segue aberta | [Evidência](evidence/public.json), submissão `58437018`; [CSV remoto idêntico ao local](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | Roteiro, renderização e motor de voz neural testados; vídeo completo pendente da configuração competitiva | [Apresentação](presentation/), testes e amostras locais identificadas |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | GitHub, site V2 e notebook públicos; site verificado anonimamente. Snapshot e mídia finais ainda serão atualizados | [Site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [notebook](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | Suíte integrada local passou, incluindo treino neural sintético; revisão cruzada realizada; CI remota bloqueada por cobrança da conta GitHub, sem iniciar job | [Testes](tests/), [CI](.github/workflows/check.yml) |

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
