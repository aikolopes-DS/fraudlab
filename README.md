# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Vídeo e downloads](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.2.0)

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026. Sete submissões foram processadas. A R10 elevou o score público de 0,95097 para 0,95134; a R11 foi rejeitada por ficar abaixo dela na OOF. A auditoria final corrigiu documentação e completou a apresentação: v0.3 com 139 slides, 54min51s e cobertura textual/visual verificada. A montagem do pacote único e a publicação desta versão estão em execução; v0.2/site V5/notebook V5 permanecem públicos. Primeiras posições não alcançadas; o ranking final depende da avaliação privada ao encerrar a competição.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | TabM treinada e descartada. R10: OOF 0,95149214, melhora em 5/5 folds e critério cumprido. R11: OOF 0,95145947; rejeitada por ficar abaixo da R10. CV exploratória com todos os rótulos, sem holdout independente | [Resultado R10](evidence/round10-results.json), [verificação individual R10](evidence/round10-verification.json), [plano R11](evidence/round11-plan.json), [resultado R11](evidence/round11-results.json) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58470267`: score público 0,95134; posição 723/2.112 confirmada em 17/11/2026 às 18h24 UTC. Arquivo remoto reconferido; competição aberta. Primeiros lugares ainda não alcançados | [Auditoria atual dos serviços](evidence/aed-services.json), [evidência atual](evidence/public.json), [arquivo remoto reconferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | Vídeo v0.3 de 54min51s, 139 slides, 55 pares técnicos/analogia, 45 fórmulas e 738 legendas. Decodificação AV integral, ASR integral e inspeção dos 139 quadros passaram. Não se declara escuta humana nem reprodução contínua integral; ASR pode distorcer termos e números | [Auditoria de cobertura](evidence/aed-content.json), [apresentação](presentation/), [prova audiovisual](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | GitHub novo público, site V5 e notebook V5 publicados. Player, legendas e saltos para seis slides em desktop e para o ranking em celular passaram anonimamente. Os três assets da release foram baixados e conferidos por hash. Notebook salvo sem execução na nuvem; 22 células remotas iguais às locais | [Publicação verificada](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [notebook](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | 172 testes e 53 subtestes passaram localmente e na CI da fonte 53dda3c; 48 arquivos de código conferem com os hashes testados. O empacotador recusa mídia adulterada, provas alheias e dados privados. R10/R11 têm cinco checkpoints reais recarregados, com erro máximo de 1,11e-16. Automação operacional na conta nova | [Testes](tests/), [verificação R10](evidence/round10-verification.json), [verificação R11](evidence/round11-results.json), [automação](evidence/automation.json), [CI aprovada](https://github.com/aikolopes-DS/fraudlab/actions/runs/44149374948) |

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
