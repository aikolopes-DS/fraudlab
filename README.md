# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Vídeo e downloads](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.1.0)

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026. Quatro submissões foram processadas. As rodadas 5 e 6 foram medidas e descartadas pelo critério prévio. A correção R7b convergiu nos cinco folds, mas piorou a mistura e foi descartada. A rodada 8 prepara uma rede TabM própria com seleção interna de épocas e refit de cada fold. A automação e as próximas publicações passam para [aikolopes-DS/fraudlab](https://github.com/aikolopes-DS/fraudlab), conforme solicitado; o repositório anterior e a release v0.1 permanecem acessíveis. O vídeo v0.1 continua identificado como snapshot de 16/11.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | v0.1 mantida; rodadas 2–7b descartadas. R7b: OOF da mistura 0,9503082, contra 0,9512586 da referência; 0/5 folds melhores. Rede TabM preparada, sem resultado competitivo; sem holdout independente | [Resultado R7b](evidence/round7b-results.json), [plano TabM](evidence/round8-plan.json) |
| Concorrer de verdade | Submissão processada, score e posição com data | Score 0,95097; posição pública 828/2.046 consultada em 17/11/2026 às 05h19 UTC. Mesmo envio; primeiros lugares ainda não alcançados | [Evidência](evidence/public.json), submissão `58438450`; [arquivo remoto reconferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | 116 slides e vídeo de 44min27s gerados; decodificação AV integral, ASR integral e inspeção dos 116 quadros passaram. Sem alegação de escuta humana ou reprodução contínua integral | [Apresentação](presentation/), [prova audiovisual](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | Site V4 e notebook V3 públicos; vídeo/ZIP v0.1 preservados. Ranking de 17/11 e aviso de vídeo histórico verificados anonimamente em desktop/mobile; reprodução e 604 legendas reconferidas | [Publicação verificada](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [downloads](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.1.0) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | 132 testes e 52 subtestes locais passaram na integração TabM; a CI da fonte R7b 6797ff2 passou com 125 testes e 52 subtestes. Vídeo e ZIP espelhados foram baixados e verificados anonimamente. O bloqueio da conta antiga permanece, sem alterações de cobrança | [Testes](tests/), [automação verificada](evidence/automation.json), [CI aprovada](https://github.com/aikolopes-DS/fraudlab/actions/runs/44081656048) |

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
