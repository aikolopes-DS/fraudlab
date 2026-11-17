# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Vídeo e downloads](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.1.0)

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026. Sete submissões foram processadas. A composição R10 foi selecionada; a variante R11 foi rejeitada por ficar abaixo dela na OOF. A [variante de inferência dos refits completos](evidence/refit-hybrid-plan.json) foi submetida e obteve 0,95107, abaixo da R10. Código e release estão em [aikolopes-DS/fraudlab](https://github.com/aikolopes-DS/fraudlab). O vídeo v0.1 é um registro histórico de 16/11; a v0.2 está em preparação. Site V4 e notebook V4 ainda têm links antigos e aguardam atualização.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | TabM treinada e descartada. R10: OOF 0,95149214, melhora em 5/5 folds e critério cumprido. R11: OOF 0,95145947; rejeitada por ficar abaixo da R10. CV exploratória com todos os rótulos, sem holdout independente | [Resultado R10](evidence/round10-results.json), [verificação individual R10](evidence/round10-verification.json), [plano R11](evidence/round11-plan.json), [resultado R11](evidence/round11-results.json) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58470267`: score público 0,95134; posição 712/2.089 observada em 17/11/2026 às 13h34min51s UTC. Primeiros lugares ainda não alcançados | [Resultado público R10](evidence/round10-public.json), [evidência atual](evidence/public.json), [arquivo remoto reconferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | Vídeo v0.1 histórico de 16/11: 116 slides e 44min27s; decodificação AV integral, ASR integral e inspeção dos 116 quadros passaram. Sem alegação de escuta humana ou reprodução contínua integral. Vídeo v0.2 em preparação | [Apresentação](presentation/), [prova audiovisual v0.1](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | Site V4 e notebook V4 públicos, ainda com links antigos. Vídeo/ZIP v0.1 preservados; atualização do site e notebook para esta continuação ainda pendente | [Publicação verificada](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [downloads v0.1](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.1.0) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | 144 testes e 52 subtestes locais passaram; a CI da fonte TabM 568d62d passou com 132 testes e 52 subtestes. R10 e R11: cinco checkpoints individuais recarregados em 1.024 validações reais por fold, com erro máximo de 1,11e-16. Vídeo e ZIP v0.1 espelhados foram baixados anonimamente | [Testes](tests/), [verificação R10](evidence/round10-verification.json), [verificação R11](evidence/round11-results.json), [automação verificada](evidence/automation.json), [CI aprovada](https://github.com/aikolopes-DS/fraudlab/actions/runs/44084542166) |

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
