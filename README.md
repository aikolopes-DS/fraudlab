# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Baixar a AED completa](https://github.com/aikolopes-DS/fraudlab/releases/download/v0.3.0/AED-FraudLab-entrega.zip) · [Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Release e hashes](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0)

Extraia o ZIP e abra `ENTREGA.html`: vídeo, slides, legendas, código, instruções e comprovantes estão juntos. Os materiais foram publicados e verificados; não houve envio ao portal da faculdade.

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026. Os materiais da AED estão montados, validados e publicados no pacote único v0.3, com site V6 e notebook V5. Sete submissões foram processadas; a R10 elevou o score público de 0,95097 para 0,95134. A auditoria final corrigiu a documentação e completou a apresentação com proposta inicial, one-hot e operação interna da TabM. As cinco frentes têm evidências abaixo. Primeiras posições ainda não alcançadas; o ranking final depende da avaliação privada ao encerrar a competição.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | TabM treinada e descartada. R10: OOF 0,95149214, melhora em 5/5 folds e critério cumprido. R11: OOF 0,95145947; rejeitada. R12: CatBoost na GPU em treino; sem resultado. R13/R14: pooling e tokens completos planejados, aguardando testes/revisão. CV exploratória com todos os rótulos, sem holdout independente | [Resultado R10](evidence/round10-results.json), [verificação R10](evidence/round10-verification.json), [resultado R11](evidence/round11-results.json), [planos R12](evidence/round12-plan.json), [R13](evidence/round13-plan.json), [R14](evidence/round14-plan.json) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58470267`: score público 0,95134; posição 723/2.112 confirmada em 17/11/2026 às 18h24 UTC. Arquivo remoto reconferido; competição aberta. Primeiros lugares ainda não alcançados | [Auditoria atual dos serviços](evidence/aed-services.json), [evidência atual](evidence/public.json), [arquivo remoto reconferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | Vídeo v0.3 de 54min51s, 139 slides, 55 pares técnicos/analogia, 45 fórmulas e 738 legendas. Decodificação AV integral, ASR integral e inspeção dos 139 quadros passaram. Não se declara escuta humana nem reprodução contínua integral; ASR pode distorcer termos e números | [Auditoria de cobertura](evidence/aed-content.json), [apresentação](presentation/), [prova audiovisual](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | GitHub público, site V6, notebook V5 e pacote v0.3 publicados. Quatro assets baixados anonimamente; 307 arquivos do pacote conferidos por hash. ZIP extraído com vídeo/slides funcionando localmente em desktop/mobile. Site: download, player, 738 legendas e saltos para nove trechos em desktop e ranking em celular passaram. Notebook salvo sem execução na nuvem | [Entrega validada](evidence/delivery.json), [publicação verificada](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [notebook](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) |
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
