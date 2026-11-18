# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Baixar a AED completa](https://github.com/aikolopes-DS/fraudlab/releases/download/v0.3.0/AED-FraudLab-entrega.zip) · [Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Release e hashes](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0)

Extraia o ZIP e abra `ENTREGA.html`: vídeo, slides, legendas, código, instruções e comprovantes estão juntos. Os materiais foram publicados e verificados; não houve envio ao portal da faculdade.

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026. O pacote v0.3 e o site V6 documentam o marco R10; o notebook V6 já contém a receita R18/R20. A melhor submissão atual é o refit R20: nono envio processado, score público 0,95162 e posição 657/2.126 no snapshot das 21h59 UTC. A apresentação v0.4 está em atualização para pooling, interações e novos experimentos; ainda não foi renderizada nem publicada. Primeiras posições não alcançadas; o ranking final depende da avaliação privada ao encerrar a competição.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | R18 selecionada: OOF 0,95165248, ganho 0,00016034 sobre R10 e melhora em 5/5 folds. R12–R17 e R19 concluídas e rejeitadas pelos critérios respectivos. R17 recuperada após falta de memória, preservando três checkpoints e treinando apenas os dois folds restantes. R20 reestimou a receita R18 em todos os dados, sem nova OOF; score público 0,95162. R21 passou no gate local por margem pequena, mas recebeu 0,95138 no Kaggle e não foi promovida. R22 RealMLP em treino competitivo: três dos cinco folds salvos; OOF integral e promoção ainda pendentes. CV exploratória, sem holdout independente | [R18](evidence/round18-results.json), [auditoria R18](evidence/round18-verification.json), [R12](evidence/round12-results.json), [R14](evidence/round14-results.json), [R16](evidence/round16-results.json), [R17](evidence/round17-verification.json), [R19](evidence/round19-verification.json) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58483010`: score público 0,95162; posição 657/2.126 em 17/11/2026 às 21h59 UTC. CSV remoto completo baixado e SHA idêntico ao local. Dez submissões processadas; competição aberta. Primeiros lugares ainda não alcançados | [Resultado público R20](evidence/round20-public.json), [auditoria do refit](evidence/round20-verification.json), [evidência atual](evidence/public.json), [arquivo remoto conferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | v0.3 histórica: 54min51s, 139 slides, 55 pares técnicos/analogia, 45 fórmulas e 738 legendas. AV/ASR integrais e 139 quadros conferidos, sem declarar escuta humana integral. v0.4 em montagem para a solução atual; ainda sem nova mídia ou validação audiovisual | [Cobertura v0.3](evidence/aed-content.json), [fontes em atualização](presentation/), [prova audiovisual v0.3](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | GitHub público, site V6, notebook V6 e pacote v0.3 publicados. Quatro assets baixados anonimamente; 307 arquivos do pacote conferidos por hash. ZIP extraído com vídeo/slides funcionando localmente em desktop/mobile. Site: download, player, 738 legendas e saltos para nove trechos em desktop e ranking em celular passaram. Notebook V6 salvo sem execução na nuvem; 39 células remotas conferidas | [Prova V6](evidence/notebook-v6.json), [Entrega validada](evidence/delivery.json), [publicação verificada](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [notebook](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | CI da fonte cab2650: 351 testes e 72 subtestes passaram; 80 arquivos de código conferem byte a byte com a fonte testada. A revisão local de apresentação e site passou em 82 testes e 72 subtestes. Rodadas recentes têm checkpoints reais e auditorias de fórmulas, dados e configurações; R17 recuperada tem cinco recargas com erro zero. Otimização de strings preservou sete famílias em 2.048 linhas reais e previsões sintéticas exatas. v0.3 conserva sua própria fonte e prova anterior | [Testes](tests/), [auditoria R18](evidence/round18-verification.json), [R17](evidence/round17-verification.json), [equivalência](evidence/string-sharing-verification.json), [Prova de software](evidence/software-v04.json), [CI cab2650](https://github.com/aikolopes-DS/fraudlab/actions/runs/44198322482) |

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
