# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Baixar a AED completa](https://github.com/aikolopes-DS/fraudlab/releases/download/v0.3.0/AED-FraudLab-entrega.zip) · [Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Release e hashes](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0)

Extraia o ZIP e abra `ENTREGA.html`: vídeo, slides, legendas, código, instruções e comprovantes estão juntos. Os materiais foram publicados e verificados; não houve envio ao portal da faculdade.

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026, horário de São Paulo. A melhor submissão atual é a R23: score público **0,95183**, posição **626/2.135** na consulta das 23h12 SP (18/11, 02h12 UTC). São doze envios processados. A apresentação v0.4 tem preview estático revisado, com 198 slides; a narração está em geração. O pacote v0.3 conserva o marco R10. A nova landing passou pela pesquisa, implementação e crítica independente locais e está em publicação. Primeiras posições não alcançadas; o ranking final depende da avaliação privada ao encerrar a competição.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | R22: RealMLP própria com 16 membros por fold; híbrida OOF 0,95170749, melhora em 4/5 folds sobre R18. R23 reestimou a receita escolhida nos 699.960 rótulos: nova RealMLP, 3 épocas; checkpoints integrais GLM/correção e LightGBM reutilizados e GPT-2 público creditado. Pesos fixos: 42,1875% / 14,0625% / 18,75% / 25%. Recargas independentes e fórmula do CSV conferidas. A OOF citada é histórica da R22, não nova validação da R23. CV exploratória, sem holdout independente | [R22](evidence/round22-results.json), [R23](evidence/round23-results.json), [auditoria R23](evidence/round23-verification.json), [histórico de pesquisa](docs/research.md) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58490653`: score público 0,95183; posição 626/2.135 em 17/11/2026 às 23h12 SP. CSV remoto completo baixado e SHA idêntico ao local. Doze submissões processadas; competição aberta. Primeiros lugares ainda não alcançados | [Resultado público R23](evidence/round23-public.json), [evidência atual](evidence/public.json), [arquivo remoto conferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | v0.4: preview real de 198 slides, 78 pares e 62 fórmulas, com a R23 atual; famílias visuais revisadas e nove analogias antes vazias preenchidas. Narração em geração; novo MP4 e validação audiovisual ainda pendentes. v0.3 histórica: 54min51s, 139 slides, AV/ASR integrais e 139 quadros conferidos, sem declarar escuta humana integral | [Fontes](presentation/), [cobertura v0.3](evidence/aed-content.json), [prova audiovisual v0.3](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e landing profissional com fotos reais, vídeo e download | GitHub, site V6, notebook V6 e pacote v0.3 publicados e verificados. Nova landing local: pesquisa de referências, fotos reais licenciadas, fontes locais, artefato da RealMLP, CTAs vídeo/AED e crítica independente em 1440/1280/390/360 px; sem overflow ou erros JS, reprodução remota curta conferida. Publicação da refatoração em andamento. Notebook V7 em atualização; v0.4 ainda sem pacote publicado | [Créditos dos assets](site/assets/credits.json), [fontes da página](site/), [prova notebook V6](evidence/notebook-v6.json), [entrega v0.3](evidence/delivery.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [notebook](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) |
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
