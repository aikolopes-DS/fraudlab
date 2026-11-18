# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Baixar a AED v0.4](https://github.com/aikolopes-DS/fraudlab/releases/download/v0.4.0/AED-FraudLab-entrega.zip) · [Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Release e hashes](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.4.0)

Extraia o ZIP e abra `ENTREGA.html`: vídeo, slides, legendas, código, instruções e comprovantes ficam juntos. Nesta fonte congelada, a v0.4 está validada localmente e em empacotamento/publicação; os links da v0.4 passam a valer após a publicação da release. A v0.3 permanece preservada. Não houve envio ao portal da faculdade.

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026, horário de São Paulo. A melhor submissão é a R23: score público **0,95183**, posição **626/2.135** no snapshot do vídeo, às 23h12 SP. Nova consulta oficial às 23h49 SP registra **627/2.136**, com o mesmo score. São doze envios processados. A apresentação v0.4 está validada localmente: **87min58s, 198 slides, 78 pares e 62 fórmulas**. A landing V7 e o notebook V7 estão publicados e verificados; a release v0.4 e a integração dos materiais no site estão em andamento. Primeiras posições não alcançadas; o ranking final depende da avaliação privada ao encerrar a competição.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | R22: RealMLP própria com 16 membros por fold; híbrida OOF 0,95170749, melhora em 4/5 folds sobre R18. R23 reestimou a receita escolhida nos 699.960 rótulos: nova RealMLP, 3 épocas; checkpoints integrais GLM/correção e LightGBM reutilizados e GPT-2 público creditado. Pesos fixos: 42,1875% / 14,0625% / 18,75% / 25%. Recargas independentes e fórmula do CSV conferidas. A OOF citada é histórica da R22, não nova validação da R23. CV exploratória, sem holdout independente | [R22](evidence/round22-results.json), [R23](evidence/round23-results.json), [auditoria R23](evidence/round23-verification.json), [histórico de pesquisa](docs/research.md) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58490653`: score público 0,95183; posição 626/2.135 em 17/11/2026 às 23h12 SP. CSV remoto completo baixado e SHA idêntico ao local. Doze submissões processadas; competição aberta. Primeiros lugares ainda não alcançados | [Resultado público R23](evidence/round23-public.json), [evidência atual](evidence/public.json), [arquivo remoto conferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | v0.4 gerada e validada localmente: 87min58s, 198 slides, 78 pares e 62 fórmulas; 13 trechos de código literal. Decodificação AV integral, ASR integral e 1.181 legendas conferidas; todos os quadros examinados em contatos, com amostragem individual dos densos. Roteiro completo auditado contra fontes, métricas e bibliotecas. Isso não declara escuta humana ou reprodução contínua integral; ASR erra nomes e números. Publicação v0.4 pendente nesta fonte congelada | [Fontes](presentation/), [cobertura e timestamps](evidence/aed-content.json), [prova audiovisual](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e landing profissional com fotos reais, vídeo e download | Landing V7 publicada e verificada no domínio em 1440/1280/390/360 px, com fotografia licenciada, fontes locais, arquitetura real e CTAs. Pesquisa e crítico independente documentados; não se afirma conversão medida. Notebook V7 público, salvo sem execução: 45 células e 49 arquivos remotos reconstituídos e comparados às fontes pinadas. Limite de 1 MB resolvido sem alterar os vocabulários ou módulos. Pacote v0.4 e troca conjunta de vídeo/slides/ZIP no site em preparação | [Revisão de design](evidence/design-review.json), [notebook V7](evidence/notebook-v7.json), [serviços conferidos](evidence/aed-services.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [notebook](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | CI da fonte c21414a: 369 testes e 81 subtestes passaram; 84 arquivos de código conferem com a fonte testada. Revisões independentes de checkpoints, fórmulas de mistura, roteiro completo e todos os quadros finais documentadas. O vídeo conserva seu snapshot de validação local da fonte 3c569d7, 361 testes e 81 subtestes; os oito testes de exportação do notebook vieram depois. Cada versão histórica conserva sua própria prova | [Testes](tests/), [auditoria R23](evidence/round23-verification.json), [conteúdo](evidence/aed-content.json), [prova de software](evidence/software-v04.json), [CI c21414a](https://github.com/aikolopes-DS/fraudlab/actions/runs/44225647965) |

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
