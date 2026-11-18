# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Baixar a AED v0.4](https://github.com/aikolopes-DS/fraudlab/releases/download/v0.4.0/AED-FraudLab-entrega.zip) · [Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Release e hashes](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.4.0)

Extraia o ZIP e abra `ENTREGA.html`: vídeo, slides, legendas, código, instruções e comprovantes ficam juntos. A v0.4 foi publicada, baixada anonimamente e testada com os arquivos extraídos em desktop e celular. O README incluído no ZIP registra a fonte congelada anterior à publicação; os recibos posteriores estão neste repositório. O envio para disputar a competição é feito no Kaggle: a submissão `58490653` foi processada, com score público 0,95183.

[![Checks](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml/badge.svg)](https://github.com/aikolopes-DS/fraudlab/actions/workflows/check.yml)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 18/11/2026, horário de São Paulo. A melhor submissão é a R23: score público **0,95183**, posição **626/2.135** no snapshot do vídeo, em 17/11 às 23h12 SP. A consulta oficial de 17/11 às 23h49 SP registra **627/2.136**, com o mesmo score. São doze envios processados. A apresentação v0.4, a landing V8, o notebook V7 e o pacote completo estão publicados e verificados: **87min58s, 198 slides, 78 pares e 62 fórmulas**. Primeiras posições não alcançadas; o ranking final depende da avaliação privada ao encerrar a competição.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | R22: RealMLP própria com 16 membros por fold; híbrida OOF 0,95170749, melhora em 4/5 folds sobre R18. R23 reestimou a receita escolhida nos 699.960 rótulos: nova RealMLP, 3 épocas; checkpoints integrais GLM/correção e LightGBM reutilizados e GPT-2 público creditado. Pesos fixos: 42,1875% / 14,0625% / 18,75% / 25%. Recargas independentes e fórmula do CSV conferidas. A OOF citada é histórica da R22, não nova validação da R23. CV exploratória, sem holdout independente | [R22](evidence/round22-results.json), [R23](evidence/round23-results.json), [auditoria R23](evidence/round23-verification.json), [histórico de pesquisa](docs/research.md) |
| Concorrer de verdade | Submissão processada, score e posição com data | Melhor envio `58490653`: score público 0,95183; vídeo registra 626/2.135 às 23h12 SP e a consulta de 23h49 SP registra 627/2.136 em 17/11/2026. CSV remoto completo baixado e SHA idêntico ao local. Doze submissões processadas; competição aberta. Primeiros lugares ainda não alcançados | [Resultado público R23](evidence/round23-public.json), [consulta posterior](evidence/aed-services.json), [arquivo remoto conferido](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | v0.4 publicada: 87min58s, 198 slides, 78 pares e 62 fórmulas; 13 trechos de código literal. Decodificação AV integral, ASR integral e 1.181 legendas conferidas; todos os quadros examinados em contatos, com abertura individual dos densos. Roteiro completo auditado contra fontes, métricas e bibliotecas. Isso não declara escuta humana ou reprodução contínua integral; ASR erra nomes e números. Fórmulas HTML longas exigem paisagem/zoom no celular | [Vídeo e arquivos](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.4.0), [cobertura e timestamps](evidence/aed-content.json), [prova audiovisual](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e landing profissional com fotos reais, vídeo e download | Landing V8 publicada com fotos licenciadas, fontes locais, arquitetura real e CTAs; pesquisa e crítica independente, sem afirmar conversão medida. Quatro larguras revisadas na refatoração; V8 conferida em desktop/celular. Player: 1.181 legendas, 11 saltos no desktop e ranking no celular; slides 198/controles e download passaram. Quatro assets inteiros baixados anonimamente e 471 arquivos internos conferidos; ZIP extraído funciona sem servidor. Notebook V7 público, 45 células e 49 arquivos conferidos; salvo sem execução na nuvem | [Design](evidence/design-review.json), [notebook V7](evidence/notebook-v7.json), [entrega offline](evidence/delivery.json), [publicação](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | CI da fonte congelada do pacote 17d43d7: 369 testes e 81 subtestes passaram; seus 84 arquivos de código foram comparados ao ZIP publicado. Revisões independentes de checkpoints, fórmulas, roteiro e quadros documentadas. O vídeo conserva o snapshot local da fonte 3c569d7, 361 testes e 81 subtestes; os oito testes de exportação do notebook vieram depois. A prova de software identifica a fonte testada atual; versões históricas conservam suas próprias provas | [Testes](tests/), [auditoria R23](evidence/round23-verification.json), [conteúdo](evidence/aed-content.json), [software](evidence/software-v04.json), [CI do pacote](https://github.com/aikolopes-DS/fraudlab/actions/runs/44226439599) |

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
