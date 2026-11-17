# FraudLab

Detecção de fraude em cartão: experimentos reproduzíveis, participação real no Kaggle e uma apresentação em português que liga a matemática ao cotidiano.

[Estudo de caso e player](https://aikolopes-fraudlab.aikol9pes.chatgpt.site) · [Slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site/presentation/slides.html) · [Vídeo e downloads](https://github.com/aikolopes-og/fraudlab/releases/tag/v0.1.0)

O objetivo é disputar as primeiras posições no [Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11). O prazo publicado é 30/11/2026 às 23h59 UTC. A métrica é ROC-AUC. A posição pública durante a competição é provisória; vencer depende dos resultados, não de uma promessa.

## Estado do projeto

Este é o único acompanhamento de tarefas. Atualizado em 17/11/2026. Quatro submissões foram processadas. Nova rodada em execução: GLM com curvas suaves e árvores residuais sobre sua margem, na mesma CV 5/42. O resultado anterior e a apresentação v0.1 permanecem como snapshots datados até haver uma melhoria comprovada.

| Pedido | Resultado esperado | Estado real | Evidência |
| --- | --- | --- | --- |
| Competição vigente e inscrição | Regras verificadas, acesso aos dados e conta inscrita | Conferido; arquivos oficiais baixados pela conta `aikoog` | [Regras verificadas](docs/competition.md) |
| Programar para competir | Rede neural treinada, comparação e melhoria com validação correta | v0.1 selecionada e publicada; implementação e validação da rodada de curvas suaves/árvores residuais em andamento. Sem holdout independente nesta extensão | [Experimentos agregados](evidence/experiments.json), [seleção](evidence/hybrid-selection.json), [origem externa](evidence/external-gpt2.json) |
| Concorrer de verdade | Submissão processada, score e posição com data | Quarto envio: 0,95097; posição pública 804/2.010 em 16/11/2026 às 17h23 UTC. A meta das primeiras posições segue aberta | [Evidência](evidence/public.json), submissão `58438450`; [CSV remoto idêntico ao local](evidence/submission-file-check.json) |
| Apresentar com profundidade | Slides narrados PT-BR, matemática seguida de analogia, código e ranking | 116 slides e vídeo de 44min27s gerados; decodificação AV integral, ASR integral e inspeção dos 116 quadros passaram. Sem alegação de escuta humana ou reprodução contínua integral | [Apresentação](presentation/), [prova audiovisual](evidence/media.json) |
| Hospedar e usar no portfólio | Repositório público e estudo de caso acessível | Site V3 e notebook V3 públicos; vídeo e ZIP na release GitHub. Acesso anônimo, reprodução/avanço e 604 legendas verificados em desktop/mobile; ZIP baixado anonimamente e íntegro | [Publicação verificada](evidence/publication.json), [site](https://aikolopes-fraudlab.aikol9pes.chatgpt.site), [downloads](https://github.com/aikolopes-og/fraudlab/releases/tag/v0.1.0) |
| Testar e revisar | Contratos testados, revisão independente e verificação publicada | 84 testes e 52 subtestes locais passaram; recorder conferido com Kaggle real; revisão cruzada e correções integradas; 29 fontes publicadas correspondem aos hashes testados. CI remota bloqueada por cobrança da conta GitHub, sem iniciar job | [Testes](tests/), [verificação](evidence/verification.json), [CI](.github/workflows/check.yml) |

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
