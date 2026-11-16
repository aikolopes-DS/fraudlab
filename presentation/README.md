# Apresentação

O conteúdo liga os experimentos executados à participação documentada no Kaggle.
Não há limite fixo de duração. Cada slide técnico é seguido imediatamente por uma
explicação simples com exemplo cotidiano; essa sequência é verificada no build.

## Gerar

Na raiz do repositório, com as dependências de apresentação instaladas:

```powershell
python -m presentation.build --help
```

Uma versão final exige uma execução MLP real, mesmo quando outro modelo tiver
produzido a submissão selecionada. Exemplo de comando, usando diretórios de
execuções que já existam:

```powershell
python -m presentation.build `
  --evidence evidence/public.json `
  --metrics artifacts/runs/catboost/metrics.json `
  --comparison artifacts/runs/mlp/metrics.json `
  --output artifacts/presentation `
  --video
```

`--comparison` pode ser repetido para incluir outras execuções. O build exige
execuções com `status: complete`, hashes de dados compatíveis, métricas válidas,
arquitetura MLP registrada e evidência de dados, treino, submissão e posição.
Ele verifica os registros fornecidos; não autentica uma submissão no Kaggle.

Para inspecionar o conteúdo enquanto faltam evidências:

```powershell
python -m presentation.build --evidence evidence/public.json --draft --output artifacts/presentation-draft
```

Esse comando gera um **rascunho marcado**, sem vídeo. Não representa uma entrega
da competição. Nunca troque os campos desconhecidos por números ilustrativos na
evidência pública.

## Artefatos

| Arquivo | Uso |
| --- | --- |
| `slides.html` | Slides editáveis, navegação por teclado, roteiro e impressão em PDF |
| `script.json` | Texto completo e narração de cada slide |
| `slide-*.png`, `formula-*.png` | Renderização 1080p e fórmulas para HTML/vídeo |
| `narration.wav` | Narração completa, quando `--video` é usado |
| `captions.srt`, `captions.vtt` | Legendas com tempos medidos por trecho de áudio |
| `video.mp4` | Vídeo H.264/AAC com slides sincronizados |
| `manifest.json` | Hashes das entradas e saídas, estado e limites da verificação |

Edite `content.py` para alterar o roteiro e regenere os artefatos. O HTML exportado
também pode ser editado diretamente; essas alterações não mudam o vídeo nem são
preservadas em uma nova geração. Os slides de código extraem trechos literais dos
arquivos do projeto e registram arquivo, linha e hash da fonte.

## Voz e revisão

A voz padrão é `pt-BR-AntonioNeural`, pelo pacote `edge-tts`, que envia o roteiro
ao serviço online de leitura da Microsoft. `--voice pt-BR-FranciscaNeural`
seleciona a voz feminina. FFmpeg precisa estar no `PATH`. `--rate` ajusta o ritmo
entre -3 e 3, em passos de 10% para Edge.

Para usar a voz local do Windows sem enviar o roteiro a um serviço externo:

```powershell
python -m presentation.build --metrics artifacts/runs/mlp/metrics.json --video --speech-engine sapi
```

O motor SAPI usa `Microsoft Maria Desktop` por padrão e exige Windows com essa
voz instalada. Se a geração Edge falhar, o build para e informa o motivo; ele
não troca silenciosamente para SAPI. A opção `--voice` também aceita outra voz
SAPI instalada quando esse motor é selecionado.

No Edge, um slide é narrado de uma vez para preservar a prosódia. As legendas
usam os tempos de palavras retornados pelo serviço. No SAPI, cada trecho é
renderizado separadamente e as legendas usam sua duração medida. As trocas de
slides seguem a duração real dos arquivos de áudio; não há estimativa de palavras
por minuto. O manifesto separa geração de revisão: `watched_in_full` e
`audio_reviewed` começam como `false`. Gerar o MP4 não comprova que ele foi
assistido ou que a voz foi aprovada.

## Testar

```powershell
python -m unittest presentation.test_presentation -v
```

As métricas nos testes são fixtures explícitas. Elas não são resultados da
competição. Os testes cobrem requisitos de evidência, sequência didática,
fórmulas, texto acessível, trechos de código e limites das legendas.
