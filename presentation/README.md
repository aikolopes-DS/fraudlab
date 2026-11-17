# Apresentação

O conteúdo liga os experimentos executados à participação documentada no Kaggle.
Não há limite fixo de duração. Cada slide técnico é seguido imediatamente por uma
explicação simples com exemplo cotidiano; essa sequência é verificada no build.

A [apresentação v0.3](https://github.com/aikolopes-DS/fraudlab/releases/tag/v0.3.0)
tem **139 slides, 54min51s e 738 legendas**, com [player e slides](https://aikolopes-fraudlab.aikol9pes.chatgpt.site).
Ela explica a R10: 56,25% offset com BPE e frequência, 18,75% LightGBM e 25% GPT-2
público v2; inclui as MLPs e a TabM próprias como comparações. O snapshot é a
submissão **58470267**, score **0,95134**, posição **723/2.112** em
**17/11/2026 às 18h24 UTC**. A [prova audiovisual](../evidence/media.json) registra
decodificação e ASR integrais e inspeção dos 139 quadros, sem declarar escuta
humana ou reprodução contínua integral. As versões v0.1 e v0.2 permanecem históricas.

## Gerar

Na raiz do repositório, com as dependências de apresentação instaladas:

```powershell
python -m presentation.build --help
```

Uma versão final exige uma execução MLP real, mesmo quando outro modelo tiver
produzido a submissão selecionada. Para gerar o conteúdo atual, com as execuções
locais completas já disponíveis:

```powershell
$fraudComparisons = @(
  "round10-offset-bpe-frequency", "round9-glm-bpe-frequency", "round9-bpe-frequency-public25",
  "explore5-glm", "explore5-lgbm", "explore5-public-gpt2-v2", "explore5-public25",
  "round8-tabm", "round8-tabm10", "dev-mlp-te-values", "dev-logistic-raw",
  "dev-mlp-standard-raw", "dev-mlp-quantile-raw", "dev-lgbm-leaf4-raw", "dev-lgbm-leaf4-digits"
)
$fraudPresentationArgs = @()
foreach ($fraudRun in $fraudComparisons) {
  $fraudPresentationArgs += @("--comparison", "artifacts/runs/$fraudRun/metrics.json")
}
python -m presentation.build --evidence evidence/public.json `
  --metrics artifacts/runs/round10-offset-bpe-frequency-public25/metrics.json `
  @fraudPresentationArgs --historical-holdout artifacts/holdout/evaluation.json `
  --output artifacts/presentation-v03 --video
```

`--comparison` pode ser repetido para incluir outras execuções. O build exige
execuções com `status: complete`, hashes de dados compatíveis, métricas válidas,
arquitetura MLP registrada e evidência de dados, treino, submissão e posição.
Ele verifica os registros fornecidos; não autentica uma submissão no Kaggle.

A R10 usa `bpe_digits`, frequência por fold, target encoding interno 5/50 e
GLM C=0,3; 1.000 árvores residuais de 4 folhas, taxa 0,03 e L2=5. Sua CV é
5/42, sem holdout independente. A OOF híbrida registrada é 0,9514921382152453;
fonte, parâmetros e resultados estão no [plano](../evidence/round10-plan.json) e
no [resultado R10](../evidence/round10-results.json). A R9 permanece uma exceção
de avaliação pública declarada; a R11 foi rejeitada e não integra este vídeo.

Para regenerar com o mesmo snapshot de evidências publicado, extraia `evidence-snapshot.json`
do ZIP v0.3 e use esse arquivo em `--evidence`. O arquivo `evidence/public.json`
pode registrar consultas posteriores. Omitir `--historical-holdout` é permitido
quando o relatório antigo não estiver disponível; ele não é exigido como uma
reserva independente da R10. Artefatos e checkpoints locais não vêm no clone.

Para uma submissão produzida por refit, forneça suas métricas em `--metrics` e as
métricas do desenvolvimento correspondente em `--comparison`. O refit usa todos
os rótulos com configuração já escolhida e não possui nova AUC OOF. A apresentação
mantém separados desenvolvimento, reserva, refit e pontuação pública.
Inclua o relatório em `--holdout artifacts/holdout/evaluation.json`; seu hash deve
coincidir com o registrado nas métricas do modelo selecionado. O score da reserva
avalia os checkpoints do protocolo de desenvolvimento, não o refit posterior.
Quando uma extensão externa tiver usado as linhas da antiga reserva em seu
treinamento, use `--historical-holdout` no lugar de `--holdout`. O relatório será
mostrado como histórico, sem atribuir independência à nova composição.

Componentes externos são descritos por `model_details.external_components` nas
métricas: autor, código/commit, licenças, fonte das previsões, validação e método
de inferência comprovado. O conteúdo GPT-2 só é ativado por essa evidência;
treinamento externo e trabalho próprio permanecem identificados.

O campo `evidence.run.submission_sha256` deve corresponder ao hash de submissão
nas métricas selecionadas. Sem essa correspondência, a versão final é recusada
e o rascunho oculta a pontuação e a posição de outro arquivo.

Os conceitos são incluídos somente quando registrados nos experimentos:
representações numéricas e de dígitos, target encoding, cross-fitting interno,
logits, quantis, árvores com limite de folhas e os componentes da rede neural.
As comparações mostram os resultados reais, inclusive quando a MLP básica não
supera a referência. A [pesquisa e atribuição](../docs/research.md) documenta as
ideias consultadas e suas fontes.

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
| `timeline.json` | Início e duração de cada slide no vídeo |
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

No comando completo acima, acrescente `--speech-engine sapi --voice "Microsoft Maria Desktop"`.

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

O áudio Edge de cada slide pode ser retomado quando o texto, a voz e o ritmo
coincidem com a requisição salva. Uma mudança em qualquer um desses campos
invalida aquele trecho. Isso evita repetir a narração inteira após falha de rede.

## Revisar o vídeo exportado

```powershell
python -m presentation.review --presentation-dir artifacts/presentation-v03 --asr --asr-model base --device cpu
```

O comando confere hashes, codec, resolução, duração, continuidade dos slides e
tempos das legendas. Extrai um quadro no meio de cada slide e cria montagens
para inspeção visual. Com `--asr`, usa `faster-whisper` para transcrever todo o
áudio **do MP4 exportado**, localmente, e organiza a transcrição por slide junto
do roteiro esperado. A primeira execução baixa os pesos públicos do modelo.
Instale a dependência opcional com `pip install "faster-whisper>=1.2,<2"`.

O relatório registra revisão automática e deixa a cobertura visual como pendente.
ASR pode errar palavras, números e nomes técnicos; não equivale a alguém ter
ouvido o vídeo. A inspeção das montagens também não equivale a assistir ao vídeo
inteiro em reprodução contínua. Registre a cobertura realmente examinada.

## Testar

```powershell
python -m unittest presentation.test_presentation presentation.test_review -v
```

As métricas nos testes são fixtures explícitas. Elas não são resultados da
competição. Os testes cobrem requisitos de evidência, sequência didática,
fórmulas, texto acessível, trechos de código e limites das legendas.
