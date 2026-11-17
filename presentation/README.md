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

As fontes em edição para **v0.4** cobrem a R18, seu refit R20 e as comparações posteriores.
Isso ainda não representa um novo vídeo publicado. A receita usa 56,25% de offset
próprio com BPE, frequência, pooling e oito interações; 18,75% de LightGBM
próprio e 25% de GPT-2 externo de Masaya Kawamata. As 92 colunas após engenharia
produzem 223 entradas codificadas no offset; a R10 anterior tinha 191.
OOF híbrida R18: **0,9516524825**, com melhora em cinco folds contra a R10.
O refit R20 reaprendeu os dois componentes próprios nos 699.960 rótulos,
sem nova OOF. Sua submissão **58483010** recebeu **0,95162** e posição
**657/2.126** no snapshot de **17/11/2026 às 21h59min16s UTC**.
O resultado ainda não alcançou as primeiras posições. A OOF R18 permanece
uma referência histórica de seleção, sem atribuição ao arquivo do refit.

## Gerar

Na raiz do repositório, com as dependências de apresentação instaladas:

```powershell
python -m presentation.build --help
```

Uma versão final exige uma execução MLP real, mesmo quando outro modelo tiver
produzido a submissão selecionada. Este exemplo prepara um preview marcado,
sem voz ou vídeo, com métricas completas disponíveis localmente:

```powershell
$fraudComparisons = @(
  "round20-refit-joint-offset", "round20-refit-lgbm",
  "round18-pooled-share100-public25", "round18-joint-bpe-pooled-offset",
  "round10-offset-bpe-frequency-public25", "round10-offset-bpe-frequency",
  "round12-catboost-bpe", "round12-catboost-bpe-w10",
  "round13-pooled-offset", "round13-pooled-share100-public25",
  "round14-full-token-pooled-offset", "round14-pooled-share100-public25",
  "round15-logit-fixed", "round15-rank-fixed", "round15-rank_logit-fixed",
  "round16-original-token-pooled-offset", "round16-pooled-share025-public25",
  "round17-neural-pooled-offset-resumed", "round17-pooled-share050-public25",
  "round19-transductive-pooled-offset", "round19-pooled-share100-public25",
  "round21-nested-owned-stack", "round21-nested-public25",
  "round9-glm-bpe-frequency", "round9-bpe-frequency-public25",
  "explore5-glm", "explore5-lgbm", "explore5-public-gpt2-v2", "explore5-public25",
  "round8-tabm", "round8-tabm10", "dev-mlp-te-values", "dev-logistic-raw",
  "dev-mlp-standard-raw", "dev-mlp-quantile-raw", "dev-lgbm-leaf4-raw", "dev-lgbm-leaf4-digits"
)
$fraudPresentationArgs = @()
foreach ($fraudRun in $fraudComparisons) {
  $fraudPresentationArgs += @("--comparison", "artifacts/runs/$fraudRun/metrics.json")
}
python -m presentation.build --evidence evidence/public.json `
  --metrics artifacts/runs/round20-refit-joint-public25/metrics.json `
  @fraudPresentationArgs --historical-holdout artifacts/holdout/evaluation.json `
  --output artifacts/presentation-v04-preview --draft
```

`--comparison` pode ser repetido para incluir outras execuções. O build exige
execuções com `status: complete`, hashes de dados compatíveis, métricas válidas,
arquitetura MLP registrada e evidência de dados, treino, submissão e posição.
Ele verifica os registros fornecidos; não autentica uma submissão no Kaggle.

A R18 mantém GLM C=0,3, TE interno 5/50 e 1.000 árvores residuais de quatro
folhas, taxa 0,03 e L2=5. `joint_bpe` preserva a base BPE e soma oito tuplas
fixas; o pooling gera 16 logits com suavizações 50/2.000/200. A CV é 5/42,
sem holdout independente. Fonte e critérios estão no
[plano R18](../evidence/round18-plan.json). A R9 permanece uma exceção pública
histórica declarada; resultados negativos não são reclassificados como ganhos.

As métricas da R19 rejeitada, do refit R20 e do stacking R21 estão incluídas
no exemplo. A R21 executada teve OOF própria **0,9512688820** e híbrida
**0,9516568348**, com melhora em quatro dos cinco folds contra a R18.
O ganho híbrido de **0,0000043523** passou o gate predefinido; não demonstra
superioridade robusta nem cria um holdout independente. O envio R21 **58483343**
recebeu **0,95138**, abaixo dos **0,95162** do refit R20; a R21 não foi promovida.
Seu CSV foi conferido byte a byte com o download oficial, e o histórico do
snapshot vincula o resultado ao SHA daquele arquivo. Não lhe atribuímos a
posição da melhor submissão da equipe. R19 usa contagens sem alvo.
R21 usa um meta-treino interno separado e mantém o GPT-2 fora dele. Seus três
logits são entradas do meta-modelo, não três colunas originais. Na OOF, cada
linha usa o meta-fold correspondente. No teste, cada meta-fold aplica sigmoid
e as cinco probabilidades são promediadas. As bases do meta-treino viram 60%
dos rótulos totais, enquanto os checkpoints de inferência viram 80%; essa
diferença de distribuição permanece uma limitação declarada. Esses
conceitos são condicionados à evidência fornecida; um plano não vira treino
executado. Para o vídeo final, congele todos os JSONs, forneça o snapshot em
`--evidence`, selecione as métricas do CSV realmente enviado e remova `--draft`.
Só então use `--video --output artifacts/presentation-v04`. As mídias anteriores
devem permanecer nos seus próprios diretórios e releases.

Para regenerar com o mesmo snapshot de evidências publicado, extraia `evidence-snapshot.json`
do ZIP v0.3 e use esse arquivo em `--evidence`. O arquivo `evidence/public.json`
pode registrar consultas posteriores. Omitir `--historical-holdout` é permitido
quando o relatório antigo não estiver disponível; ele não é exigido como uma
reserva independente da R18. Artefatos e checkpoints locais não vêm no clone.

Para uma submissão produzida por refit, forneça suas métricas em `--metrics` e as
métricas do desenvolvimento correspondente em `--comparison`. O refit usa todos
os rótulos com configuração já escolhida e não possui nova AUC OOF. A apresentação
mantém separados desenvolvimento, reserva, refit e pontuação pública.
O `--holdout` é reservado ao protocolo próprio que registra o hash daquela
avaliação. Para a extensão com GPT-2 e para o refit derivado da R18, a reserva
inicial só pode aparecer em `--historical-holdout`: ela não avalia a nova receita.
Um refit sem OOF não recebe a AUC histórica como se fosse nova medição.

`presentation_context` do snapshot fornece o probe de strings, a suíte de
testes observada e eventuais estados ainda sem resultado. O probe registra
buffer e objetos de strings distintos, com `process_rss_measured: false`;
não é pico de RAM nem benchmark de treinamento. Estados são fatos datados do
snapshot, sem uma segunda lista de acompanhamento. O snapshot final deve
substituí-los pelos resultados efetivamente fechados.
Quando uma extensão externa tiver usado as linhas da antiga reserva em seu
treinamento, use `--historical-holdout` no lugar de `--holdout`. O relatório será
mostrado como histórico, sem atribuir independência à nova composição.

Componentes externos são descritos por `model_details.external_components` nas
métricas: autor, código/commit, licenças, fonte das previsões, validação e método
de inferência comprovado. O conteúdo GPT-2 só é ativado por essa evidência;
treinamento externo e trabalho próprio permanecem identificados.

O conteúdo RealMLP permanece condicional. Ele só aparece para uma execução
`complete` com arquitetura, pré-processamento, parâmetros resolvidos e épocas
de refit registrados. Um smoke sintético não é tratado como treino competitivo.
`presentation/realmlp.py` lê os campos produzidos pelo wrapper da biblioteca
oficial `pytabkit==1.8.0`; ausência ou incompatibilidade desses campos bloqueia
a explicação daquela execução em vez de preencher defaults presumidos.
Os pares cobrem escala robusta e RMS por coluna, PBLD com oito coordenadas no
total, SiLU parametrizada, redes completas vetorizadas, média após softmax,
entropia cruzada suavizada, schedules e seleção interna seguida de refit novo.
A soma das perdas dos membros e a média das probabilidades de inferência
permanecem operações distintas. Resultados só são mostrados quando métricas
completas são fornecidas; esta documentação não atribui AUC à R22 em preparação.

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
