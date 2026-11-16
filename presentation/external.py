"""Conditional explanations of the licensed, externally trained GPT-2 source.

No training configuration is inferred from a model family's usual defaults.
The numerical settings shown here must come from the imported component record.
"""

from __future__ import annotations


def components_from(runs: list[dict]) -> list[dict]:
    unique = {}
    for run in runs:
        for component in run.get("model_details", {}).get("external_components", []):
            key = (component.get("training_code_url"), component.get("training_code_commit"))
            unique.setdefault(key, component)
    return list(unique.values())


def gpt2_slides(component: dict) -> list[dict]:
    """Return slide specifications, always in technical/analogy pairs."""
    inference = component.get("inference", {})
    if component.get("kind") != "gpt2_target_first":
        return []
    slides = []

    def add(title, points, narration, kind="explain", formula="", formula_text=""):
        slides.append({"chapter": "06 / A fonte neural externa", "title": title, "points": points,
                       "narration": narration, "kind": kind, "formula": formula, "formula_text": formula_text})

    def pair(title, points, narration, formula, formula_text, simple_title, simple_points, simple_narration):
        add(title, points, narration, "technical", formula, formula_text)
        add(simple_title, simple_points, simple_narration, "analogy")

    author = component.get("trained_by", "autor não registrado")
    credit_points = [
        f"Treinamento externo: {author}",
        "FraudLab auditou e importou previsões; não executou esse treinamento",
        f"Código: {component.get('code_license', 'licença não registrada')}; previsões: {component.get('predictions_license', component.get('data_license', 'licença não registrada'))}",
        "A contribuição própria inclui validação, alinhamento e comparação da combinação",
    ]
    if component.get("weight") is not None:
        percentage = f"{component['weight'] * 100:g}".replace(".", ",")
        credit_points.append("Peso registrado na composição: " + percentage + "% em probabilidades")
    add("GPT-2 externo: autoria e responsabilidade", credit_points,
       "Esta fonte neural foi treinada por " + str(author) + ". "
       "O projeto não reivindica esse treinamento. Aqui foram "
       "auditados os arquivos, os hashes dos dados, as divisões "
       "e a métrica reproduzida. A fonte descreve adaptação "
       "de GPT-2 ao dataset original e ajuste nos folds da "
       "competição. Consumir previsões públicas com licença "
       "e crédito é uma etapa distinta de treinar o modelo.")
    if component.get("method_credits"):
        add("Créditos do método e da fonte", [
            *[str(credit) for credit in component["method_credits"]],
            "Previsões CC BY 4.0: https://creativecommons.org/licenses/by/4.0/",
        ], "A fonte também atribui o método target-first "
           "a Chris Deotte. A adaptação de linguagem "
           "a tabelas usa a linha de trabalho GReaT, "
           "de Borisov e coautores. O dataset original "
           "é de uditjain13. As previsões compartilhadas "
           "por Masaya Kawamata mantêm crédito e "
           "licença. A conversão e a combinação "
           "feitas aqui são registradas separadamente.")
    if not inference.get("target_first"):
        return slides
    pair("Uma transação passa a ser uma sequência", [
        "Cada célula vira texto no formato coluna is valor",
        "A hipótese de classe aparece antes das características",
        "Exemplo de formato: is_fraud is True, amount_usd is 100.00, ...",
        "O exemplo ilustra o formato; não representa uma linha observada",
    ], "O modelo externo transforma uma linha da tabela em "
       "uma sequência textual. Nomes de colunas e valores "
       "são escritos com um formato definido. A hipótese "
       "True ou False aparece no começo. Cada linha será "
       "avaliada sob as duas hipóteses, usando as mesmas "
       "características. O rótulo real da linha avaliada "
       "não é lido para escolher a hipótese de entrada.",
       r"s_y=[\mathrm{classe}=y,\;\mathrm{campo}_1=x_1,\ldots,\mathrm{campo}_m=x_m]",
       "Sequência sob hipótese y = classe y seguida dos campos e valores da linha",
       "No dia a dia: ler o mesmo recibo sob duas hipóteses", [
           "Primeiro: como seria este recibo se fosse fraude?",
           "Depois: como seria o mesmo recibo se fosse legítimo?",
           "Comparamos a compatibilidade, sem fornecer a resposta verdadeira",
       ], "Imagine analisar o mesmo recibo com duas hipóteses. "
          "Uma supõe fraude e outra supõe legitimidade. "
          "As informações do recibo não mudam. O que muda "
          "é a hipótese sob a qual avaliamos essas informações.")
    pair("Tokens e posições entram no Transformer", [
        "O tokenizer transforma o texto em identificadores de tokens",
        "Um token pode ser parte de uma palavra ou de um número",
        "Vetores de token e posição compõem a representação inicial",
    ], "O Transformer não recebe palavras como objetos "
       "humanos. O tokenizer converte o texto em identificadores "
       "de tokens. Esses tokens podem representar pedaços "
       "de palavras, números ou pontuação. O GPT-2 combina "
       "um vetor aprendido para o token com um vetor de "
       "posição. A fórmula mostra essa entrada, sem "
       "presumir uma dimensão ou quantidade de camadas "
       "não registrada na auditoria.",
       r"h_i^{(0)}=E[t_i]+P[i]",
       "Representação inicial da posição i = vetor do token i mais vetor de sua posição",
       "No dia a dia: cartões e lugares na fila", [
           "O cartão informa qual pedaço do texto está presente",
           "O lugar na fila informa onde esse pedaço aparece",
           "A combinação preserva conteúdo e ordem",
       ], "Pense em cartões com pedaços de uma frase. "
          "Saber o que está escrito importa, mas a ordem "
          "dos cartões também importa. A entrada combina "
          "informação sobre o cartão e sobre sua posição.")
    pair("Linguagem autoregressiva", [
        "Cada token é previsto a partir dos tokens anteriores",
        "A probabilidade da sequência é um produto de probabilidades condicionais",
        "No cálculo, somar log-probabilidades evita produtos minúsculos",
    ], "Um modelo autoregressivo representa uma sequência "
       "como um produto de probabilidades condicionais. "
       "Cada termo considera os tokens anteriores. Para "
       "avaliar uma sequência conhecida, o modelo mede "
       "a probabilidade de cada próximo token observado. "
       "Em vez de multiplicar muitos números pequenos, "
       "o cálculo soma seus logaritmos.",
       r"P(t_1,\ldots,t_T)=\prod_{i=1}^{T}P(t_i\mid t_{<i})",
       "Probabilidade da sequência = produto da probabilidade de cada token dado o prefixo anterior",
       "No dia a dia: completar um formulário passo a passo", [
           "O que já foi preenchido oferece contexto para o próximo campo",
           "A avaliação segue a sequência",
           "O futuro não é consultado por esse passo de previsão",
       ], "Imagine preencher um formulário em sequência. "
          "Os campos já preenchidos dão contexto ao "
          "próximo. O modelo autoregressivo faz uma "
          "previsão a cada passo usando o prefixo disponível.")
    pair("Atenção causal: combinar o contexto permitido", [
        "Q, K e V são projeções aprendidas da representação",
        "A softmax transforma compatibilidades em pesos de atenção",
        "A máscara M bloqueia posições futuras; a fórmula mostra uma cabeça",
    ], "A atenção compara consultas Q com chaves K. "
       "Essas comparações são escalonadas pela raiz da "
       "dimensão das chaves e transformadas em pesos "
       "pela softmax. Os pesos combinam os valores V. "
       "Na máscara causal, posições futuras recebem "
       "menos infinito antes da softmax e deixam de "
       "contribuir. A fórmula representa uma cabeça "
       "de atenção; não atribuímos um número de "
       "cabeças não confirmado ao artefato externo.",
       r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V",
       "Atenção = softmax das compatibilidades escalonadas e mascaradas, multiplicada pelos valores",
       "No dia a dia: consultar as anotações anteriores", [
           "Antes de preencher o próximo campo, algumas anotações ajudam mais que outras",
           "A atenção calcula quanto considerar de cada uma",
           "A máscara impede consultar campos futuros",
       ], "Imagine olhar as anotações anteriores para "
          "preencher o próximo campo. Algumas são mais "
          "relevantes que outras. A atenção calcula "
          "pesos para combinar essas anotações. A "
          "máscara mantém fechadas as páginas que "
          "ainda estariam no futuro da sequência.")
    pair("Normalização e caminho residual", [
        "LayerNorm ajusta a escala da representação dentro do exemplo",
        "O caminho residual soma uma transformação à representação existente",
        "O GPT-2 normaliza a entrada de seus sub-blocos",
    ], "Dentro do GPT-2, normalização e conexões residuais "
       "ajudam a organizar as transformações. A normalização "
       "usa média e variância da representação e parâmetros "
       "aprendidos de escala e deslocamento. O caminho "
       "residual acrescenta a transformação à representação "
       "já existente. Isso é diferente do escalonamento "
       "de colunas do dataset que explicamos anteriormente.",
       r"\mathrm{LN}(h)=\gamma\odot\frac{h-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\quad h'=h+F(\mathrm{LN}(h))",
       "LayerNorm normaliza h com escala e deslocamento aprendidos; o sub-bloco soma sua transformação a h",
       "No dia a dia: acrescentar uma revisão ao rascunho", [
           "A revisão reorganiza a informação em uma escala de trabalho",
           "A contribuição nova é somada ao que já existia",
           "O caminho original permanece disponível",
       ], "Imagine revisar um rascunho sem jogar fora "
          "tudo o que já foi escrito. Você organiza a "
          "informação e acrescenta uma contribuição. "
          "A conexão residual mantém um caminho "
          "para a representação anterior.")
    if inference.get("likelihood_ratio"):
        pair("Pontuar as características sob cada classe", [
            "O prefixo da classe é uma condição, não parte da soma pontuada",
            "São somados os log-scores dos tokens depois do prefixo",
            "O código auditado inclui o token de encerramento",
        ], "Para cada hipótese de classe, o modelo "
           "soma os log-scores dos tokens das "
           "características que aparecem depois do "
           "prefixo. O prefixo é a condição usada "
           "para prever a sequência. Ele não é "
           "contado novamente nessa soma. Isso "
           "permite comparar as características "
           "sob as duas hipóteses sem duplicar "
           "a contribuição da probabilidade da classe.",
           r"L_y(x)=\sum_{i\in\mathrm{corpo}}\log P(t_i\mid\mathrm{prefixo}_y,t_{<i})",
           "Score da classe y = soma das log-probabilidades dos tokens do corpo dado o prefixo e os tokens anteriores",
           "No dia a dia: conferir os detalhes de duas histórias", [
               "A hipótese é o título da história",
               "Os detalhes do recibo formam o corpo avaliado",
               "Queremos saber em qual história esses detalhes se encaixam melhor",
           ], "Imagine duas histórias possíveis para "
              "o mesmo recibo. O título anuncia a "
              "hipótese. A avaliação considera "
              "como os detalhes se encaixam em "
              "cada história. O título não é "
              "usado de novo como se fosse "
              "mais um detalhe independente.")
        pair("Bayes: da verossimilhança ao risco estimado", [
            "L₁ − L₀ compara as características sob as duas classes",
            "O prior vem apenas das linhas usadas no treino do fold",
            "A sigmoid converte o log-odds resultante em probabilidade",
        ], "A diferença entre os dois scores mede "
           "qual classe torna as características "
           "mais prováveis segundo o modelo. Pela "
           "regra de Bayes, acrescentamos o log "
           "da razão prévia das classes. O prior "
           "usa a taxa positiva das linhas de "
           "treino do fold, sem consultar os "
           "rótulos da parte avaliada. A sigmoid "
           "transforma o log-odds em uma previsão. "
           "Isso continua sendo uma estimativa "
           "do modelo, não uma garantia de calibração.",
           r"\ell=L_1-L_0+\log\frac{\pi}{1-\pi},\qquad\hat p=\sigma(\ell)",
           "Log-odds = diferença de verossimilhanças em log mais log-odds do prior; probabilidade = sigmoid desse resultado",
           "No dia a dia: considerar a frequência de partida", [
               "Uma hipótese rara precisa ser considerada junto da evidência",
               "Os detalhes atualizam a suspeita inicial",
               "A frequência de partida não vem da resposta do caso avaliado",
           ], "Imagine investigar um acontecimento "
              "raro. Os indícios importam, mas "
              "também importa a frequência de "
              "partida. Bayes combina a evidência "
              "com essa referência, em vez de "
              "começar supondo que as duas "
              "hipóteses são igualmente frequentes.")
    orders = inference.get("orders")
    if orders is not None and "log" in str(inference.get("order_aggregation", "")).lower():
        pair("Combinar diferentes ordens dos campos", [
            f"Ordens registradas nesta release: {orders}",
            "A mesma ordem é usada sob True e False de cada linha",
            "Log-sum-exp combina as verossimilhanças por classe",
            "Não é uma média dos log-odds entre ordens",
        ], "A ordem dos campos pode afetar um "
           "modelo de linguagem. A fonte registrada "
           f"usa {orders} ordens por linha. Para "
           "cada classe, as verossimilhanças "
           "dessas ordens são combinadas por "
           "log-sum-exp. Depois fazemos a diferença "
           "entre as classes e acrescentamos "
           "o prior. Como a quantidade de ordens "
           "é igual nas duas classes, o fator "
           "de normalização da média cancela "
           "na razão. Isso reduz a dependência "
           "de uma única ordem, sem provar "
           "invariância perfeita à permutação.",
           r"\ell=\log\sum_r e^{L_{1,r}}-\log\sum_r e^{L_{0,r}}+\log\frac{\pi}{1-\pi}",
           "Log-odds = log da soma das verossimilhanças sob classe 1 menos o equivalente sob classe 0, mais o prior em log-odds",
           "No dia a dia: reorganizar o mesmo relato", [
               "Os fatos são os mesmos, mas sua apresentação muda",
               "Considerar várias ordens reduz a dependência de uma apresentação só",
               "A regra matemática define como combinar essas leituras",
           ], "Imagine apresentar os mesmos fatos "
              "em ordens diferentes. Uma ordem "
              "pode destacar um aspecto e outra "
              "pode destacar outro. Considerar "
              "várias apresentações reduz a "
              "dependência de uma única ordem.")
    folds = component.get("validation", {}).get("n_splits")
    if folds and "log" in str(inference.get("test_aggregation", "")).lower():
        pair("O arquivo de teste combina logits dos folds", [
            f"Modelos de folds registrados: {folds}",
            "A fonte calcula a média de seus log-odds para o teste",
            "FraudLab aplica sigmoid ao importar esses valores",
            "Média de logits e média de probabilidades são operações diferentes",
        ], "Para cada linha OOF, a fonte usa "
           "o modelo do fold que não treinou "
           "naquela linha. Para o teste, usa "
           f"a média dos log-odds dos {folds} "
           "modelos. Ao importar o arquivo, "
           "aplicamos sigmoid para obter "
           "probabilidades entre zero e um. "
           "Fazer sigmoid depois da média "
           "não equivale a tirar a média "
           "das probabilidades.",
           r"\hat p_{\mathrm{teste}}=\sigma\left(\frac{1}{K}\sum_{k=1}^{K}\ell_k\right)",
           "Probabilidade do teste = sigmoid da média dos log-odds dos modelos de folds",
           "No dia a dia: escolher a escala antes de combinar", [
               "As avaliações podem ser expressas em escalas diferentes",
               "Combinar primeiro e converter depois pode mudar o resultado",
               "O código precisa respeitar a escala declarada pela fonte",
           ], "Imagine combinar avaliações que "
              "estão em uma escala interna. "
              "Converter cada avaliação e "
              "depois combinar pode produzir "
              "outro número. Por isso primeiro "
              "identificamos a escala do arquivo "
              "e seguimos a regra da fonte.")
    return slides
