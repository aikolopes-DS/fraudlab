"""Presentation content derived from experiment and competition evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


KAGGLE = "https://www.kaggle.com/competitions/playground-series-s6e11"
SOURCES = [
    ("Competição e métrica", KAGGLE + "/overview"),
    ("ROC-AUC", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html"),
    ("Entropia cruzada binária", "https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html"),
    ("AdamW", "https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html"),
]


@dataclass
class Slide:
    chapter: str
    title: str
    points: list[str]
    narration: str
    kind: str = "explain"
    formula: str = ""
    formula_text: str = ""
    visual: str = ""
    code: str = ""
    code_reference: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


def number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "não verificado"
    if digits:
        return f"{value:.{digits}f}".replace(".", ",")
    return f"{value:,.0f}".replace(",", ".")


def validate_evidence(evidence: dict, runs: list[dict], draft: bool) -> list[str]:
    """A final deck requires evidence, never merely a successful rendering."""
    missing = []
    checks = evidence.get("checks", {})
    for name in ("data_verified", "training_verified", "submission_verified"):
        if checks.get(name) is not True:
            missing.append(name)
    if not evidence.get("competition", {}).get("url"):
        missing.append("competition.url")
    if not evidence.get("links", {}).get("github"):
        missing.append("links.github")
    results = evidence.get("results", {})
    for key in ("submission_id", "observed_at"):
        if not results.get(key):
            missing.append("results." + key)
    score = results.get("public_auc")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        missing.append("results.public_auc")
    rank = results.get("ranking")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        missing.append("results.ranking")
    total = results.get("total_teams")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1 or (isinstance(rank, int) and rank > total):
        missing.append("results.total_teams")
    holdout_auc = results.get("holdout_auc")
    if holdout_auc is not None and (not isinstance(holdout_auc, (int, float)) or isinstance(holdout_auc, bool)
                                    or not math.isfinite(holdout_auc) or not 0 <= holdout_auc <= 1):
        missing.append("results.holdout_auc")
    if not runs:
        missing.append("experiment metrics")
    for run in runs:
        auc = run.get("roc_auc")
        if not isinstance(auc, (float, int)) or isinstance(auc, bool) or not math.isfinite(auc) or not 0 <= auc <= 1:
            missing.append("experiment.roc_auc")
        if run.get("status") != "complete":
            missing.append("experiment.status complete")
        if not run.get("data_hashes"):
            missing.append("experiment.data_hashes")
        elif runs and run["data_hashes"] != runs[0].get("data_hashes"):
            missing.append("experiment compatible data_hashes")
    neural = next((run for run in runs if run.get("model") == "mlp"), None)
    if not neural or not neural.get("model_details", {}).get("architecture"):
        missing.append("trained MLP architecture")
    if missing and not draft:
        raise ValueError("Apresentação final bloqueada; falta evidência: " + ", ".join(missing))
    return missing


def ensure_pairs(slides: list[Slide]) -> None:
    for index, slide in enumerate(slides):
        if slide.kind == "technical" and (
            index + 1 == len(slides) or slides[index + 1].kind != "analogy"
        ):
            raise ValueError(f"Slide técnico {index + 1} sem analogia imediatamente após.")


def create_slides(evidence: dict, runs: list[dict], draft: bool = False) -> list[Slide]:
    missing = validate_evidence(evidence, runs, draft)
    selected = runs[0] if runs else {}
    neural = next((run for run in runs if run.get("model") == "mlp"), {})
    details = neural.get("model_details", {})
    architecture = details.get("architecture", {})
    results = evidence.get("results", {})
    rows = selected.get("rows", {})
    features = selected.get("features", {})
    validation = selected.get("validation", {})
    reserved_run = next((run for run in runs if run.get("validation", {}).get("holdout_fraction", 0) > 0), {})
    slides: list[Slide] = []

    def add(chapter: str, title: str, points: list[str], narration: str, **kwargs: Any) -> None:
        slides.append(Slide(chapter, title, points, narration, **kwargs))

    def pair(chapter: str, title: str, points: list[str], narration: str,
             formula: str, formula_text: str, simple_title: str,
             simple_points: list[str], simple_narration: str, visual: str = "") -> None:
        add(chapter, title, points, narration, kind="technical", formula=formula,
            formula_text=formula_text, visual=visual)
        add(chapter, simple_title, simple_points, simple_narration, kind="analogy")

    add("01 / O problema", "Fraude em cartão: do dado à competição", [
        evidence.get("competition", {}).get("title", "Predicting Credit Card Fraud"),
        "Redes neurais, experimentos reproduzíveis e participação no Kaggle",
        "Rascunho: evidências incompletas" if draft else "Experimentos e submissão documentados",
    ], "Este projeto investiga como reconhecer sinais de fraude em transações de cartão. "
       "Vamos percorrer o problema, os dados, os modelos, a matemática e a avaliação. "
       "Ao final, veremos os códigos e a evidência da participação na competição. "
       + ("Esta versão é um rascunho. Ela não comprova treinamento, submissão ou colocação." if draft else
          "A posição mostrada é uma observação do ranking na data indicada, e pode mudar até o encerramento."))
    add("01 / O problema", "O que exatamente precisamos prever?", [
        "Entrada: as características disponíveis de cada transação",
        "Saída: probabilidade de is_fraud = 1 para cada id do teste",
        "A classificação da competição usa ROC-AUC",
        "Pontuação da competição não equivale à aprovação de um sistema bancário",
    ], "A tarefa não é apenas escrever fraude ou não fraude. Para cada transação do teste, "
       "o modelo deve produzir um número entre zero e um. Esse número permite ordenar os casos "
       "do menos para o mais suspeito. A competição avalia essa ordenação. Um banco real também "
       "precisaria considerar custos, privacidade, monitoramento e regras operacionais, que não "
       "são comprovados por uma boa posição no Kaggle.")
    pair("01 / O problema", "Formalização: aprender uma função de risco", [
        "x: características; y: rótulo observado; θ: parâmetros aprendidos",
        "A previsão estima a chance da classe positiva",
        "Os rótulos do teste não participam do ajuste",
    ], "Na fórmula, x representa o conjunto de características de uma transação. "
       "O rótulo y vale um para fraude e zero para a outra classe. A função f, controlada "
       "pelos parâmetros teta, produz uma estimativa da probabilidade de fraude. "
       "O treinamento ajusta esses parâmetros usando apenas exemplos de treino.",
       r"\hat{p}=f_{\theta}(x)\approx P(y=1\mid x)",
       "p estimado = f teta de x, aproximadamente P de y igual a 1 dado x",
       "No dia a dia: uma fila de investigação", [
           "Pense em separar recibos para conferir primeiro os mais suspeitos",
           "Uma pista isolada pode ser normal; a combinação de pistas muda a suspeita",
           "O número organiza a fila; não é uma sentença sobre uma pessoa",
       ], "Imagine uma pessoa conferindo recibos. Ela percebe alguns sinais e decide quais "
          "merecem atenção primeiro. O modelo tenta aprender uma forma consistente de fazer "
          "essa priorização a partir dos exemplos disponíveis. Suspeita alta não significa "
          "certeza, e uma fila bem ordenada não elimina a necessidade de verificação.")
    add("02 / Os dados", "O que foi observado nos arquivos", [
        f"Treino: {number(rows.get('train'))} registros; teste: {number(rows.get('test'))}",
        f"Variáveis de entrada: {number(features.get('count'))}",
        "Taxa positiva no treino: " + (number(selected.get("target", {}).get("positive_rate", 0) * 100, 3) + "%" if selected.get("target", {}).get("positive_rate") is not None else "não verificada"),
        "Identificadores e alvo são tratados separadamente das entradas",
    ], "Antes de escolher uma arquitetura, é preciso conhecer os arquivos. "
       + (f"Neste experimento foram registrados {number(rows.get('train'))} exemplos de treino "
          f"e {number(rows.get('test'))} exemplos de teste. A quantidade de entradas é "
          f"{number(features.get('count'))}. " if selected else
          "Os arquivos ainda não foram verificados nesta versão. Nenhuma quantidade é apresentada como resultado. ")
       + "O perfil dos dados deve revelar valores ausentes, tipos de variáveis, duplicatas e "
       "desbalanceamento. Os hashes dos arquivos ligam cada experimento à versão dos dados utilizada.")
    if features.get("names"):
        examples = ", ".join(str(name) for name in features["names"][:4])
        add("02 / Os dados", "As entradas usadas neste experimento", [
            f"Numéricas: {number(features.get('numeric'))}; categóricas: {number(features.get('categorical'))}",
            "Exemplos de colunas: " + examples,
            "Excluídas das entradas: " + ", ".join(features.get("excluded", [])),
            "Nenhuma linha de dados é exposta nesta apresentação",
        ], "As variáveis de entrada estão registradas na execução. "
           "Entre elas aparecem " + examples.replace("_", " ") + ". "
           "Tipos numéricos e categorias exigem tratamentos distintos. "
           "O identificador da linha não representa uma explicação da fraude "
           "e o rótulo é a resposta que o modelo precisa aprender a estimar, "
           "por isso ficam separados das entradas.")
    method = str(validation.get("method", "não verificado"))
    add("02 / Os dados", "A divisão precisa representar a pergunta", [
        f"Estratégia registrada: {method}",
        f"Divisões registradas: {number(validation.get('n_splits'))}",
        str(validation.get("note", "Tempo, grupos e duplicatas precisam ser inspecionados antes da escolha.")),
        "Pré-processamento é ajustado somente no treino de cada divisão",
    ], "Uma divisão aleatória não é automaticamente correta. Se houver transações ligadas "
       "à mesma pessoa, ou se o desafio for prever o futuro, a validação precisa considerar "
       "essa estrutura. O método registrado aqui é " + method + ". "
       "Imputação, escalonamento e codificação são ajustados apenas nos dados de treino "
       "da divisão. Aprender essas transformações usando a validação contaminaria a comparação.", kind="technical")
    add("02 / Os dados", "No dia a dia: testar uma rota nova", [
        "Treinar no mesmo caminho que será avaliado pode dar uma impressão otimista",
        "Se a pergunta é sobre amanhã, o teste precisa respeitar o tempo",
        "Se a pergunta é sobre pessoas novas, elas não devem aparecer no treino",
    ], "Imagine testar sua capacidade de dirigir em uma rota nova. "
       "Se você já decorou cada curva desse mesmo trajeto, o teste será mais fácil "
       "do que a situação prometida. A divisão dos dados precisa representar "
       "a novidade que queremos medir, seja outro momento ou outro grupo de exemplos.", kind="analogy")
    pair("03 / A avaliação", "Previsões fora do treino: OOF", [
        "Cada exemplo avaliado fica fora do ajuste que produziu sua previsão",
        "OOF reúne essas previsões para calcular a métrica local",
        "Média e variação entre divisões ajudam a avaliar estabilidade",
    ], "OOF significa fora da divisão de treino. Para avaliar um exemplo, usamos um modelo "
       "que não o viu durante o ajuste. Repetimos esse processo nas divisões planejadas. "
       "A fórmula resume que a previsão do exemplo i é produzida por parâmetros ajustados "
       "sem a divisão k que contém esse exemplo. Comparações repetidas ainda podem se adaptar "
       "demais à validação; por isso a estabilidade e uma verificação independente importam.",
       r"\hat p_i^{\mathrm{OOF}}=f_{\theta_{-k}}(x_i),\quad i\in V_k",
       "Previsão OOF do exemplo i usa modelo treinado sem a divisão de validação que contém i",
       "No dia a dia: estudar sem ver o gabarito", [
           "Um simulado só mede aprendizado se as respostas não forem conhecidas",
           "Cada grupo de questões é reservado enquanto o modelo estuda os outros",
           "Decorar o mesmo simulado também pode enganar",
       ], "Pense em estudar para uma prova. Acertar exercícios que você acabou de decorar "
          "é diferente de resolver perguntas novas. A validação faz o papel do simulado "
          "reservado. Se a gente consulta esse mesmo simulado centenas de vezes para escolher "
          "a melhor estratégia, também começa a decorar suas particularidades.")
    if reserved_run:
        reserved = reserved_run["rows"]
        fraction = reserved_run["validation"]["holdout_fraction"]
        pair("03 / A avaliação", "Uma reserva fora da seleção de modelos", [
            f"Reserva registrada: {number(fraction * 100, 0)}% do treino ({number(reserved.get('holdout'))} linhas)",
            f"Desenvolvimento: {number(reserved.get('development'))} linhas para comparar configurações",
            "A reserva não participa dos folds usados na escolha",
            "Uma avaliação registrada é necessária antes de citar seu score",
        ], "O processo de desenvolvimento também separou uma reserva. "
           "Na fórmula, D é o conjunto de treino completo, desenvolvimento "
           "é a parte usada para escolher os modelos e reserva é a parte "
           "mantida fora desse processo. A interseção é vazia: a mesma linha "
           "não pertence às duas partes. Essa reserva só cumpre seu papel "
           "se não for consultada repetidamente para escolher configurações.",
           r"D=D_{\mathrm{dev}}\cup D_{\mathrm{reserva}},\quad D_{\mathrm{dev}}\cap D_{\mathrm{reserva}}=\varnothing",
           "Dados completos = desenvolvimento união reserva; a interseção entre desenvolvimento e reserva é vazia",
           "No dia a dia: uma prova lacrada", [
               "Os simulados ajudam a decidir como estudar",
               "Uma prova diferente fica guardada até a estratégia estar definida",
               "Se você a consultar sempre, ela deixa de ser independente",
           ], "Pense em guardar uma prova lacrada enquanto escolhe como estudar. "
              "Os simulados ajudam no desenvolvimento da estratégia. "
              "A prova lacrada serve para uma conferência posterior. "
              "Se abrirmos o envelope toda vez que queremos escolher "
              "uma mudança, ele deixa de oferecer essa verificação independente.")
    pair("03 / A avaliação", "ROC-AUC: qualidade da ordenação", [
        "Compara a pontuação de um positivo com a de um negativo",
        "Empates contribuem com metade de um acerto",
        "Não exige escolher um limite fixo para classificar",
    ], "A área sob a curva ROC pode ser interpretada como a chance de um exemplo positivo "
       "receber pontuação maior que um negativo, acrescentando metade da chance de empate. "
       "Na fórmula, s positivo e s negativo são essas pontuações. Uma ordenação aleatória "
       "tem AUC esperada de zero vírgula cinco. AUC igual a um corresponde à separação "
       "perfeita dos pares avaliados. Essa métrica não mede, sozinha, calibração nem custo financeiro.",
       r"\mathrm{AUC}=P(s_+>s_-)+\frac{1}{2}P(s_+=s_-)",
       "AUC = chance de pontuar positivo acima de negativo, mais metade da chance de empate",
       "No dia a dia: ordenar a fila corretamente", [
           "Escolha um recibo fraudulento e um legítimo",
           "O fraudulento ficou antes na fila de investigação?",
           "AUC resume esse acerto sobre todos os pares possíveis",
       ], "Imagine escolher um recibo que realmente era fraude e outro que era legítimo. "
          "Se o sistema colocou o fraudulento antes na fila, acertou a ordem desse par. "
          "A AUC resume esse raciocínio para todos os pares. Isso explica por que mandar "
          "probabilidades é útil, mesmo sem decidir um ponto fixo de aprovação ou bloqueio.")
    if any(run.get("model") == "logistic" for run in runs):
        pair("04 / Os modelos", "Referência: regressão logística", [
            "Combina as entradas por uma soma ponderada",
            "A sigmoid transforma a soma em um número entre 0 e 1",
            "Serve de referência simples antes de aumentar a complexidade",
        ], "A regressão logística cria uma soma ponderada das entradas, mais um termo "
           "constante chamado viés. A função sigmoid transforma essa soma em uma previsão "
           "entre zero e um. Ela é uma referência importante porque ajuda a descobrir "
           "se o ganho de um modelo maior compensa sua complexidade.",
           r"z=w^\top x+b,\qquad \hat p=\frac{1}{1+e^{-z}}",
           "z = produto de pesos e entradas mais viés; p = 1 dividido por 1 mais exponencial de menos z",
           "No dia a dia: uma ficha de pontos", [
               "Cada pista acrescenta ou retira pontos da suspeita",
               "O peso controla a influência de cada pista",
               "A soma vira uma escala contínua de suspeita",
           ], "Pense em uma ficha que soma pontos. Uma característica aumenta a suspeita, "
              "outra reduz, e cada uma tem uma importância. A regressão logística aprende "
              "esses pesos a partir dos exemplos. A ficha é simples e útil, mas pode "
              "não representar relações mais complicadas entre as pistas.")
    if any(run.get("model") == "catboost" for run in runs):
        pair("04 / Os modelos", "Comparação: árvores em sequência", [
            "CatBoost combina muitas árvores de decisão",
            "Cada etapa ajusta a função construída pelas anteriores",
            "A comparação usa a mesma estratégia de validação",
        ], "O CatBoost é uma implementação de gradient boosting com árvores. Cada árvore "
           "faz divisões nas variáveis, e a função final combina suas contribuições. "
           "Na expressão, h t é uma árvore, eta controla a contribuição da etapa e F zero "
           "é o ponto de partida. As novas etapas procuram reduzir o erro da função atual. "
           "Ter um bom modelo de árvores como comparação impede atribuir vantagem à rede neural sem medir.",
           r"F_T(x)=F_0(x)+\eta\sum_{t=1}^{T}h_t(x)",
           "F final = F inicial mais taxa de aprendizado vezes soma das contribuições das árvores",
           "No dia a dia: revisar uma estimativa", [
               "Uma primeira avaliação é feita com regras simples",
               "As revisões seguintes procuram corrigir falhas da avaliação atual",
               "Muitas correções pequenas formam a previsão final",
           ], "Imagine revisar uma estimativa de orçamento. A primeira versão é simples. "
              "Na revisão seguinte, você procura os pontos em que a versão atual errou "
              "e faz uma correção. O processo se repete com ajustes controlados. "
              "No boosting, as árvores cumprem o papel dessas correções sucessivas.")
    if neural:
        hidden = architecture.get("hidden_dims", [])
        hidden_text = " → ".join(str(size) for size in hidden)
        add("05 / A rede neural", "A arquitetura que foi treinada", [
            f"Entradas numéricas: {number(architecture.get('numeric_features'))}",
            "Camadas ocultas: " + (hidden_text or "ver configuração registrada"),
            f"Saída: {architecture.get('output_activation', 'ver configuração registrada')}",
            f"Dropout: {architecture.get('dropout', 'não registrado')}",
        ], "Agora entramos na rede neural que foi efetivamente executada. "
           "As quantidades apresentadas vêm dos metadados do treinamento, não de uma "
           "arquitetura ilustrativa. As camadas ocultas registradas têm dimensões "
           + hidden_text + ". As entradas percorrem essas transformações até a saída, "
           "que produz a pontuação usada na avaliação.", kind="technical", visual="architecture")
        add("05 / A rede neural", "No dia a dia: etapas de uma central de triagem", [
            "As informações entram pela recepção",
            "Etapas intermediárias combinam as pistas",
            "A saída coloca cada caso em uma escala de suspeita",
        ], "Pense em uma central de triagem com várias etapas. A recepção "
           "organiza as informações, as etapas intermediárias combinam pistas "
           "e a saída atribui uma prioridade. A rede faz esse percurso com "
           "operações numéricas. O número de etapas e suas dimensões são "
           "decisões registradas, que precisam ser avaliadas em experimentos.", kind="analogy")
        if "standard scaling" in str(details.get("preprocessing", "")).lower():
            pair("05 / A rede neural", "Preparar entradas numéricas", [
                "Ausências são preenchidas com a mediana do treino da divisão",
                "Média e desvio do treino ajustam a escala de cada variável",
                "As mesmas transformações são aplicadas à validação e ao teste",
            ], "As entradas numéricas têm escalas diferentes. O pré-processamento "
               "preenche ausências com a mediana calculada no treino da divisão "
               "e aplica escalonamento padrão. Na fórmula, a média e o desvio "
               "são calculados nesse treino. O conjunto avaliado não redefine "
               "essas estatísticas. Para uma coluna constante, a implementação "
               "usa uma escala segura, evitando divisão por zero.",
               r"x'_j=\frac{x_j-\mu_{j,\mathrm{treino}}}{\sigma_{j,\mathrm{treino}}}",
               "Entrada escalonada = entrada menos média do treino, dividida pelo desvio do treino",
               "No dia a dia: comparar medidas na mesma referência", [
                   "Uma distância e um valor em dinheiro usam unidades diferentes",
                   "A transformação coloca cada medida em relação à sua própria escala",
                   "A referência é definida no treino, antes de olhar o teste",
               ], "Imagine comparar uma distância com um preço. Seus números "
                  "têm unidades e tamanhos diferentes. Em vez de misturá-los "
                  "diretamente, expressamos cada um em relação à sua escala "
                  "de referência. A referência vem do treino e é mantida "
                  "quando avaliamos os exemplos que ficaram de fora.")
        pair("05 / A rede neural", "Como uma camada transforma as entradas", [
            "W: matriz de pesos; b: viés; h: representação intermediária",
            "A transformação é repetida nas camadas registradas",
            "A ativação permite modelar relações não lineares",
        ], "Uma camada recebe uma representação, multiplica pela matriz de pesos e "
           "acrescenta um vetor de viés. Em seguida aplica uma função de ativação. "
           "A saída se torna a entrada da próxima camada. Sem ativações não lineares, "
           "empilhar essas transformações continuaria representando apenas uma transformação afim.",
           r"h^{(l)}=\phi\left(W^{(l)}h^{(l-1)}+b^{(l)}\right)",
           "Representação da camada l = ativação de pesos vezes representação anterior mais viés",
           "No dia a dia: uma equipe que resume pistas", [
               "Um grupo combina informações e produz um resumo",
               "O próximo grupo combina esses resumos de outra maneira",
               "As combinações são aprendidas, não escritas à mão",
           ], "Pense em uma equipe na qual cada pessoa recebe algumas informações, "
              "combina essas informações e passa um resumo adiante. A próxima equipe "
              "combina os resumos. Na rede, os pesos definem a combinação. A analogia "
              "ajuda a visualizar o fluxo, mas os neurônios são operações matemáticas, "
              "sem compreensão humana das transações.")
        if architecture.get("hidden_activation", "").lower() == "relu":
            pair("05 / A rede neural", "ReLU: uma ativação simples", [
                "Valores negativos são substituídos por zero",
                "Valores positivos seguem adiante",
                "Essa mudança permite compor funções não lineares",
            ], "A ReLU retorna o maior valor entre zero e a entrada. "
               "Isso significa bloquear resultados negativos e manter os positivos. "
               "Sua derivada é um para entradas positivas e zero para negativas, "
               "com uma convenção no ponto zero. A simplicidade reduz o custo da operação, "
               "mas a composição de muitas unidades permite funções mais flexíveis.",
               r"\mathrm{ReLU}(z)=\max(0,z)", "ReLU de z = máximo entre zero e z",
               "No dia a dia: um portão com limite", [
                   "Uma regra só deixa passar sinais acima de zero",
                   "Vários portões combinados criam caminhos diferentes",
                   "O limite atua sobre números internos da rede",
               ], "Imagine um portão que só deixa uma informação passar quando seu "
                  "valor está acima de zero. Um único portão faz pouco. Muitos "
                  "portões conectados permitem comportamentos diferentes conforme "
                  "a combinação de sinais recebida.")
        if architecture.get("categorical_cardinalities"):
            pair("05 / A rede neural", "Embeddings para categorias", [
                "Cada categoria é associada a um vetor aprendido",
                "A dimensão vem da configuração do treinamento",
                "Categoria desconhecida precisa ter tratamento definido",
            ], "Uma categoria, como uma classe de estabelecimento quando essa variável "
               "existe, não possui uma distância numérica natural em relação a outra. "
               "Uma tabela de embeddings atribui um vetor a cada categoria. O treinamento "
               "ajusta os números desse vetor junto com os demais parâmetros. A notação "
               "E de c significa selecionar a linha correspondente à categoria c.",
               r"e(c)=E[c,:]", "Embedding da categoria c = linha c da matriz E",
               "No dia a dia: uma ficha para cada rótulo", [
                   "Em vez de tratar nomes como números em uma escala, cada nome ganha uma ficha",
                   "Os campos da ficha são ajustados conforme ajudam a previsão",
                   "Categorias novas usam a regra registrada no código",
               ], "Pense em uma ficha para cada tipo de item. Um nome sozinho não "
                  "explica sua relação com os outros. A ficha guarda características "
                  "que são ajustadas a partir dos exemplos. O embedding faz isso com "
                  "números aprendidos e uma regra explícita para categorias desconhecidas.")
        pair("06 / O aprendizado", "Sigmoid: do logit à probabilidade", [
            "O logit z pode assumir qualquer valor real",
            "A sigmoid o transforma para o intervalo entre 0 e 1",
            "Valores maiores de z mantêm uma ordem maior de p",
        ], "O logit é uma pontuação que pode ser negativa, zero ou positiva. "
           "A sigmoid o transforma para o intervalo entre zero e um, "
           "mantendo a ordem. Para um logit igual a zero, a saída é "
           "zero vírgula cinco. Logits muito positivos se aproximam de um "
           "e logits muito negativos se aproximam de zero. Esse formato "
           "não garante, sozinho, uma probabilidade bem calibrada.",
           r"p=\sigma(z)=\frac{1}{1+e^{-z}}",
           "Probabilidade p = sigmoid de z = um dividido por um mais exponencial de menos z",
           "No dia a dia: colocar uma nota em uma escala", [
               "A pontuação interna precisa virar uma escala de 0 a 1",
               "A escala preserva qual caso parece mais suspeito",
               "Estar na escala certa não prova que a confiança está calibrada",
           ], "Imagine uma avaliação interna que produz notas sem um limite "
              "fixo. Precisamos colocá-las em uma escala de zero a um. "
              "A sigmoid faz essa conversão sem trocar a ordem dos casos. "
              "Mas usar essa escala não significa que, entre cem casos "
              "com nota zero vírgula oito, exatamente oitenta serão fraude. "
              "Essa é uma pergunta de calibração, que exige avaliação própria.")
        pair("06 / O aprendizado", "Entropia cruzada binária: medir o erro", [
            "O rótulo observado é comparado à probabilidade prevista",
            "A perda penaliza previsões incompatíveis com o rótulo",
            "A implementação combina as operações para estabilidade numérica",
        ], "A camada final produz um número chamado logit. A sigmoid converte esse "
           "número em uma probabilidade. A entropia cruzada binária usa o rótulo "
           "observado e a previsão para medir o erro. Errar com muita confiança "
           "produz uma penalidade grande. O código usa a versão que recebe logits "
           "diretamente para evitar problemas numéricos. Minimizar essa perda "
           "não garante maximizar a AUC; a AUC continua sendo medida na validação.",
           r"\mathcal{L}=-\frac{1}{N}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]",
           "Perda média = menos média de y log p mais 1 menos y vezes log de 1 menos p",
           "No dia a dia: errar com muita certeza custa mais", [
               "Dizer 99% e estar errado recebe uma penalidade forte",
               "Uma previsão cautelosa errada recebe outra penalidade",
               "O treino ajusta a confiança; a competição avalia a ordenação",
           ], "Imagine alguém dando palpites com um grau de certeza. Um palpite "
              "errado feito com certeza quase absoluta é um erro maior do que um "
              "palpite cauteloso. Essa ideia ajuda a entender a entropia cruzada. "
              "Ao mesmo tempo, a competição observa a ordem das pontuações, "
              "então uma perda menor não é automaticamente uma AUC maior.")
        pair("06 / O aprendizado", "Backpropagation: atribuir influência ao erro", [
            "A regra da cadeia conecta o erro aos parâmetros de cada camada",
            "O gradiente aponta como a perda varia com cada parâmetro",
            "O framework calcula essas derivadas automaticamente",
        ], "Depois de calcular a perda, precisamos saber como cada peso influenciou "
           "esse erro. A regra da cadeia permite multiplicar derivadas ao longo "
           "das operações. O backpropagation organiza esse cálculo do resultado "
           "em direção às entradas. O gradiente obtido não é a atualização em si: "
           "ele informa ao otimizador a sensibilidade da perda em relação aos parâmetros.",
           r"\frac{\partial\mathcal{L}}{\partial W}=\frac{\partial\mathcal{L}}{\partial z}\frac{\partial z}{\partial W}",
           "Derivada da perda em relação a W = derivada da perda em relação a z vezes derivada de z em relação a W",
           "No dia a dia: rastrear uma receita que deu errado", [
               "O resultado final saiu diferente do esperado",
               "Você rastreia quais etapas e quantidades influenciaram o resultado",
               "A derivada mede essa sensibilidade de forma matemática",
           ], "Pense em uma receita que ficou salgada. Você volta pelas etapas "
              "para entender quais quantidades contribuíram para o resultado. "
              "Na rede, esse caminho de volta usa derivadas, não intuição. "
              "Elas indicam como uma pequena mudança em cada parâmetro afetaria o erro.")
        optimizer = str(details.get("optimizer", "ver configuração registrada"))
        add("06 / O aprendizado", "O otimizador executado", [
            "Otimizador: " + optimizer,
            f"Lotes: {details.get('batch_size', 'ver configuração')} exemplos",
            "O gradiente é estimado em lotes de treino",
            "A validação acompanha a generalização ao longo das épocas",
        ], "O otimizador registrado para esta execução é " + optimizer + ". "
           "O treino percorre pequenos lotes de exemplos, calcula a perda e os "
           "gradientes, e atualiza os parâmetros. Uma época corresponde a "
           "percorrer o conjunto de treino. O tamanho do lote, a taxa de "
           "aprendizado e o número de épocas afetam custo e resultado e estão "
           "registrados na configuração reproduzível.", kind="technical")
        add("06 / O aprendizado", "No dia a dia: corrigir exercícios por blocos", [
            "Um lote é um bloco de exercícios",
            "Uma época é uma passagem pelo material de treino",
            "Depois de cada bloco, os parâmetros recebem um ajuste",
        ], "Imagine estudar resolvendo blocos de exercícios. Depois de cada "
           "bloco, você confere os erros e ajusta sua maneira de resolver. "
           "Quando percorre todo o material, completou uma passagem. "
           "No treinamento, o bloco é o lote, a passagem é a época "
           "e o ajuste é calculado pelo otimizador.", kind="analogy")
        if "adamw" in optimizer.lower():
            pair("06 / O aprendizado", "AdamW: passos adaptativos e decaimento", [
                "m e v resumem gradientes e seus quadrados ao longo do treino",
                "η controla o tamanho global do passo",
                "λ controla o decaimento dos pesos separado do gradiente",
            ], "O AdamW mantém médias móveis dos gradientes e dos seus quadrados. "
               "Essas estimativas, com correção de viés, adaptam o passo para cada "
               "parâmetro. A fórmula mostra o passo adaptativo e o decaimento "
               "separado dos pesos. Eta é a taxa de aprendizado, lambda controla "
               "o decaimento e epsilon evita divisões numericamente problemáticas.",
               r"\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}",
               "Novo parâmetro = parâmetro com decaimento menos passo adaptativo calculado a partir das médias dos gradientes",
               "No dia a dia: ajustar o passo da caminhada", [
                   "Em trechos instáveis, passos menores ajudam",
                   "O histórico influencia o tamanho de cada ajuste",
                   "O decaimento também reduz o crescimento dos pesos",
               ], "Pense em caminhar por um terreno irregular. Usar sempre o "
                  "mesmo tamanho de passo pode atrapalhar. O AdamW usa o histórico "
                  "dos gradientes para ajustar os passos. Separadamente, "
                  "o decaimento reduz os pesos. A analogia explica a ideia, "
                  "mas os critérios exatos são os números da fórmula.")
        if architecture.get("dropout", 0):
            pair("06 / O aprendizado", "Dropout: reduzir dependências internas", [
                f"Probabilidade registrada: {architecture['dropout']}",
                "No treino, algumas ativações são zeradas aleatoriamente",
                "Na inferência, a máscara aleatória é desativada",
            ], "O dropout sorteia uma máscara durante o treinamento. Cada ativação "
               "pode ser zerada com probabilidade d. As demais são reescaladas "
               "por um sobre um menos d, preservando a esperança da ativação. "
               "Isso reduz a dependência de caminhos internos específicos. "
               "Na inferência, o modelo usa todas as ativações e não aplica essa máscara.",
               r"\tilde h=\frac{m\odot h}{1-d},\quad m_j\sim\mathrm{Bernoulli}(1-d)",
               "Ativação com dropout = máscara vezes ativação dividida por 1 menos d; máscara tem chance 1 menos d de valer 1",
               "No dia a dia: uma equipe que precisa se virar", [
                   "Durante os ensaios, parte da equipe fica temporariamente indisponível",
                   "Os demais precisam aprender a trabalhar sem depender sempre dos mesmos",
                   "Na apresentação final, a equipe toda participa",
               ], "Imagine uma equipe ensaiando com algumas pessoas temporariamente "
                  "fora de cada rodada. Os demais precisam se adaptar e evitar "
                  "depender de uma única pessoa. Na hora de apresentar, todos "
                  "participam. O dropout faz algo parecido com ativações numéricas.")
        if details.get("early_stopping_patience"):
            pair("06 / O aprendizado", "Early stopping: escolher uma época", [
                "A métrica de validação é acompanhada a cada época",
                "O checkpoint de melhor validação é preservado",
                "Essa seleção faz parte do ajuste; não é uma avaliação independente",
            ], "O early stopping acompanha a métrica de validação e conserva o "
               "estado com melhor resultado. Quando a melhora deixa de ocorrer "
               "por um número definido de épocas, o treino para. Na fórmula, "
               "t estrela é a época que maximiza a AUC observada na validação. "
               "Como a validação escolhe essa época, ela participa da seleção do modelo.",
               r"t^*=\operatorname{arg\,max}_t\;\mathrm{AUC}_{\mathrm{valid}}(t)",
               "Época selecionada = época de maior AUC de validação",
               "No dia a dia: guardar a melhor tentativa", [
                   "Mais tempo ensaiando não garante melhora",
                   "Uma tentativa pode ficar melhor que as seguintes",
                   "Guardamos a melhor versão observada no ensaio",
               ], "Imagine gravar várias tentativas de uma apresentação. "
                  "A última pode estar mais cansada e pior que uma anterior. "
                  "Você guarda a melhor tentativa do ensaio. Para saber "
                  "como ela funciona com uma plateia nova, ainda precisa "
                  "de uma avaliação fora desse processo de escolha.")
    else:
        add("05 / A rede neural", "Arquitetura e treinamento: pendentes", [
            "Nenhuma execução MLP foi fornecida a esta versão",
            "A arquitetura será descrita a partir de model_details",
            "Não há resultado neural a apresentar como observado",
        ], "Esta versão ainda não recebeu os metadados de uma rede neural treinada. "
           "Por isso não apresenta uma arquitetura hipotética como se tivesse sido "
           "executada. O roteiro matemático específico será incluído a partir "
           "dos mecanismos registrados no experimento real.")
    if selected.get("model") == "blend":
        pair("07 / Os experimentos", "Combinar previsões", [
            "A combinação usa previsões compatíveis por id e divisão",
            "Pesos não negativos somam 1",
            "O ganho deve ser comparado com os modelos isolados",
        ], "Uma combinação ponderada soma as previsões de modelos diferentes. "
           "Os pesos somam um e as previsões devem estar alinhadas pelo mesmo "
           "identificador. Se os erros forem complementares, a combinação "
           "pode melhorar. Não existe garantia de ganho. Escolher muitos "
           "pesos sobre a mesma validação também pode superajustar.",
           r"\hat p=\sum_j w_j\hat p_j,\quad w_j\geq 0,\quad\sum_j w_j=1",
           "Previsão combinada = soma dos pesos vezes previsões, com pesos não negativos somando 1",
           "No dia a dia: combinar opiniões complementares", [
               "Duas pessoas podem perceber pistas diferentes",
               "Se ambas cometem sempre o mesmo erro, combinar ajuda pouco",
               "A combinação precisa ser medida, não presumida",
           ], "Imagine pedir a duas pessoas para conferir um orçamento. "
              "Uma percebe erros de quantidade e a outra percebe valores "
              "fora do padrão. A combinação pode ajudar se elas trouxerem "
              "informações diferentes. Se as duas repetem os mesmos erros, "
              "ter duas opiniões não resolve o problema.")
    scores = [f"{run.get('model', 'modelo')}: AUC OOF {number(run.get('roc_auc'), 5)}"
              for run in runs]
    add("07 / Os experimentos", "Resultados locais registrados", scores[:5] or [
        "Nenhum resultado de treinamento verificado nesta versão",
    ], "Os números deste slide vêm dos arquivos de métricas das execuções "
       "fornecidas. " + (". ".join(scores) + ". " if scores else
       "Não há métricas de treinamento para comparar neste rascunho. ") +
       "A mesma estratégia de validação e os mesmos dados são necessários "
       "para uma comparação justa. Diferenças pequenas precisam ser lidas "
       "junto da variação entre divisões, sementes e custo de execução.")
    if selected:
        add("07 / Os experimentos", "O resultado selecionado e seus limites", [
            f"Modelo selecionado: {selected.get('model', 'não verificado')}",
            f"AUC OOF: {number(selected.get('roc_auc'), 5)}; desvio entre divisões: {number(selected.get('fold_std'), 5)}",
            f"Tempo registrado: {number(selected.get('elapsed_seconds'), 1)} segundos",
            "Score local e score público medem amostras diferentes",
        ], "A execução selecionada é " + str(selected.get("model")) +
           ". Seu resultado local é uma estimativa sobre as divisões usadas. "
           "O desvio entre divisões descreve a variação observada, não "
           "um intervalo de confiança automático. O tempo registrado "
           "ajuda a avaliar o custo do ganho. Uma pontuação maior no "
           "ranking público pode não se repetir na parte privada do teste.")
    if results.get("holdout_auc") is not None:
        add("07 / Os experimentos", "Conferência independente registrada", [
            f"AUC da reserva: {number(results['holdout_auc'], 5)}",
            f"Observação: {results.get('holdout_observed_at') or 'data não informada'}",
            "O registro precisa identificar a configuração congelada avaliada",
            "A reserva não deve voltar a orientar a seleção de hiperparâmetros",
        ], "Além das métricas usadas no desenvolvimento, foi fornecido "
           "um registro de avaliação da reserva, com AUC de "
           + number(results["holdout_auc"], 5) + ". Esse resultado só representa "
           "uma conferência independente se a configuração tiver sido definida "
           "antes de consultar a reserva. A data e a configuração avaliada "
           "devem acompanhar essa evidência.")
    add("08 / Da execução à competição", "Como o código produz a submissão", [
        "Inspecionar arquivos → definir validação → treinar e avaliar",
        "Salvar configurações, hashes, métricas e previsões por id",
        "Validar CSV → submeter → conferir processamento no Kaggle",
        "Manter dados brutos e credenciais fora do repositório público",
    ], "A operação começa pela inspeção dos dados e pela escolha da validação. "
       "O treinamento salva sua configuração, os hashes dos arquivos, as "
       "métricas e as previsões. Antes do envio, o CSV precisa ter uma "
       "probabilidade válida para cada identificador esperado. Depois "
       "do envio, é necessário conferir o processamento no Kaggle. "
       "Ter um arquivo no computador não comprova uma submissão.", visual="pipeline")
    add("08 / Da execução à competição", "Código público e reprodução", [
        evidence.get("links", {}).get("github", "URL do repositório ainda não verificada"),
        "python -m fraudlab inspect --help",
        "python -m fraudlab train --help",
        "A configuração completa de cada execução acompanha suas métricas",
    ], "O repositório público permite inspecionar a implementação em vez "
       "de depender apenas dos slides. Os pontos de entrada da execução "
       "são os comandos de inspeção, treinamento e validação da submissão. "
       "As dependências e a configuração registrada permitem reproduzir "
       "o procedimento com acesso autorizado aos dados. Testes sintéticos "
       "verificam contratos do código, mas não demonstram desempenho competitivo.")
    add("09 / A evidência final", "Submissão e posição observadas", [
        f"AUC pública: {number(results.get('public_auc'), 5)}",
        f"Posição: {number(results.get('ranking'))} de {number(results.get('total_teams'))} equipes",
        f"Submissão: {results.get('submission_id') or 'não verificada'}",
        f"Consulta: {results.get('observed_at') or 'não verificada'}",
    ], "A prova de participação é a submissão processada pela plataforma. "
       + (f"O registro fornecido informa AUC pública de {number(results.get('public_auc'), 5)} "
          f"e posição {number(results.get('ranking'))}, observadas em {results.get('observed_at')}. "
          if not draft else "Neste rascunho, os campos ausentes permanecem explicitamente não verificados. ") +
       "Essa é uma fotografia da classificação na data indicada. "
       "A classificação definitiva depende do encerramento e da avaliação "
       "privada. Buscar vencer orienta os experimentos; não é uma "
       "declaração de que a vitória foi alcançada.")
    limitations = [str(item) for item in evidence.get("limitations", [])]
    if draft:
        limitations = ["Rascunho; falta evidência: " + ", ".join(missing)] + limitations
    add("09 / A evidência final", "O que a evidência permite afirmar", (limitations or [
        "Os resultados apresentados correspondem às execuções identificadas",
        "O ranking é uma observação datada e pode mudar",
        "A generalização fora desta competição não foi comprovada",
    ])[:4], "A conclusão precisa respeitar os limites da evidência. "
       + " ".join(limitations) + " " +
       "Não basta um teste passar, um build terminar ou uma página abrir "
       "para provar o fluxo operacional inteiro. O valor deste trabalho "
       "está em ligar cada escolha a código, experimento, resultado "
       "e evidência verificáveis.")
    add("Referências", "Fontes e próximos pontos de inspeção", [name + ": " + url for name, url in SOURCES],
        "As fontes primárias listadas documentam a competição, a métrica "
        "e as operações matemáticas utilizadas. O repositório registra "
        "as versões instaladas e as configurações dos experimentos. "
        "Esses materiais permitem revisar as decisões e localizar "
        "os trechos de código por trás de cada explicação.")
    ensure_pairs(slides)
    return slides
