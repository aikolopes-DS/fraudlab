"""Later methods and comparisons, included only from supplied execution evidence."""

import re

from .continuation import name, selected_runs


def complete(runs):
    return [run for run in runs if run.get("status") == "complete"]


def decimal(value):
    return f"{value:g}".replace(".", ",")


def pooling_run(runs):
    return next(
        (
            run
            for run in selected_runs(runs)
            if run.get("model_details", {}).get("pooled_tokens", {}).get("algorithm")
            == "hierarchical_pooled_decimal_token_rates"
        ),
        None,
    )


def has_transductive(run):
    return bool(run.get("model_details", {}).get("transductive_frequency"))


def append_pooling_pairs(pair, runs):
    run = pooling_run(runs)
    if not run:
        return
    metadata = run["model_details"]["pooled_tokens"]
    smoothing = metadata["smoothing"]
    alpha, beta = smoothing["pooled_to_prior"], smoothing["field_to_pool"]
    gamma = smoothing["markov_to_field_decimal"]
    folds = metadata["crossfit"]["n_splits"]
    pair(
        "04 / Representar os dados",
        "Pooling: compartilhar evidência entre colunas",
        [
            "Grupos compartilham partes decimais ou tokens de campos compatíveis",
            f"Suavização: grupo → prior {alpha:g}; campo → grupo {beta:g}",
            f"Cross-fitting interno de {folds} partes; o próprio rótulo fica fora",
            "As 16 taxas viram logits e são padronizadas; não são 16 novas medições",
        ],
        "Alguns pedaços numéricos aparecem em várias colunas. Em vez de estimar "
        "cada combinação isoladamente, reunimos contagens de ocorrências e "
        "rótulos positivos no grupo de campos. Na fórmula, s é a soma de "
        "positivos, n é o número de ocorrências e pi é a média do treino "
        "interno disponível. A taxa do grupo é suavizada em direção a pi. "
        "Depois, a taxa de cada campo é suavizada em direção ao grupo. Os "
        f"pesos registrados são {alpha:g} e {beta:g}. São forças de suavização, "
        "não quantidades de observações novas. Uma linha pode contribuir "
        "em vários campos do grupo; não tratamos essas ocorrências como "
        "pessoas independentes. As 16 taxas são produzidas com cross-fitting, "
        "convertidas em logits e padronizadas dentro do treino permitido.",
        r"q_G(v)=\frac{\sum_{j\in G}s_j(v)+\alpha\pi}{\sum_{j\in G}n_j(v)+\alpha},\quad r_j(v)=\frac{s_j(v)+\beta q_G(v)}{n_j(v)+\beta}",
        "Taxa do grupo = positivos mais alfa vezes prior, sobre ocorrências mais alfa; taxa do campo combina positivos com beta vezes a taxa do grupo",
        "No dia a dia: avaliações de lojas da mesma rede",
        [
            "Uma loja com poucas avaliações começa usando a referência da rede",
            "A própria rede também usa uma referência geral quando há poucos exemplos",
            "Com mais avaliações, a informação específica ganha espaço",
        ],
        "Pense numa rede de lojas. Para uma filial com poucas avaliações, "
        "a média da rede oferece uma referência. Se a rede inteira ainda "
        "tem poucas avaliações, também recorremos a uma referência geral. "
        "Com mais evidência, a informação daquela filial ganha espaço. "
        "Isso ajuda a entender os dois níveis de suavização, sem "
        "transformar a analogia numa regra sobre fraude.",
    )
    pair(
        "04 / Representar os dados",
        "Pooling condicional: token e decimal juntos",
        [
            "Quatro campos combinam último token inteiro com o decimal completo",
            f"A taxa da combinação é suavizada com peso {gamma:g} para a taxa decimal do campo",
            "Combinação desconhecida retorna à taxa decimal e às referências anteriores",
            "Nenhum alvo da validação ou do teste entra nessas contagens",
        ],
        "O último bloco do pooling acrescenta uma condição. Para quatro "
        "campos, cruzamos o último token da parte inteira com o texto "
        "decimal completo. A taxa dessa combinação recebe evidência "
        f"específica e uma referência de peso {gamma:g}, que é a taxa decimal "
        "daquele campo já suavizada. Uma combinação nunca observada retorna "
        "diretamente a essa referência. Se o próprio decimal não foi visto, "
        "a etapa anterior recorre ao grupo e ao prior. O nome Markov no "
        "código descreve essa dependência entre pedaços do número; não "
        "significa que treinamos uma cadeia temporal de transações. "
        "Os mesmos folds internos excluem o rótulo da linha codificada.",
        r"r_j(t,v)=\frac{s_j(t,v)+\gamma r_j(v)}{n_j(t,v)+\gamma}",
        "Taxa da combinação token e decimal = positivos mais gama vezes referência decimal, dividido por ocorrências mais gama",
        "No dia a dia: uma receita com uma variação nova",
        [
            "Você conhece a receita básica, mas tem poucas opiniões sobre uma variação",
            "Começa pela avaliação da receita básica e acrescenta a evidência da variação",
            "Sem nenhuma opinião nova, mantém a referência básica",
        ],
        "Imagine uma receita conhecida que ganhou uma variação. Se só "
        "uma pessoa provou a variação, você ainda se apoia bastante na "
        "receita básica. Novas avaliações podem mudar essa impressão. "
        "Sem nenhuma avaliação da variação, fica a referência da receita. "
        "A fórmula faz uma combinação semelhante, com contagens e um "
        "peso fixado antes do experimento.",
    )


def append_joint_pair(pair, runs):
    run = next(
        (
            run
            for run in selected_runs(runs)
            if run.get("feature_encoding", {})
            .get("joint_encoding", {})
            .get("output_count")
            == 8
        ),
        None,
    )
    if not run:
        return
    pair(
        "04 / Representar os dados",
        "Oito interações fixas: cruzar pistas",
        [
            "Estrangeira×IP; categoria×risco10; categoria×faixa do valor",
            "Golpe por IA×velocidade5; lojista novo×dia; hora×dia",
            "Idade10×anos de cartão; autenticação×canal",
            "Tuplas categoriais; cortes determinísticos, sem busca G com os rótulos",
        ],
        "Uma interação permite que a combinação de duas pistas tenha uma "
        "contribuição diferente das pistas separadas. Foram fixadas oito "
        "tuplas: transação estrangeira com divergência de IP; categoria "
        "com risco dividido em faixas de dez; categoria com faixa "
        "logarítmica do valor; indicador de golpe por inteligência artificial com velocidade em faixas de "
        "cinco; lojista novo com dia da semana; hora com dia da semana; "
        "idade em décadas com meses de cartão divididos por doze; e "
        "autenticação com canal. Piso significa arredondar para o inteiro "
        "inferior. A faixa logarítmica começa nos limites zero, um, três, "
        "sete e assim por diante. As tuplas preservam as categorias sem "
        "ambiguidade de separadores. Não multiplicamos os riscos nem "
        "procuramos pares usando todos os alvos. A escolha foi informada "
        "por pesquisa pública e domínio e continua exploratória.",
        r"J(x)=(u_a(x_a),u_b(x_b)),\quad u_{10}(v)=\lfloor v/10\rfloor,\quad u_{\log}(v)=\lfloor\log_2(1+v)\rfloor",
        "Interação = tupla de duas representações; faixa dez usa piso de valor dividido por dez; faixa log usa piso de log base dois de um mais valor",
        "No dia a dia: rua e número formam um endereço",
        [
            "A rua sozinha não identifica uma casa; o número sozinho também não",
            "Juntos, os dois campos criam uma informação mais específica",
            "Agrupar valores em faixas reduz combinações muito raras",
        ],
        "Pense num endereço. A rua sozinha não identifica uma casa, "
        "e o número sozinho também não. Juntar os dois cria uma "
        "informação mais específica. Agrupar valores em faixas seria "
        "como falar de um trecho da rua, quando faltam exemplos de "
        "cada casa. Aqui a combinação vira uma categoria que o "
        "modelo precisa avaliar nos dados de treino.",
    )


def append_neural_offset_pair(pair, runs):
    run = next(
        (run for run in complete(runs) if run.get("model") == "neural_offset"), None
    )
    if not run:
        return
    architecture = run["model_details"]["architecture"]
    dimensions = architecture["hidden_dims"]
    pair(
        "06 / Experimento neural",
        "Rede residual: corrigir um GLM congelado",
        [
            f"{architecture['input_features']} entradas; camadas {dimensions}; SiLU; dropout {decimal(architecture['dropout'])}",
            "SiLU(a)=a·sigmoid(a); cabeça de saída começa com pesos e viés zero",
            "BCE recebe a soma das margens; GLM e rede usam o mesmo treino cross-fitted",
            "Época escolhida dentro de 15%; depois GLM, rede e AdamW são refeitos por E",
        ],
        "Nesta experiência, a rede não substitui o GLM: aprende uma "
        "correção para sua margem congelada. A entrada é a mesma matriz "
        "de treino com codificações cross-fitted usada pelo GLM. Foram "
        f"registradas {architecture['input_features']} entradas e duas camadas "
        f"de {dimensions[0]} e {dimensions[1]} unidades. A ativação SiLU "
        "multiplica a entrada pela sua sigmoid; o dropout foi registrado "
        f"como {decimal(architecture['dropout'])}. A última camada começa com pesos e "
        "viés zero. Assim, a primeira margem combinada é a do GLM "
        "convertida para float trinta e dois. A perda binária é calculada "
        "sobre a soma das margens, não sobre um rótulo residual improvisado. "
        "Uma divisão interna de quinze por cento escolhe E. Depois, "
        "pré-processamento, GLM, rede e otimizador são descartados e "
        "reaprendidos no treino externo completo por E épocas. Os "
        "rótulos da validação externa não entram nessa escolha.",
        r"z=z_{\mathrm{GLM}}+r_\theta(\phi(x)),\quad\mathcal{L}=\mathrm{BCEWithLogits}(z,y),\quad W_o=b_o=0\ \mathrm{no\ inicio}",
        "Margem combinada = margem do GLM mais correção neural; perda BCE usa essa soma; pesos e viés da cabeça começam em zero",
        "No dia a dia: revisar um orçamento já calculado",
        [
            "O orçamento inicial já traz uma primeira estimativa",
            "O revisor começa sem alterar nada e aprende onde ajustar",
            "A avaliação compara o total corrigido com o resultado, não só o ajuste",
        ],
        "Imagine revisar um orçamento já calculado. O revisor começa "
        "sem mudar o total e aprende onde acrescentar ou retirar "
        "pequenas quantias. A avaliação considera o orçamento completo "
        "depois das correções. Escolher como revisar num conjunto "
        "interno e depois recomeçar com a regra escolhida ajuda a "
        "preservar a avaliação externa. Para entrar na composição, "
        "a experiência precisaria demonstrar um ganho suficiente.",
    )


def append_transductive_pair(pair, runs):
    run = next((run for run in complete(runs) if has_transductive(run)), None)
    if not run:
        return
    pair(
        "04 / Representar os dados",
        "Experimento: contar entradas conhecidas do teste",
        [
            "A referência de contagens usa X do treino oficial e X do teste oficial",
            "Nenhum rótulo do teste existe; os rótulos externos não entram nas contagens",
            "TE, pooling supervisionado e escalonamento continuam no treino externo",
            "É um protocolo transdutivo declarado, diferente da frequência só do treino",
        ],
        "Este experimento muda somente a referência das contagens. "
        "As características do conjunto de teste já são conhecidas na "
        "competição. Podemos contar ocorrências no treino oficial e no "
        "teste oficial, sem consultar rótulos. Isso também inclui as "
        "características das linhas de validação, por isso chamamos o "
        "procedimento de transdutivo e não o descrevemos como ajuste "
        "exclusivo no treino do fold. Target encoding, taxas supervisionadas "
        "do pooling e scalers continuam aprendidos no treino externo. "
        "A hipótese depende dessa disponibilidade prévia das entradas; "
        "não é uma garantia para transações futuras ainda desconhecidas.",
        r"n_j(v)=\sum_{i\in T}\mathbf{1}[x_{ij}=v]+\sum_{i\in U}\mathbf{1}[x_{ij}=v],\quad f_j(v)=\log(1+n_j(v))",
        "Contagem considera as entradas conhecidas dos dois conjuntos, sem alvos; a entrada ao modelo é log de um mais a contagem",
        "No dia a dia: contar itens numa lista já recebida",
        [
            "Você recebeu a lista completa de peças de uma encomenda",
            "Pode contar modelos repetidos sem saber quais peças têm defeito",
            "Essa informação não estava disponível para uma encomenda futura",
        ],
        "Imagine receber a lista completa de peças de uma encomenda. "
        "Você pode contar quantas peças de cada modelo aparecem, sem "
        "saber quais têm defeito. Essa informação descreve a lista, "
        "não revela as respostas. Mas a mesma vantagem não existe "
        "para uma encomenda que ainda não chegou. É por isso que "
        "registramos claramente o protocolo de contagem.",
    )


def append_storage_pair(pair, runs, evidence):
    if not any(
        run.get("model") != "blend"
        and "strings.py" in run.get("source_hashes", {})
        and run.get("model_details", {}).get(
            "feature_set", run.get("run_config", {}).get("feature_set")
        )
        in {
            "digits",
            "value_digits",
            "paired_digits",
            "bpe_digits",
            "token_digits",
            "joint_bpe",
        }
        for run in selected_runs(runs)
    ):
        return
    probe = evidence.get("presentation_context", {}).get("string_storage_probe", {})
    measured = (
        probe.get("values_exact") is True
        and probe.get("process_rss_measured") is False
        and all(
            isinstance(probe.get(key), int)
            for key in ("rows", "unique_values", "legacy_bytes", "shared_bytes")
        )
    )
    point = (
        f"Teste sintético: {decimal(probe['legacy_bytes'] / 1e6)} → {decimal(probe['shared_bytes'] / 1e6)} MB de buffer + strings únicas; não é RSS"
        if measured
        else "A prova mede buffer e objetos de strings; não mede a RAM total do processo"
    )
    pair(
        "04 / Construir sem mudar valores",
        "Memória: guardar cada texto repetido uma vez",
        [
            "unique encontra valores distintos; inverse recompõe a ordem original",
            "Cada texto inteiro/Unicode distinto é convertido uma vez e compartilhado",
            "Valores, dtype object, ordem, índices e previsões são preservados",
            point,
        ],
        "Muitas linhas carregam o mesmo texto de uma categoria. A "
        "otimização encontra valores distintos, converte cada texto "
        "uma vez e usa índices inversos para recompor a sequência. "
        "As posições repetidas apontam para o mesmo objeto de string, "
        "em vez de criar cópias iguais. O resultado mantém os valores, "
        "a ordem e o tipo object. Não é um novo atributo, um vocabulário "
        "aprendido ou uma explicação para ganho de AUC. "
        + (
            f"Num teste fabricado de {probe['rows']} posições e {probe['unique_values']} valores, "
            f"a soma do buffer com objetos de strings distintos passou de {decimal(probe['legacy_bytes'] / 1e6)} "
            f"para {decimal(probe['shared_bytes'] / 1e6)} megabytes. "
            if measured
            else ""
        )
        + "Essa contagem exclui temporários e outras estruturas. Não mede "
        "RSS, pico total de RAM ou velocidade do treinamento real.",
        r"v_i=u_{\iota_i},\quad s_i=\mathrm{str}(u_{\iota_i})",
        "Cada valor original é recuperado pelo índice inverso nos valores únicos; o texto correspondente é compartilhado",
        "No dia a dia: fichas apontando para o mesmo verbete",
        [
            "Um dicionário guarda uma definição e várias fichas podem apontar para ela",
            "As fichas continuam na mesma ordem e mantêm o mesmo significado",
            "Economizar cópias não muda a resposta da ficha",
        ],
        "Pense num dicionário. Em vez de copiar a mesma definição "
        "em milhares de fichas, cada ficha pode apontar para um "
        "único verbete. A ordem e o significado das fichas continuam "
        "iguais. Economizamos cópias, não mudamos as informações "
        "usadas para decidir. A contagem de verbetes e fichas "
        "também não representa toda a memória de uma biblioteca.",
    )


def append_nested_stack_pair(pair, runs):
    run = next(
        (
            run
            for run in complete(runs)
            if run.get("model_details", {}).get("algorithm")
            == "nested_owned_logit_holdout_stack"
        ),
        None,
    )
    if not run:
        return
    details = run["model_details"]
    if (
        details.get("inner_fraction") != 0.25
        or details.get("inner_seed") != 2026
        or len(details.get("base_runs", [])) != 3
        or details.get("architecture", {}).get("external_teacher_in_meta") is not False
    ):
        raise ValueError(
            "Stacking executado sem evidência do protocolo interno 75/25 e professor externo separado."
        )
    meta = details.get("architecture", {}).get("meta", {})
    meta_parameters = (
        f" Parâmetros registrados: C igual a {decimal(meta['C'])}, solver {meta['solver']}."
        if meta.get("C") is not None and meta.get("solver")
        else ""
    )
    pair(
        "06 / Experimento de composição",
        "Stacking: aprender a combinar sem ver o fold externo",
        [
            "Três logits de bases próprias são entradas meta, não três colunas originais",
            "Bases internas veem 60% dos rótulos; bases na inferência veem 80%; diferença declarada",
            "OOF usa só o fold correspondente; teste faz média de cinco probabilidades após sigmoid",
            "GPT-2 de Masaya fica fora desse meta-treino; entra depois no híbrido 75/25",
        ],
        "Uma média fixa escolhe pesos antes da avaliação. O stacking "
        "treina outro modelo para combinar as previsões. Nesta experiência, "
        "o treino externo é separado novamente. Três bases próprias novas "
        "aprendem nos setenta e cinco por cento internos e produzem "
        "previsões para os vinte e cinco por cento restantes. "
        "São três logits como entradas do meta-modelo, não três colunas "
        "originais nem a dimensão codificada das bases. "
        "As probabilidades extremas são limitadas antes da transformação "
        "logit, evitando infinitos. Esses logits são entradas para uma "
        "regressão logística, que aprende os coeficientes e o intercepto. "
        + meta_parameters
        + " Os alvos "
        "do fold externo ficam fora de todos esses ajustes. Na aplicação "
        "externa, usamos as bases do respectivo fold já treinadas no "
        "treino externo completo. Isso significa sessenta por cento dos "
        "rótulos totais para as bases internas e oitenta por cento para "
        "as bases de inferência. Essa diferença de distribuição é uma "
        "limitação conhecida; não fizemos uma calibração depois do resultado. "
        "Na OOF, cada linha recebe somente a previsão do seu fold externo. "
        "No teste, cada um dos cinco meta-modelos aplica sua própria sigmoid. "
        "Depois calculamos a média dessas cinco probabilidades; não aplicamos "
        "sigmoid à média das margens. Os coeficientes do meta-modelo não "
        "precisam ser positivos nem somar um; não são percentuais de "
        "probabilidade. Só depois combinamos setenta e cinco por cento "
        "da previsão própria com vinte e cinco por cento do GPT-2 "
        "externo auditado. Ele não participou do meta-treino.",
        r"p_k=\sigma\left(b_k+\sum_j a_{jk}\,\mathrm{logit}(p_{jk})\right),\quad\bar p_{\mathrm{teste}}=\frac{1}{5}\sum_{k=1}^{5}p_k,\quad p=0.75\bar p_{\mathrm{teste}}+0.25p_{\mathrm{Masaya}}",
        "Cada fold aplica sua sigmoid; no teste a média é feita sobre cinco probabilidades; a OOF usa apenas o fold da linha; o híbrido combina probabilidades próprias e externas depois",
        "No dia a dia: um supervisor aprende com três revisores",
        [
            "Três revisores praticam em exemplos separados dos usados pelo supervisor",
            "O supervisor aprende como combinar suas estimativas",
            "No teste, cinco equipes dão notas; fazemos a média das cinco notas prontas",
            "A prova externa continua reservada; uma opinião externa só entra depois",
        ],
        "Imagine três revisores e um supervisor. Os revisores "
        "praticam num conjunto, e o supervisor aprende a combinar "
        "suas estimativas em outro. A prova final fica de fora "
        "dessas duas etapas. Repetimos a organização em cinco equipes. "
        "Para um novo recibo, cada equipe dá sua nota pronta e fazemos "
        "a média dessas notas. No ensaio fora do treino, o recibo é avaliado "
        "somente pela equipe que não o usou. Uma opinião externa pode ser combinada "
        "depois, sem ter ensinado o supervisor. Separar essas funções "
        "ajuda a entender o protocolo; não garante que a composição "
        "aprendida seja melhor que a média fixa.",
    )


def append_nested_result(add, runs, evidence, number):
    own = next(
        (
            run
            for run in complete(runs)
            if run.get("model") == "nested_stack"
            and run.get("model_details", {}).get("algorithm")
            == "nested_owned_logit_holdout_stack"
        ),
        None,
    )
    if not own:
        return
    hybrid = next(
        (
            run
            for run in complete(runs)
            if run.get("model") == "blend"
            and name(own)
            in [
                str(path).replace("\\", "/").split("/")[-1]
                for path in run.get("model_details", {}).get("sources", [])
            ]
        ),
        None,
    )
    reference = next(
        (
            run
            for run in complete(runs)
            if name(run) == "round18-pooled-share100-public25"
        ),
        None,
    )
    if not hybrid or not reference:
        return
    folds = list(zip(hybrid.get("fold_scores", []), reference.get("fold_scores", [])))
    points = [
        f"Própria: OOF {number(own['roc_auc'], 8)}; híbrida: {number(hybrid['roc_auc'], 8)}",
        f"Contra R18: Δ {number(hybrid['roc_auc'] - reference['roc_auc'], 8)}; melhora em {sum(a > b for a, b in folds)}/{len(folds)} folds",
        "Ganho local pequeno; validação exploratória reutilizada, sem nova reserva independente",
    ]
    record = next(
        (
            item
            for item in evidence.get("historical_submissions", [])
            if item.get("run") == name(hybrid)
            and item.get("submission_file_verified") is True
            and isinstance(item.get("submission_sha256"), str)
            and re.fullmatch(r"[a-f0-9]{64}", item["submission_sha256"])
            and item.get("submission_sha256") == hybrid.get("submission_sha256")
        ),
        None,
    )
    narration = (
        "O stack e a mistura foram executados com o protocolo definido. "
        + ". ".join(points)
        + ". "
    )
    if record:
        points.append(
            f"Envio {record['submission_id']}: AUC pública {number(record['public_auc'], 5)}; "
            + str(record.get("note", "Resultado vinculado ao seu próprio CSV"))
        )
        narration += (
            f"A submissão {record['submission_id']} recebeu AUC pública {number(record['public_auc'], 5)}. "
            + str(record.get("note", ""))
            + " Não atribuímos a este teste a posição do melhor arquivo da equipe."
        )
    add(
        "07 / Os experimentos",
        "R21: pequeno ganho local, avaliação pública separada",
        points,
        narration,
    )


ROUND_LABELS = {
    12: "CatBoost nativo",
    13: "pooling",
    14: "tokens completos",
    15: "escalas do blend",
    16: "original adicional",
    17: "rede residual",
    18: "oito interações",
    19: "frequência transdutiva",
}


def round_number(run):
    match = re.match(r"round(1[2-9])(?:-|$)", name(run))
    return int(match[1]) if match else None


def append_later_results(add, runs, number):
    baseline = next(
        (
            run
            for run in complete(runs)
            if name(run) == "round10-offset-bpe-frequency-public25"
        ),
        None,
    )
    if not baseline:
        return
    rows = []
    for index, label in ROUND_LABELS.items():
        candidates = [
            run
            for run in complete(runs)
            if round_number(run) == index
            and run.get("model") == "blend"
            and isinstance(run.get("roc_auc"), (int, float))
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda run: run["roc_auc"])
        delta = best["roc_auc"] - baseline["roc_auc"]
        folds = list(zip(best.get("fold_scores", []), baseline.get("fold_scores", [])))
        improved = sum(a > b for a, b in folds)
        passes = delta > 0.0001 and len(folds) == 5 and improved >= 4
        own_candidates = [
            run
            for run in complete(runs)
            if round_number(run) == index
            and run.get("model") != "blend"
            and isinstance(run.get("roc_auc"), (int, float))
        ]
        own_score = (
            f"própria {number(max(run['roc_auc'] for run in own_candidates), 5)}; "
            if own_candidates
            else ""
        )
        rows.append(
            f"R{index} {label}: "
            + own_score
            + f"mistura {number(best['roc_auc'], 5)}; Δ {number(delta, 5)}; {improved}/5 folds; "
            + ("passou gate R10" if passes else "não passou gate R10")
        )
    for start in range(0, len(rows), 3):
        explanation = (
            "O gate contra a R10 exige ganho maior que zero vírgula zero "
            "zero zero um e melhora em pelo menos quatro dos cinco folds. "
            "Passar esse gate também não autoriza substituir uma candidata "
            "já selecionada com OOF maior. Esses resultados usam a mesma "
            "validação exploratória; não são uma nova reserva independente."
            if start == 0
            else "As regras e a limitação de validação continuam as mesmas."
        )
        add(
            "07 / Os experimentos",
            "Continuação: resultados preservados, inclusive negativos",
            rows[start : start + 3],
            "Cada linha mostra a melhor mistura fornecida daquela rodada, "
            "entre as comparações predefinidas. "
            + ". ".join(rows[start : start + 3])
            + ". "
            + explanation,
        )
    own18 = next(
        (
            run
            for run in complete(runs)
            if round_number(run) == 18 and run.get("model") == "offset"
        ),
        None,
    )
    mixtures18 = [
        run
        for run in complete(runs)
        if round_number(run) == 18 and run.get("model") == "blend"
    ]
    if own18 and mixtures18:
        best18 = max(mixtures18, key=lambda run: run["roc_auc"])
        folds = list(zip(best18["fold_scores"], baseline["fold_scores"]))
        own13 = next(
            (run for run in complete(runs) if name(run) == "round13-pooled-offset"),
            None,
        )
        comparison = (
            f"Adicionar juntas à R13: Δ próprio {number(own18['roc_auc'] - own13['roc_auc'], 8)}; "
            f"{number(own13['model_details'].get('encoded_features'))} → {number(own18['model_details'].get('encoded_features'))} entradas"
            if own13
            else "A ablação das juntas usa R13; a promoção usa R10 como referência"
        )
        add(
            "07 / Os experimentos",
            "R18: separar a ablação da regra de promoção",
            [
                f"Offset próprio: AUC {number(own18['roc_auc'], 8)}; {own18['features']['count']} colunas → {own18['model_details']['encoded_features']} entradas",
                comparison,
                f"Híbrida R18: {number(best18['roc_auc'], 8)}; R10: {number(baseline['roc_auc'], 8)}; Δ {number(best18['roc_auc'] - baseline['roc_auc'], 8)}",
                f"Melhora em {sum(a > b for a, b in folds)}/{len(folds)} folds contra R10; inclui pooling e juntas; OOF exploratória",
            ],
            "A R18 acrescentou as oito interações à base com BPE, frequência "
            "e pooling, preservando os parâmetros definidos. O componente "
            f"próprio alcançou AUC {number(own18['roc_auc'], 8)}. "
            f"São {own18['features']['count']} colunas antes do pré-processamento "
            f"e {own18['model_details']['encoded_features']} entradas na matriz usada "
            "pelo GLM e pelas árvores residuais. Colunas originais, colunas "
            "criadas e dimensões codificadas são quantidades diferentes. "
            + (
                f"A comparação direta com a R13 dá diferença própria de {number(own18['roc_auc'] - own13['roc_auc'], 8)}. "
                if own13
                else ""
            )
            + "A comparação da híbrida com a R10 usa os mesmos folds e "
            "rótulos. É a evidência local da mudança, ainda usada para "
            "seleção. Como a R10 não tinha pooling, esse ganho da híbrida "
            "inclui pooling e juntas e não isola apenas as interações. "
            "Score e posição públicos têm uma prova separada.",
        )
    original = next(
        (
            run
            for run in complete(runs)
            if round_number(run) == 16 and run.get("additional_training_data")
        ),
        None,
    )
    neural = next(
        (
            run
            for run in complete(runs)
            if run.get("model") == "neural_offset" and run.get("recovery")
        ),
        None,
    )
    if original or neural:
        points = []
        if original:
            data = original["additional_training_data"]
            points.append(
                f"R16: {number(data['rows'])} originais {data.get('license') or 'licença não informada'} apenas no treino externo; OOF própria {number(original['roc_auc'], 5)}"
            )
        if neural:
            points.append(
                f"R17: OOF própria {number(neural['roc_auc'], 5)}; três checkpoints completos reutilizados, dois folds treinados na retomada"
            )
            points.append(
                f"Retomada: {number(neural['recovery']['resume_elapsed_seconds'], 1)} s; tempo total original não foi persistido"
            )
        narration = (
            "Os exemplos originais foram acrescentados apenas ao treino externo da R16; "
            "não receberam posições na OOF da competição. Essa mistura não foi promovida. "
            if original
            else ""
        )
        narration += (
            "A rede residual da R17 sofreu falta de memória depois de três folds completos. "
            "A recuperação preservou esses pesos e treinou somente os dois folds restantes, "
            "com a mesma fonte e configuração. O tempo de recuperação não é o tempo total "
            "original, que não foi persistido."
            if neural
            else ""
        )
        add(
            "07 / Os experimentos",
            "Dados adicionais e recuperação também precisaram de prova",
            points,
            narration,
        )


def append_context(add, evidence):
    context = evidence.get("presentation_context", {})
    pending = context.get("pending_rounds", [])
    if pending:
        add(
            "07 / Os experimentos",
            "O que ainda não tem resultado final neste snapshot",
            [
                f"R{item['round']}: {item['status']} — {item['note_pt']}"
                for item in pending
            ],
            "Estes estados foram fornecidos para este snapshot. "
            + ". ".join(
                f"Rodada {item['round']}: {item['status']}. {item['note_pt']}"
                for item in pending
            )
            + ". Planejamento, execução em andamento e resultado verificado "
            "são estados diferentes. Não atribuímos score ou ranking "
            "a um arquivo ainda não submetido e processado.",
        )
    tests = context.get("tests", {})
    if isinstance(tests.get("passed"), int) and tests.get("scope"):
        counts = f"Testes aprovados: {tests['passed']}"
        if isinstance(tests.get("subtests"), int):
            counts += f"; subtestes: {tests['subtests']}"
        add(
            "08 / Da execução à competição",
            "Verificação de software no snapshot",
            [
                counts + f"; escopo: {tests['scope']}",
                "A suíte verifica contratos e comportamento; não comprova vitória na competição",
                "A QA audiovisual final tem sua própria prova e seus próprios limites",
            ],
            "O número de testes pertence ao snapshot e ao escopo "
            "registrados. Eles verificam software, como contratos de "
            "dados, persistência, fórmulas e publicação. Um teste "
            "aprovado não demonstra uma boa colocação. A validação "
            "do vídeo também é separada: transcrição automática e "
            "quadros examinados não equivalem a escuta humana ou "
            "reprodução contínua integral.",
        )
