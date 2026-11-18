"""Conditional RealMLP teaching content from completed, resolved model records."""

import math
import re
import statistics

OFFICIAL = "https://github.com/dholzmueller/pytabkit/blob/d232eecd47b127b29da0378423cb475b88fb0e80/"


def decimal(value):
    return f"{value:g}".replace(".", ",")


def resolved_realmlp(runs):
    run = next(
        (
            run
            for run in runs
            if run.get("status") == "complete"
            and run.get("model") == "realmlp"
            and run.get("phase") != "full_data_refit"
        ),
        None,
    )
    if run is None:
        return None
    details = run.get("model_details", {})
    architecture = details.get("architecture", {})
    parameters = details.get("refit", {}).get("resolved_parameters", {})
    preprocessing = details.get("preprocessing", {})
    expected = {
        "num_emb_type": "pbld",
        "plr_act_name": "linear",
        "plr_use_densenet": True,
        "plr_use_cos_bias": True,
        "use_parametric_act": True,
        "act": "silu",
        "ens_av_before_softmax": False,
        "opt": "adam",
        "use_ls": True,
        "use_ls_prior": False,
        "use_plr_embeddings": True,
        "lr_sched": "flat_anneal",
        "wd_sched": "cos_log_15",
        "p_drop_sched": "pow5",
        "ls_eps_sched": "sqrt_cos",
    }
    required = (
        "plr_hidden_1",
        "plr_hidden_2",
        "n_ens",
        "hidden_sizes",
        "n_epochs",
        "embedding_size",
        "max_one_hot_cat_size",
        "lr",
        "wd",
        "p_drop",
        "ls_eps",
        "sq_mom",
        "smooth_clip_max_abs_value",
        "robust_scale_eps",
        "l2_normalize_eps",
        "tfms",
    )
    valid = (
        architecture.get("library") == "pytabkit==1.8.0"
        and all(parameters.get(key) == value for key, value in expected.items())
        and all(key in parameters for key in required)
        and parameters.get("tfms")
        == [
            "one_hot",
            "median_center",
            "robust_scale",
            "smooth_clip",
            "embedding",
            "l2_normalize",
        ]
        and details.get("config", {}).get("inner_fraction") == 0.15
        and details.get("config", {}).get("inner_seed") == 42
        and architecture.get("n_ens") == parameters.get("n_ens")
        and architecture.get("hidden_dims") == parameters.get("hidden_sizes")
        and isinstance(run.get("roc_auc"), (int, float))
        and not isinstance(run.get("roc_auc"), bool)
        and math.isfinite(run["roc_auc"])
        and 0 <= run["roc_auc"] <= 1
        and len(preprocessing.get("columns", []))
        == preprocessing.get("numeric_features", -1)
        + preprocessing.get("categorical_features", -1)
    )
    if not valid:
        raise ValueError(
            "RealMLP completa sem configuração resolvida compatível com PBLD, SiLU parametrizada e treinamento registrado."
        )
    folds = details.get("folds") or [details]
    if not folds or any(
        fold.get("refit", {}).get("resolved_parameters") != parameters
        or not isinstance(fold.get("selected_epoch"), int)
        or not 1 <= fold["selected_epoch"] <= parameters["n_epochs"]
        or fold.get("refit", {}).get("completed_epochs") != fold["selected_epoch"]
        for fold in folds
    ):
        raise ValueError(
            "RealMLP sem épocas selecionadas e refits concluídos registrados."
        )
    return run, details, parameters, preprocessing, folds


def append_realmlp_result(add, runs, evidence, number):
    record = resolved_realmlp(runs)
    if record is None:
        return
    own, details, p, prep, folds = record
    own_name = str(own.get("run_name", ""))
    candidates = [
        run
        for run in runs
        if run.get("status") == "complete"
        and run.get("model") == "blend"
        and run.get("phase") != "full_data_refit"
        and own_name
        in [
            str(path).replace("\\", "/").split("/")[-1]
            for path in run.get("model_details", {}).get("sources", [])
        ]
    ]
    if not own_name or not candidates:
        return
    best = max(candidates, key=lambda run: run["roc_auc"])
    by_name = {str(run.get("run_name", "")): run for run in runs}
    percentages = []
    for candidate in candidates:
        source_names = [
            str(path).replace("\\", "/").split("/")[-1]
            for path in candidate["model_details"]["sources"]
        ]
        weights = candidate["model_details"]["weights"]
        neural_weight = weights[source_names.index(own_name)]
        offset_weight = sum(
            weight
            for source, weight in zip(source_names, weights)
            if by_name.get(source, {}).get("model") == "offset"
        )
        if neural_weight + offset_weight <= 0:
            raise ValueError(
                "Mistura RealMLP sem parcela própria de offset e rede registrada."
            )
        fraction = neural_weight / (neural_weight + offset_weight)
        percentages.append(fraction * 100)
    reference = next(
        (
            run
            for run in runs
            if run.get("run_name") == "round18-pooled-share100-public25"
        ),
        None,
    )
    delta = (
        f"; Δ R18 {number(best['roc_auc'] - reference['roc_auc'], 8)}; "
        f"{sum(a > b for a, b in zip(best['fold_scores'], reference['fold_scores']))}/5 folds"
        if reference
        else ""
    )
    epochs = [fold["selected_epoch"] for fold in folds]
    grade = (
        "Grade fornecida: substituir "
        + ", ".join(number(value) + "%" for value in sorted(percentages))
        + " da parcela do offset"
    )
    add(
        "07 / Os experimentos",
        "R22: execução neural e grade de combinações",
        [
            f"RealMLP própria: OOF {number(own['roc_auc'], 8)}; E por fold {epochs}",
            grade,
            f"{len(folds)} folds × {p['n_ens']} redes = {len(folds) * p['n_ens']} membros finais de CV; nenhum refit único em todos os rótulos",
            "A grade preserva as parcelas do LightGBM e do GPT-2 externo",
        ],
        f"A RealMLP própria alcançou AUC OOF {number(own['roc_auc'], 8)}. "
        f"Foram registrados {len(folds)} refits de fold, com {p['n_ens']} membros por fold, "
        f"totalizando {len(folds) * p['n_ens']} membros finais nesse protocolo de CV. "
        "Isso não descreve um refit único em todos os rótulos. " + grade + ". "
        "A grade altera a parte própria do offset e preserva as parcelas do LightGBM e do GPT-2 externo. "
        + (
            f"O tempo de parede registrado foi {number(own['elapsed_seconds'], 2)} segundos. "
            "É uma duração observada dessa execução, sem benchmark controlado de velocidade. "
            if isinstance(own.get("elapsed_seconds"), (int, float))
            else ""
        ),
    )
    labels = {
        "offset": "offset",
        "realmlp": "RealMLP",
        "lightgbm": "LightGBM",
        "external_gpt2": "GPT-2 externo",
    }
    best_sources = [
        str(path).replace("\\", "/").split("/")[-1]
        for path in best["model_details"]["sources"]
    ]
    weights = "; ".join(
        f"{number(weight * 100, 4)}% {labels.get(by_name.get(source, {}).get('model'), source)}"
        for source, weight in zip(best_sources, best["model_details"]["weights"])
    )
    points = [
        f"Melhor mistura local: OOF {number(best['roc_auc'], 8)}" + delta,
        "Pesos efetivos: " + weights,
        "A validação exploratória reutilizada não é uma nova reserva independente",
    ]
    history = next(
        (
            item
            for item in evidence.get("historical_submissions", [])
            if item.get("run") == best.get("run_name")
            and item.get("submission_file_verified") is True
            and isinstance(item.get("submission_sha256"), str)
            and re.fullmatch(r"[a-f0-9]{64}", item["submission_sha256"])
            and item["submission_sha256"] == best.get("submission_sha256")
        ),
        None,
    )
    if history:
        points.append(
            f"Envio {history['submission_id']}: AUC pública {number(history['public_auc'], 5)}; "
            + str(history.get("note", "Resultado vinculado ao seu CSV"))
        )
    narration = (
        ". ".join(points)
        + ". O ganho é exploratório, medido nos folds reutilizados. "
        + (
            "O empate na precisão pública exibida, registrado nessa consulta, não prova melhora "
            "numa precisão oculta. Não atribuímos posição individual a essa tentativa. "
            if history and "empate" in str(history.get("note", "")).lower()
            else ""
        )
    )
    add(
        "07 / Os experimentos",
        "R22: ganho local e avaliação pública separados",
        points,
        narration,
    )


def full_realmlp_record(selected_members, runs):
    full = next((run for run in selected_members if run.get("status") == "complete"
                 and run.get("model") == "realmlp" and run.get("phase") == "full_data_refit"), None)
    if full is None:
        return None
    cv = resolved_realmlp(runs)
    if cv is None:
        raise ValueError("Refit RealMLP exige a arquitetura CV própria completa.")
    _, cv_details, parameters, _, folds = cv
    epoch = statistics.median(fold["selected_epoch"] for fold in folds)
    details = full.get("model_details", {})
    architecture = details.get("architecture", {})
    if (full.get("roc_auc") is not None or full.get("fold_scores") != [] or full.get("rows", {}).get("oof") != 0
            or not isinstance(details.get("refit_rows"), int) or details["refit_rows"] <= 0
            or details["refit_rows"] != full.get("rows", {}).get("train")
            or details.get("validation_rows") != 0 or details.get("epochs") != epoch
            or details.get("refit", {}).get("completed_epochs") != epoch
            or architecture.get("library") != cv_details["architecture"]["library"]
            or architecture.get("hidden_dims") != parameters["hidden_sizes"]
            or architecture.get("n_ens") != parameters["n_ens"]):
        raise ValueError("Refit RealMLP sem treino integral, mediana de épocas e ausência de nova OOF comprovados.")
    return full, int(epoch), architecture["n_ens"]


def append_realmlp_pairs(pair, runs):
    record = resolved_realmlp(runs)
    if record is None:
        return
    run, details, p, prep, folds = record
    chapter = "06 / Experimento neural"
    total = len(prep["columns"])
    limit = p["smooth_clip_max_abs_value"]
    pair(
        chapter,
        "RealMLP: preparar escalas sem consultar a prova",
        [
            f"{total} representações: {prep['numeric_features']} numéricas e {prep['categorical_features']} categóricas",
            "Medianas, cortes por quantis, vocabulários e TE são aprendidos no treino de cada etapa",
            f"Escala robusta por IQR; compressão suave com limite {decimal(limit)}; normalização RMS por coluna",
            "Coluna constante vira zero; IQR zero usa metade da amplitude antes desse fallback",
        ],
        "A RealMLP usa outra representação dos mesmos dados oficiais. As colunas originais "
        "ganham partes decimais, categorias dos inteiros, faixas por quantis, duas combinações "
        "de campos e três codificações de alvo cross-fitted. "
        f"O registro tem {total} representações, sendo {prep['numeric_features']} numéricas e "
        f"{prep['categorical_features']} categóricas. Os mecanismos de dígitos, quantis e target encoding "
        "já foram explicados; aqui mudam a seleção e o encadeamento. Nas entradas contínuas, "
        "a biblioteca subtrai a mediana e divide pela distância entre os quartis. Se essa distância "
        "for zero, usa metade da amplitude; uma coluna constante fica zero. Depois comprime "
        "suavemente valores extremos e ajusta a escala pela raiz da média dos quadrados de cada "
        "coluna no treino. Isso não é normalizar cada transação pela sua própria norma. "
        "As estatísticas são reaprendidas dentro da etapa permitida, sem alvos da validação externa.",
        r"u=\frac{x-m}{s+\epsilon},\quad v=\frac{u}{\sqrt{1+(u/c)^2}},\quad r=\frac{v}{\sqrt{\mathrm{mean}_{T}(v^2)+\epsilon_2}}",
        "Subtrair a mediana e dividir pela escala robusta; comprimir extremos; normalizar a escala quadrática por coluna usando o treino, com os fallbacks descritos",
        "No dia a dia: ajustar a escala de várias réguas",
        [
            "Uma régua mede em centímetros e outra em metros",
            "Usamos referências do material de treino para compará-las",
            "Valores extremos são comprimidos gradualmente, sem um corte brusco",
        ],
        "Imagine comparar medidas de várias réguas. Primeiro escolhemos uma referência "
        "central e uma escala para cada régua. Depois, uma marca muito distante é aproximada "
        "de forma gradual, em vez de ser cortada de repente. Essas referências vêm do material "
        "de treino. Não usamos as respostas da prova para decidir como ajustar as réguas.",
    )
    periodic, size = p["plr_hidden_1"], p["plr_hidden_2"]
    pair(
        chapter,
        "RealMLP: ondas aprendidas para entradas numéricas",
        [
            f"PBLD: {periodic} cossenos com frequências e fases aprendidas por entrada",
            f"Projeção linear para {size - 1} coordenadas + valor original = {size} coordenadas no total",
            "Nesta variante, a projeção é linear; não há GELU ou ReLU nesse ponto",
            "A codificação anterior entra nas ondas; os pesos são aprendidos no treino",
        ],
        "O embedding PBLD representa cada valor preparado com ondas. A frequência controla "
        "a rapidez da oscilação, e a fase desloca essa onda. Ambas são aprendidas. "
        f"A configuração registrada usa {periodic} cossenos e uma projeção linear para {size - 1} "
        f"coordenadas. O próprio valor preparado é preservado por uma conexão direta. Assim, são "
        f"{size} coordenadas no total por entrada, incluindo essa conexão. A fórmula usa u para "
        "o valor preparado, não para um identificador categórico. As ondas oferecem padrões "
        "de variação que a rede pode combinar; não demonstram uma regra real de geração dos dados.",
        r"c_j=\cos(2\pi u_j\omega_j+\beta_j),\quad e_j=[W_jc_j+b_j;u_j]\in\mathbb{R}^{"
        + str(size)
        + "}",
        "Cossenos de frequências e fases aprendidas; projeção linear mais o valor original, com dimensão total registrada",
        "No dia a dia: observar ponteiros em ritmos diferentes",
        [
            "Ponteiros que giram em ritmos diferentes oferecem várias perspectivas da mesma medida",
            "Aprendemos quais perspectivas ajudam e como combiná-las",
            "A medida original continua disponível junto dessas perspectivas",
        ],
        "Imagine vários ponteiros ligados à mesma medida, girando em ritmos diferentes. "
        "Juntos, eles oferecem outras maneiras de localizar esse valor. A rede aprende "
        "quais perspectivas usar e como combiná-las. A medida original também fica na ficha. "
        "A analogia ajuda a entender as ondas; não significa que uma transação tenha um relógio oculto.",
    )
    dimensions = " → ".join(map(str, p["hidden_sizes"]))
    pair(
        chapter,
        "RealMLP: a própria ativação também aprende",
        [
            f"Camadas ocultas: {dimensions}; saída com dois logits, um por classe",
            "SiLU parametrizada: mistura a identidade com SiLU por coeficiente aprendido",
            f"Categorias menores: one-hot até {p['max_one_hot_cat_size']}; maiores: embeddings de dimensão {p['embedding_size']}",
            f"{p['n_ens']} redes completas vetorizadas; seus pesos não são compartilhados como no TabM",
        ],
        "As transformações lineares, os vieses e os embeddings categóricos seguem os conceitos "
        f"já apresentados. A configuração executada tem camadas ocultas {dimensions.replace(' → ', ', ')} "
        "e produz dois logits, um para cada classe. A novidade da ativação é um coeficiente "
        "aprendido, alfa, que mistura a identidade com a SiLU. Quando alfa vale um, temos a SiLU; "
        "quando vale zero, temos a identidade. Esse coeficiente não é uma probabilidade e não "
        "afirmamos que ficará limitado ao intervalo de zero a um. "
        f"O ensemble contém {p['n_ens']} redes completas com parâmetros próprios, executadas de forma "
        "vetorizada. Isso é diferente do backbone compartilhado do TabM.",
        r"a_\alpha(u)=u+\alpha\,[u\sigma(u)-u]",
        "Ativação = identidade mais coeficiente aprendido vezes a diferença entre SiLU e identidade",
        "No dia a dia: ajustar a intensidade de uma receita",
        [
            "Uma receita pode preservar mais do sabor original ou aplicar uma transformação maior",
            "O ajuste da intensidade também é aprendido com exemplos",
            "Vários cozinheiros mantêm suas próprias receitas e parâmetros",
        ],
        "Pense num ajuste de intensidade em uma receita. Podemos preservar mais do sabor "
        "original ou aplicar uma transformação maior. Aqui esse ajuste também é aprendido. "
        "Os cozinheiros mantêm receitas próprias; trabalhar ao mesmo tempo não faz com que "
        "todos usem os mesmos parâmetros.",
    )
    pair(
        chapter,
        "RealMLP: combinar probabilidades das redes completas",
        [
            "Softmax converte os dois logits de cada membro em probabilidades",
            f"A média usa {p['n_ens']} probabilidades da classe fraude, depois da softmax",
            "No teste, o protocolo externo também combina probabilidades dos folds",
            "OOF usa só o fold da linha; o número de membros não cria validações independentes",
        ],
        "Cada rede produz dois logits. A softmax transforma esse par em probabilidades que "
        "somam um. Depois calculamos a média das probabilidades da classe fraude. Não tiramos "
        "a média dos logits para aplicar uma única softmax depois. A biblioteca devolve o logaritmo "
        "da média, e nosso wrapper recupera as probabilidades dessa representação. "
        "A média entre membros ocorre dentro de cada fold. A média entre folds ocorre depois, "
        "para o teste; cada linha OOF continua usando somente seu fold externo. Os membros não "
        "transformam a validação reutilizada em uma reserva independente.",
        r"p(x)=\frac{1}{M}\sum_{m=1}^{M}\frac{e^{z_{m,1}}}{e^{z_{m,0}}+e^{z_{m,1}}},\quad M="
        + str(p["n_ens"]),
        "Média das probabilidades de fraude obtidas pela softmax de cada um dos M membros",
        "No dia a dia: tirar a média de avaliações prontas",
        [
            "Cada avaliador termina sua própria análise e entrega uma nota",
            "A média usa essas notas prontas, não os cálculos intermediários",
            "Mais avaliadores não significam mais provas independentes",
        ],
        "Imagine vários avaliadores. Cada um termina sua análise e entrega uma nota. "
        "A média é feita sobre essas notas prontas. Não misturamos os cálculos intermediários "
        "antes de cada um chegar à sua avaliação. E ter mais avaliadores olhando o mesmo "
        "material não cria novas provas independentes.",
    )
    pair(
        chapter,
        "RealMLP: entropia cruzada com alvo suavizado",
        [
            f"Suavização ativa; amplitude-base ε = {decimal(p['ls_eps'])}, controlada pelo schedule registrado",
            "Alvo de duas classes se aproxima levemente da distribuição uniforme",
            "Cada rede calcula a perda média do lote; a implementação soma as perdas dos membros",
            "É entropia cruzada sobre dois logits; não é BCE sobre uma margem única",
        ],
        "Nesta rede usamos entropia cruzada para duas classes. O alvo é convertido em duas "
        "posições e misturado levemente com a distribuição uniforme. A intensidade, épsilon, "
        "varia ao longo do treinamento pelo schedule registrado. Isso muda a perda, sem trocar "
        "os rótulos usados na avaliação de AUC. Para cada rede, calculamos a média das perdas "
        "do lote. O código oficial soma essas médias entre os membros. Não atribuímos a este "
        "treinamento a redução média sobre lote e membros que apareceu no BCE do TabM.",
        r"q_{ic}=(1-\epsilon_t)\mathbf{1}[y_i=c]+\epsilon_t/2,\quad\mathcal{L}=-\sum_m\frac{1}{B}\sum_{i,c}q_{ic}\log\mathrm{softmax}(z_{mi})_c",
        "Alvo suavizado mistura a classe observada com uma distribuição uniforme; perda soma, entre membros, as médias da entropia cruzada do lote",
        "No dia a dia: ensinar sem exigir confiança absoluta",
        [
            "A resposta continua sendo a referência correta",
            "Uma pequena suavização reduz a exigência de confiança extrema no treino",
            "Na prova, a resposta original continua valendo",
        ],
        "Imagine corrigir um exercício e ensinar o aluno a evitar confiança absoluta em "
        "todo exemplo de treino. A resposta continua sendo a referência, mas a cobrança da "
        "perda é levemente suavizada. Na avaliação, usamos a resposta original. Isso não "
        "autoriza mudar os rótulos da prova nem garante ganho de AUC.",
    )
    pair(
        chapter,
        "RealMLP: passo e dropout mudam ao longo do treino",
        [
            f"Adam oficial; taxa-base {decimal(p['lr'])}; segundo momento {decimal(p['sq_mom'])}",
            "flat_anneal: constante nos primeiros 60%; redução cossenoidal depois; δ = 10⁻⁵",
            f"Dropout: {decimal(p['p_drop'])} × (1 − progresso)⁵",
            f"O progresso usa o horizonte de {p['n_epochs']} épocas, inclusive no refit interrompido em E",
        ],
        "O passo do otimizador não permanece constante durante todo o treinamento. O schedule "
        "flat anneal mantém a taxa nos primeiros sessenta por cento do horizonte e a reduz por "
        "um cosseno nos quarenta por cento finais. A taxa termina em um fator de dez elevado "
        "a menos cinco da taxa-base. O dropout também varia: sua probabilidade-base é multiplicada "
        "por um menos o progresso elevado à quinta potência. O progresso é medido no horizonte "
        "original de épocas. Ao refazer o treino por E épocas, esse horizonte não é comprimido "
        "para terminar o schedule mais cedo. Usamos o Adam e o tratamento de decaimento da "
        "biblioteca oficial, sem declarar que seja a mesma chamada AdamW da MLP anterior.",
        r"h(t)=\max(0,(t-0.6)/0.4),\quad\eta_t=\eta_0[\delta+(1-\delta)(1+\cos(\pi h(t)))/2],\quad d_t=d_0(1-t)^5",
        "Progresso t entre zero e um; taxa constante até 60% e depois cosseno; probabilidade de dropout segue potência cinco",
        "No dia a dia: praticar com passos menores no final",
        [
            "No começo, seguimos um ritmo de prática estável",
            "Mais adiante, os ajustes ficam menores e as interrupções diminuem",
            "Usar só uma parte da agenda não transforma essa parte na agenda inteira",
        ],
        "Pense numa agenda de prática. Começamos num ritmo estável e, mais adiante, fazemos "
        "ajustes menores e menos interrupções. Se a prática terminar antes, não fingimos que "
        "toda a agenda foi percorrida. Da mesma forma, E épocas de refit não comprimem "
        "automaticamente o horizonte original dos schedules.",
    )
    pair(
        chapter,
        "RealMLP: regularização e suavização têm agendas próprias",
        [
            f"Decaimento-base {decimal(p['wd'])}; schedule cos_log_15",
            f"Suavização-base {decimal(p['ls_eps'])}; schedule sqrt_cos",
            "As duas intensidades oscilam com o progresso; não ficam fixas nos valores-base",
            "A implementação oficial aplica os fatores registrados de cada grupo de parâmetros",
        ],
        "A força do decaimento e a suavização dos alvos também variam. O schedule cos log "
        "quinze aplica um cosseno ao logaritmo de um mais quinze vezes o progresso. A suavização "
        "usa uma oscilação cossenoidal de cinco ciclos com uma amplitude que diminui pela "
        "raiz do progresso. Por isso os valores-base registrados não são intensidades constantes "
        "durante todas as atualizações. A versão da biblioteca também aplica fatores próprios "
        "a grupos de parâmetros; o código e a versão ficam pinados para preservar essa regra.",
        r"\lambda_t=\lambda_0\frac{1-\cos(2\pi\log_2(1+15t))}{2},\quad\epsilon_t=\epsilon_0\frac{0.05}{\sqrt{t}+0.05}\frac{1-\cos(10\pi t)}{2}",
        "Decaimento e suavização são multiplicados pelas duas funções explícitas do progresso registradas na biblioteca",
        "No dia a dia: controles diferentes numa aula",
        [
            "O ritmo da aula e a intensidade das revisões podem seguir agendas distintas",
            "Um controle não precisa acompanhar o outro a cada instante",
            "As agendas foram definidas antes de observar a prova externa",
        ],
        "Imagine uma aula em que o ritmo de explicação e a intensidade das revisões "
        "seguem agendas diferentes. Ajustar um controle não exige mudar o outro do mesmo "
        "modo. Aqui essas agendas são fórmulas definidas antes de observar o resultado "
        "externo. Elas fazem parte da receita testada; não são uma explicação automática "
        "de eventual ganho.",
    )
    epochs = [fold["selected_epoch"] for fold in folds]
    pair(
        chapter,
        "RealMLP: escolher E por dentro e recomeçar o treino",
        [
            f"Reserva interna de 15%, seed 42; seleção por AUC até {p['n_epochs']} épocas",
            "Pré-processamento, rede e otimizador da seleção são descartados",
            f"Refit novo no treino externo completo; épocas registradas por fold: {epochs}",
            "Rótulos externos ausentes da API; entram na métrica somente após a inferência",
        ],
        "A quantidade de épocas é escolhida numa divisão interna de quinze por cento do "
        "treino externo. A melhor AUC interna escolhe E dentro do limite registrado. "
        "Não tratamos isso como interromper a tentativa inicial e continuar seus pesos. "
        "Descartamos o pré-processamento, a rede e o otimizador e construímos outro conjunto "
        "de objetos. Eles aprendem novamente no treino externo completo por E épocas, "
        "com a semente de refit registrada. A escala de progresso dos schedules continua "
        f"referida ao limite de {p['n_epochs']} épocas. Os rótulos da validação externa não "
        "entram na API deste ajuste nem na escolha de E.",
        r"E^*=\arg\max_{1\leq E\leq E_{\max}}\mathrm{AUC}_{\mathrm{interna}}(E),\quad\theta_{\mathrm{outer}}=\mathrm{TrainFresh}(D_{\mathrm{outer}};E^*)",
        "Escolher a época pela AUC interna e treinar objetos novos no treino externo completo por esse número de épocas",
        "No dia a dia: ensaiar a receita e preparar outra porção",
        [
            "Um pequeno ensaio ajuda a escolher quanto tempo praticar",
            "Depois começamos outra porção, usando a regra escolhida",
            "A prova externa não decide o tempo nem reaproveita a porção do ensaio",
        ],
        "Imagine fazer um pequeno ensaio para escolher o tempo de uma receita. "
        "Depois você prepara outra porção, desde o começo, com a regra escolhida. "
        "Não continua a porção do ensaio nem usa a prova externa para mudar o tempo. "
        "O recomeço deixa claro qual material participou de cada decisão.",
    )


def realmlp_sources(runs):
    if resolved_realmlp(runs) is None:
        return []
    return [
        (
            "RealMLP: Holzmüller, Grinsztajn e Steinwart",
            "https://arxiv.org/abs/2407.04491",
        ),
        ("PBLD: implementação 1.8.0", OFFICIAL + "pytabkit/models/nn_models/nn.py"),
        (
            "SiLU parametrizada: implementação 1.8.0",
            OFFICIAL + "pytabkit/models/nn_models/activations.py",
        ),
        (
            "Ensemble e perda: implementação 1.8.0",
            OFFICIAL + "pytabkit/models/training/lightning_modules.py",
        ),
        (
            "Escalas robustas e RMS: implementação 1.8.0",
            OFFICIAL + "pytabkit/models/nn_models/pipeline.py",
        ),
        (
            "Funções dos schedules: implementação 1.8.0",
            OFFICIAL + "pytabkit/models/training/scheduling.py",
        ),
        (
            "Receita de representação: yekenot, notebook v15",
            "https://www.kaggle.com/code/yekenot/ps-s6-e11-realmlp-pytabkit",
        ),
    ]
