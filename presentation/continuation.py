"""Conditional additions backed by completed runs; no scores are invented here."""


def name(run):
    return str(run.get("run_name", run.get("model", "")))


def leaf(path):
    return str(path).replace("\\", "/").rstrip("/").split("/")[-1]


def selected_runs(runs):
    """Follow blend fractions or explicit stack inputs, never meta-coefficients."""
    if not runs:
        return []
    by_name = {name(run): run for run in runs}
    pending, found = [runs[0]], {}
    while pending:
        run = pending.pop()
        key = name(run)
        if key in found:
            continue
        found[key] = run
        details = run.get("model_details", {})
        if run.get("model") == "nested_stack":
            sources = details.get("base_runs", [])
            weights = [1] * len(sources)  # Membership, not meta-coefficients or probability weights.
        else:
            sources = details.get("sources", [])
            weights = details.get("weights", [1] * len(sources))
        pending.extend(by_name[source] for path, weight in zip(sources, weights)
                       if weight > 0 and (source := leaf(path)) in by_name)
    return [run for run in found.values() if run.get("status") == "complete"]


def has_bpe(runs):
    return any(run.get("model") != "blend" and
               run.get("feature_encoding", {}).get("bpe", run.get("feature_encoding", {})).get("encoding") == "gpt2"
               for run in selected_runs(runs))


def has_frequency(runs):
    return any(run.get("model_details", {}).get("category_frequency") or
               run.get("run_config", {}).get("glm_frequency") is True
               for run in selected_runs(runs))


def append_proposal(add, runs, external_components, number):
    selected = runs[0] if runs else {}
    by_name = {name(run): run for run in runs}
    labels = {"offset": "GLM com correções de árvores", "lightgbm": "LightGBM próprio",
              "external_gpt2": "GPT-2 externo", "mlp": "MLP própria", "tabm": "TabM própria",
              "neural_offset": "rede residual própria", "nested_stack": "stack próprio"}
    details = selected.get("model_details", {})
    composition = [f"{number(weight * 100, 2)}% {labels.get(by_name.get(leaf(path), {}).get('model'), leaf(path))}"
                   for path, weight in zip(details.get("sources", []), details.get("weights", [])) if weight > 0]
    points = (["Composição final em probabilidades: " + "; ".join(composition)] if composition else
              ["Modelo selecionado: " + labels.get(selected.get("model"), selected.get("model", "aguardando execução"))])
    if has_bpe(runs) and has_frequency(runs):
        offset = next((run for run in selected_runs(runs) if run.get("model") == "offset"), {})
        stack = next((run for run in selected_runs(runs) if run.get("model") == "nested_stack"), {})
        encoded = offset.get("model_details", {}).get("encoded_features")
        if stack:
            points.append("O stack próprio aprende a combinar três logits de bases; os coeficientes internos não são percentuais")
        elif encoded:
            methods = ["BPE", "frequência"]
            if offset.get("model_details", {}).get("pooled_tokens"):
                methods.append("pooling")
            if offset.get("feature_encoding", {}).get("joint_encoding"):
                methods.append("oito interações")
            points.append(f"Base offset própria: {offset.get('features', {}).get('count')} colunas → {encoded} entradas; " + ", ".join(methods))
        else:
            points.append("O GLM reúne pistas numéricas, chaves BPE e frequência")
    compared = sorted({labels[run["model"]] for run in runs
                       if run.get("status") == "complete" and run.get("model") in {"mlp", "tabm", "neural_offset"}
                       and run not in selected_runs(runs)})
    if compared:
        points.append("Redes próprias treinadas e comparadas: " + "; ".join(compared))
    authors = sorted({component["trained_by"] for component in external_components if component.get("trained_by")})
    if authors:
        points.append("Treinamento do GPT-2 externo: " + "; ".join(authors) + "; importamos suas previsões auditadas")
    add("01 / O problema", "A proposta: comparar, combinar e comprovar", points,
        "Nossa proposta é comparar modelos e reunir previsões complementares. "
        + ". ".join(points) + ". A seguir, vamos explicar como cada parte funciona, "
        "onde foi utilizada e o que os experimentos realmente mostraram. "
        "Treinar uma rede para comparação não significa que ela precise entrar na composição final.")


def append_one_hot_pair(pair, runs):
    if not any(run.get("status") == "complete" and run.get("model") in {"glm", "offset"} for run in runs):
        return
    pair("04 / Os modelos", "One-hot: uma posição para cada categoria", [
        "Cada categoria conhecida do treino ganha uma posição no vetor",
        "A posição correspondente recebe 1; as demais recebem 0",
        "O GLM usa esta representação nas categorias originais",
        "No código, categoria desconhecida produz zeros; o vetor não usa o alvo",
    ], "One-hot transforma um rótulo em posições de zero ou um. A lista de "
       "categorias é aprendida somente no treino. Para a categoria observada, "
       "ativamos sua posição; as outras ficam em zero. A fórmula usa um "
       "indicador, que vale um quando c é igual à categoria conhecida v de "
       "índice j. O GLM recebe essa representação das categorias originais "
       "junto das medidas e dos logits codificados por alvo. Na configuração "
       "executada, uma categoria desconhecida é ignorada pelo encoder e "
       "recebe zeros nesse bloco. Não atribuímos uma ordem de grandeza aos rótulos.",
       r"\mathrm{OHE}(c)_j=\mathbf{1}[c=v_j],\qquad j=1,\ldots,C",
       "Posição j = um se a categoria é v de j; zero caso contrário",
       "No dia a dia: marcar uma opção no formulário", [
           "Um formulário oferece azul, verde e vermelho",
           "Escolher verde marca somente a segunda opção: 0, 1, 0",
           "A posição identifica a escolha; não diz que verde vale mais que azul",
       ], "Imagine um formulário com três opções de cor. Ao escolher verde, "
          "você marca apenas essa opção e deixa as outras vazias. O vetor "
          "fica zero, um, zero. A marca identifica a opção, sem dizer que "
          "uma cor é maior ou melhor que outra.")


def missing_selection_context(evidence):
    missing = []
    status = evidence.get("presentation_input_status")
    if status and status != "final_frozen":
        missing.append("presentation_input_status final_frozen")
    if status and not evidence.get("run", {}).get("selection_reason_pt"):
        missing.append("run.selection_reason_pt")
    for item in evidence.get("historical_submissions", []):
        if item.get("protocol_deviation"):
            if item.get("declared_before_upload") is not True:
                missing.append("historical deviation declared_before_upload")
            if any(not isinstance(item.get(key), int) or isinstance(item.get(key), bool)
                   for key in ("improved_folds", "required_folds", "total_folds")):
                missing.append("historical deviation fold evidence")
    decision = evidence.get("public_evaluation", {})
    if not decision.get("protocol_deviation"):
        return missing
    if decision.get("run") != leaf(evidence.get("run", {}).get("path", "")):
        missing.append("public_evaluation.run matching selected run")
    for key in ("improved_folds", "required_folds", "total_folds"):
        value = decision.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            missing.append("public_evaluation." + key)
    if decision.get("consistency_gate_passed") is not False:
        missing.append("public_evaluation.consistency_gate_passed false")
    if decision.get("declared_before_upload") is not True:
        missing.append("public_evaluation.declared_before_upload")
    if not evidence.get("run", {}).get("selection_reason_pt"):
        missing.append("run.selection_reason_pt")
    return missing


def append_feature_pairs(pair, runs):
    if has_bpe(runs):
        pair("04 / Representar os dados", "BPE numérico: seis chaves, sem treinar um GPT-2", [
            "Valor e saldo: três chaves categóricas por coluna",
            "1269,35 → texto ' 1269' → tokens ' 12' (1105) e '69' (3388)",
            "Chaves: primeiro ID, segundo ID e segundo ID:centavos; exemplo 3388:35",
            "IDs não indicam ordem; segundo ausente = −1; tokens após o segundo são descartados",
        ], "Aqui usamos somente o tokenizer oficial do GPT-2, não seus pesos ou previsões. "
           "O valor é convertido em centavos com arredondamento NumPy rint sobre float sessenta e quatro. "
           "Se houver uma passagem para o próximo inteiro, ela é preservada. "
           "No exemplo, mil duzentos e sessenta e nove vírgula trinta e cinco fornece o inteiro "
           "mil duzentos e sessenta e nove e trinta e cinco centavos. O texto começa com um espaço. "
           "O tokenizer separa esse inteiro em dois pedaços. Guardamos seus identificadores "
           "e uma combinação do segundo identificador com os centavos. Fazemos isso para valor "
           "e saldo, produzindo seis chaves. Os identificadores são categorias, não medidas "
           "de grandeza ou de risco. Essa hipótese não prova como os dados foram gerados.",
           r"c=\mathrm{rint}(100x),\quad u=\lfloor c/100\rfloor,\quad d=c-100u",
           "Centavos = arredondamento de 100 vezes x; inteiro = piso de centavos dividido por 100; restante = centavos menos 100 vezes inteiro",
           "No dia a dia: separar uma etiqueta em partes", [
               "Uma etiqueta de estoque pode ter partes que se repetem em produtos diferentes",
               "Um identificador maior não significa um produto melhor ou mais arriscado",
               "O modelo aprende se compartilhar esses pedaços ajuda a comparar os casos",
           ], "Imagine etiquetas de estoque divididas em blocos. Produtos diferentes podem "
              "compartilhar uma parte da etiqueta. Isso oferece outra maneira de agrupá-los. "
              "Mas o código novecentos não é automaticamente melhor ou pior que o código "
              "duzentos. Aqui acontece o mesmo: os pedaços recebem nomes numéricos, mas são "
              "tratados como rótulos. Só o experimento informa se essa representação ajuda.")
    if has_frequency(runs) and not any(run.get("model_details", {}).get("transductive_frequency")
                                     for run in selected_runs(runs)):
        pair("04 / Representar os dados", "Frequência: quantas vezes uma chave apareceu no treino", [
            "Cada coluna tem sua própria tabela de contagens, ajustada no treino do fold",
            "A entrada adicional é log1p da contagem; depois ela é escalonada",
            "Chave desconhecida tem contagem zero; a contagem não usa o rótulo",
            "Frequência não é taxa de fraude nem substitui o target encoding",
        ], "Além de estimar o risco associado a uma chave, podemos informar quantas vezes "
           "ela apareceu. A tabela é construída somente com as características do treino "
           "da divisão, sem usar os rótulos. Para cada coluna, contamos as ocorrências "
           "do valor e aplicamos log de um mais essa contagem. O log comprime diferenças "
           "muito grandes de frequência. Uma chave desconhecida recebe zero antes dessa "
           "transformação. Esta entrada descreve suporte na amostra; ela não diz, por si "
           "só, que um caso é fraudulento. O modelo aprende sua contribuição junto das "
           "outras representações.",
           r"n_j(v)=\sum_{i\in T}\mathbf{1}[x_{ij}=v],\quad f_j(v)=\log(1+n_j(v))",
           "Contagem da chave v na coluna j = soma dos indicadores no treino; frequência transformada = log de 1 mais a contagem",
           "No dia a dia: palavras em um caderno de exemplos", [
               "Uma palavra vista uma vez tem menos exemplos que outra vista cem vezes",
               "Contar ocorrências não diz se a palavra está certa ou errada",
               "As páginas reservadas para conferir o resultado ficam fora da contagem",
           ], "Pense num caderno de exemplos. Saber que uma palavra apareceu cem vezes "
              "ou apenas uma vez descreve a quantidade de exemplos disponíveis. Isso "
              "não diz se a palavra foi usada corretamente. Para conferir o aprendizado, "
              "as páginas de teste não entram na contagem. Nossa tabela segue essa "
              "separação entre o material permitido no treino e os exemplos reservados.")


def append_offset_pair(pair, runs):
    run = next((run for run in selected_runs(runs) if run.get("model") == "offset"), None)
    if not run:
        return
    details = run.get("model_details", {})
    parameters = details.get("residual_parameters", {})
    pair("04 / Os modelos", "Offset: árvores corrigem a margem do GLM", [
        "O GLM fornece init_score no treino, antes da sigmoid",
        f"Árvores: {details.get('best_iteration', parameters.get('n_estimators'))}; folhas: {parameters.get('num_leaves')}; taxa: {parameters.get('learning_rate')}; L2: {parameters.get('reg_lambda')}",
        "GLM e árvores usam a mesma matriz de treino com target encoding cross-fitted",
        "Na inferência, as margens são somadas explicitamente; uma única sigmoid vem ao final",
    ], "O ponto de partida das árvores é a margem que o GLM já calculou. "
       "As árvores aprendem correções sob a perda binária. Isso não é uma regressão "
       "comum sobre o rótulo menos a probabilidade. A matriz codificada de treino "
       "é a mesma para as duas etapas. No teste, init score não reaparece "
       "automaticamente: o código soma a margem do GLM ao raw score das árvores "
       "e só então aplica a sigmoid. O raw score já incorpora a taxa de aprendizado "
       "das árvores; não a multiplicamos novamente.",
       r"z(x)=z_{\mathrm{GLM}}(x)+\eta\sum_t h_t(\phi(x)),\quad\hat p=\sigma(z)",
       "Margem final = margem do GLM mais contribuições das árvores; previsão = sigmoid da margem final",
       "No dia a dia: primeira avaliação e correções", [
           "Uma primeira estimativa dá o ponto de partida de um orçamento",
           "Revisões pequenas acrescentam ou retiram quantias",
           "O total precisa reunir a estimativa inicial e todas as correções",
       ], "Imagine um orçamento. Primeiro você faz uma estimativa geral e depois "
          "corrige alguns detalhes. Guardar apenas as correções faria você perder "
          "o valor inicial. Por isso salvamos o GLM e as árvores e somamos as duas "
          "partes na hora de produzir a previsão.")


def append_tabm_embedding_pair(pair, runs):
    run = next((run for run in runs if run.get("status") == "complete" and run.get("model") == "tabm"), None)
    if not run:
        return
    embedding = run.get("model_details", {}).get("architecture", {}).get("num_embeddings", {})
    if (embedding.get("name") != "PiecewiseLinearEmbeddings" or embedding.get("version") != "B"
            or embedding.get("activation") is not False):
        return
    dimensions, bins = embedding.get("d_embedding"), embedding.get("n_bins")
    pair("06 / Experimento neural", "TabM: representar posições entre cortes do treino", [
        f"Cortes por quantis do treino; até {bins} intervalos por entrada ativa",
        "Entre duas marcas, θ representa a fração percorrida no intervalo",
        f"Versão B: caminho linear + projeção aprendida; vetor de {dimensions} dimensões, sem ReLU final",
        "A expressão abaixo está limitada à faixa coberta pelos cortes do treino",
    ], "Antes dos blocos TabM, cada entrada numérica preparada ganha uma "
       "representação por partes. Os cortes são calculados somente no treino "
       "da etapa, com até " + str(bins) + " intervalos. Dentro de um intervalo, "
       "teta mede quanto caminhamos entre a marca esquerda e a direita. "
       "A codificação acumula os intervalos completos, a fração do intervalo "
       "atual e zeros nos seguintes. Na versão B, uma projeção linear do "
       "valor é somada à projeção aprendida dessa codificação, formando "
       + str(dimensions) + " coordenadas por entrada. A correção por partes "
       "começa em zero e é aprendida durante o treino. Esta fórmula descreve "
       "valores entre os cortes extremos do treino; não especifica o "
       "comportamento fora dessa faixa. As coordenadas não são um ranking "
       "de risco e não atribuímos ordem numérica a identificadores brutos.",
       r"\theta=\frac{x-b_\ell}{b_{\ell+1}-b_\ell},\quad e(x)=ax+b+W[1,\ldots,1,\theta,0,\ldots,0]^\top\in\mathbb{R}^{" + str(dimensions) + "}",
       "Dentro do intervalo: teta é a fração entre os cortes; embedding = caminho linear mais projeção da codificação acumulada, com " + str(dimensions) + " coordenadas",
       "No dia a dia: uma régua entre duas marcas", [
           "Entre as marcas 10 e 20, o valor 15 fica na metade do caminho",
           "Representamos essa passagem de forma gradual",
           f"Uma ficha de {dimensions} campos aprende como usar essa posição",
       ], "Imagine uma régua com as marcas dez e vinte. O valor quinze está "
          "na metade do caminho. Essa fração é o teta da fórmula. Em vez de "
          "saltar de uma etiqueta para outra, registramos a passagem gradual "
          "entre as marcas. Depois, o modelo aprende a transformar essa "
          "posição numa ficha de " + str(dimensions) + " campos. A régua "
          "localiza o número; aprender a ficha é trabalho do treinamento.")


def append_tabm_backbone_pair(pair, runs):
    run = next((run for run in runs if run.get("status") == "complete" and run.get("model") == "tabm"
                and run.get("versions", {}).get("tabm") == "0.0.3"), None)
    if not run:
        return
    dropout = run.get("model_details", {}).get("architecture", {}).get("dropout")
    pair("06 / Experimento neural", "TabM: uma matriz comum, ajustes de cada membro", [
        "Em cada camada do backbone, W é compartilhada entre os membros",
        "Os vetores r, s e o viés b são próprios de cada membro",
        f"Após a transformação: ReLU e dropout registrado de {dropout}",
        "A cabeça de saída é uma camada linear própria de cada membro, com seus pesos e viés",
    ], "O TabM executado usa a arquitetura tabm da biblioteca oficial, "
       "versão zero ponto zero ponto três. Em cada camada interna, os "
       "membros compartilham uma matriz de pesos W. Cada membro também "
       "tem vetores próprios r e s e um viés próprio b. O símbolo de "
       "multiplicação com círculo representa multiplicar posição por "
       "posição. Primeiro multiplicamos a entrada por r, depois usamos "
       "a matriz compartilhada transposta, multiplicamos por s e somamos "
       "o viés daquele membro. Os blocos aplicam ReLU e dropout. A cabeça "
       "de saída tem pesos e viés separados para cada membro; ela não "
       "compartilha a matriz W dessa fórmula. O compartilhamento reduz "
       "o custo de manter várias previsões, sem torná-las idênticas.",
       r"a_m=((h_m\odot r_m)W^{\top})\odot s_m+b_m",
       "Saída interna do membro m = entrada vezes seu r, matriz W compartilhada transposta, vezes seu s, mais seu viés b",
       "No dia a dia: uma receita comum, temperos individuais", [
           "Os cozinheiros partem da mesma receita",
           "Cada um ajusta quantidades e acabamento à sua maneira",
           "Ao final, cada cozinheiro também produz sua própria avaliação",
       ], "Imagine cozinheiros seguindo a mesma receita. Cada um pode "
          "ajustar algumas quantidades antes de cozinhar e o acabamento "
          "depois. Compartilhar a receita economiza trabalho, mas os "
          "ajustes continuam individuais. Ao final, cada cozinheiro "
          "produz sua própria avaliação. No TabM, os ajustes e as "
          "avaliações são operações numéricas aprendidas.")


def append_tabm_pair(pair, runs):
    run = next((run for run in runs if run.get("status") == "complete" and run.get("model") == "tabm"), None)
    if not run:
        return
    architecture = run.get("model_details", {}).get("architecture", {})
    k = architecture.get("k")
    pair("06 / Experimento neural", "TabM: várias previsões com pesos compartilhados", [
        f"Arquitetura registrada: {architecture.get('n_blocks')} blocos de {architecture.get('d_block')}; {k} membros",
        "Cada membro recebe sua própria perda binária; não treinamos a perda da média",
        "Na inferência, combinamos probabilidades; não logits",
        "É um experimento próprio comparado; sua presença aqui não implica seleção",
    ], "Também treinamos a arquitetura TabM usando as bibliotecas oficiais. "
       "Os membros compartilham a maior parte dos pesos e mantêm diferenças "
       "próprias. Cada um produz uma previsão e recebe uma perda binária. "
       "A perda média é calculada sobre exemplos e membros. Na inferência, "
       "aplicamos sigmoid em cada saída e tiramos a média das probabilidades. "
       "A época foi escolhida numa divisão interna de quinze por cento; depois "
       "o modelo foi reiniciado e treinado por essa quantidade de épocas no "
       "treino completo do fold. Os rótulos externos não escolheram a época.",
       r"\mathcal{L}=\frac{1}{Bk}\sum_{b,m}\ell(y_b,z_{bm}),\quad\hat p_b=\frac{1}{k}\sum_m\sigma(z_{bm})",
       "Perda = média das perdas de todos os exemplos e membros; previsão = média das probabilidades dos membros",
       "No dia a dia: analistas com um manual em comum", [
           "Os analistas usam o mesmo manual e destacam detalhes de formas diferentes",
           "Cada analista recebe uma avaliação durante o estudo",
           "Na decisão final, combinamos suas estimativas",
       ], "Imagine analistas que compartilham um manual, mas destacam detalhes "
          "de maneira própria. Cada analista recebe sua avaliação durante o estudo. "
          "Na decisão final, combinamos as estimativas. Compartilhar o manual "
          "economiza recursos, mas ter mais analistas não garante uma decisão "
          "melhor. Por isso essa arquitetura também precisou passar pelo experimento.")


def append_tabm_result(add, runs, number):
    by_name = {name(run): run for run in runs if run.get("status") == "complete"}
    candidate = next((run for run in by_name.values() if run.get("model") == "tabm"), None)
    baseline, mixture = by_name.get("explore5-public25"), by_name.get("round8-tabm10")
    if not all([candidate, baseline, mixture]) or candidate in selected_runs(runs):
        return
    paired = list(zip(mixture.get("fold_scores", []), baseline.get("fold_scores", [])))
    delta = mixture["roc_auc"] - baseline["roc_auc"]
    add("07 / Os experimentos", "TabM: treinado e descartado desta composição", [
        f"TabM própria: AUC OOF {number(candidate['roc_auc'], 5)}",
        f"Base anterior: {number(baseline['roc_auc'], 5)}; mistura com 10% TabM: {number(mixture['roc_auc'], 5)}",
        f"Delta observado: {number(delta, 5)}; melhora em {sum(a > b for a, b in paired)}/{len(paired)} folds",
        "A comparação fixa não justificou sua inclusão; a rede não foi selecionada",
    ], "A arquitetura foi treinada de verdade, mas isso não garante utilidade "
       "na solução final. A mistura fixa com dez por cento de TabM ficou "
       "abaixo da base anterior nesta comparação. O resultado foi preservado "
       "como negativo e a rede não entrou nesta composição. Não transformamos "
       "uma arquitetura mais complexa em uma vitória sem medir.")


def append_selection_note(add, evidence):
    decision = evidence.get("public_evaluation", {})
    if not decision.get("protocol_deviation"):
        return False
    improved, required = decision.get("improved_folds"), decision.get("required_folds")
    add("07 / Os experimentos", "Decisão exploratória: o critério original não foi cumprido", [
        f"Melhora em {improved}/{decision.get('total_folds')} folds; regra exigia {required}/{decision.get('total_folds')}",
        "O ganho agregado não tornou verdadeira a exigência de quatro folds melhores",
        "A exceção foi registrada antes do upload; não foi uma promoção robusta pela OOF",
        "O score público passou a ser feedback exploratório de seleção",
    ], "Esta decisão exige uma distinção importante. O candidato melhorou "
       "a AUC agregada, mas não atingiu a quantidade de folds melhores exigida "
       "na regra original. Esse critério continuou registrado como não "
       "cumprido. Depois de observar os resultados locais, foi permitida uma "
       "avaliação pública exploratória, registrada antes do upload. Isso foi "
       "uma exceção informada pelos resultados, não uma demonstração de "
       "robustez. O score público ajudou a escolher entre arquivos e não "
       "substitui uma avaliação independente final.")
    return True


def append_submission_history(add, evidence, number, observation_time):
    history = [item for item in evidence.get("historical_submissions", [])
               if item.get("submission_file_verified") is True]
    for start in range(0, len(history), 3):
        group = history[start:start + 3]
        points = [f"Envio {item['submission_id']}: AUC {number(item['public_auc'], 5)}; "
                  f"{observation_time(item['observed_at'])}" for item in group]
        add("09 / A evidência final", "Envios registrados: snapshots preservados", points,
            "Os registros anteriores permanecem associados aos seus próprios arquivos "
            "e datas. Um score de um envio não pode ser transferido para outro CSV. "
            "Essas consultas públicas fazem parte da trajetória exploratória; a posição "
            "atual desta apresentação aparece após esse histórico e continua provisória.")


def append_historical_deviations(add, evidence):
    for item in evidence.get("historical_submissions", []):
        if item.get("submission_file_verified") is not True or not item.get("protocol_deviation"):
            continue
        add("07 / Os experimentos", "Etapa histórica: uma exceção exploratória", [
            f"Envio {item['submission_id']}: melhora em {item['improved_folds']}/{item['total_folds']} folds; exigia {item['required_folds']}/{item['total_folds']}",
            "O critério original continuou registrado como não cumprido",
            "A avaliação pública excepcional foi declarada antes do upload",
            "A decisão atual é apresentada separadamente; não reclassifica esse resultado",
        ], "Este envio anterior foi uma exceção exploratória. Ele melhorou a "
           "métrica agregada, mas não a quantidade de folds exigida pelo critério "
           "original. A falha permaneceu registrada. A exceção foi informada "
           "pelos resultados locais e declarada antes do upload. O feedback "
           "público entrou na seleção exploratória. Essa história não muda "
           "os critérios nem os resultados da solução atual.")


def continuation_sources(runs):
    sources = []
    if has_bpe(runs):
        sources += [
            ("Tokenizer oficial GPT-2", "https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py"),
            ("Ideia de BPE numérico: azizdzhon", "https://www.kaggle.com/code/azizdzhon/how-great-tokenizes-numbers-bpe-token-target"),
        ]
    if has_frequency(runs):
        sources.append(("Contagem por coluna", "https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html"))
    if any(run.get("status") == "complete" and run.get("model") == "tabm" for run in runs):
        sources += [("TabM 0.0.3: arquitetura e implementação", "https://pypi.org/project/tabm/0.0.3/"),
                    ("Embeddings numéricos 0.0.12", "https://pypi.org/project/rtdl_num_embeddings/0.0.12/")]
    if any(run.get("model_details", {}).get("pooled_tokens") for run in selected_runs(runs)):
        sources.append(("Ideias de pooling e interações: crédito a megayak, Part 12 v16 de goodpjw2008",
                        "https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298"))
    if any(run.get("status") == "complete" and run.get("model") == "neural_offset" for run in runs):
        sources.append(("SiLU: implementação oficial PyTorch", "https://docs.pytorch.org/docs/2.11/generated/torch.nn.SiLU.html"))
    if any("strings.py" in run.get("source_hashes", {}) for run in selected_runs(runs)):
        sources.append(("Valores únicos e índices inversos: NumPy", "https://numpy.org/doc/2.2/reference/generated/numpy.unique.html"))
    return sources


def method_code_specs(runs):
    specifications = []
    if has_bpe(runs):
        specifications.append(("src/fraudlab/bpe.py", "integers, inverse = np.unique", 2,
            "Código: extrair pedaços do número", ["Trecho próprio; tokenizer oficial e entradas já convertidas em centavos"],
            "O trecho obtém o inteiro dos centavos canonizados e chama o tokenizer. "
            "A extração não recebe rótulos. Os identificadores resultantes entram como categorias."))
    if has_frequency(runs):
        specifications.append(("src/fraudlab/frequency.py", "counts = frame[name].map", 2,
            "Código: consultar o suporte aprendido no treino", ["A tabela já foi ajustada somente nas características do treino"],
            "O código consulta a tabela da coluna, preenche desconhecidos com zero e aplica log de um mais a contagem. "
            "A tabela fica salva para usar a mesma referência na inferência."))
    if any(run.get("model") == "offset" for run in selected_runs(runs)):
        specifications.append(("src/fraudlab/offset.py", "margin = self.glm", 3,
            "Código: preservar a margem inicial na inferência", ["O GLM e as árvores são somados antes da sigmoid"],
            "Estas linhas somam explicitamente as duas margens. "
            "O resultado das árvores sozinho não contém a previsão inicial do GLM."))
    if any(run.get("model_details", {}).get("pooled_tokens") for run in selected_runs(runs)):
        specifications.append(("src/fraudlab/pooled.py", 'rate = (pooled["sum"]', 3,
            "Código: suavizar a taxa compartilhada", ["Somas e contagens vêm somente do treino permitido pelo cross-fitting"],
            "A taxa compartilhada combina positivos do grupo com o prior e a força de suavização. "
            "Ela fornece a referência para a taxa de cada campo; o arquivo também contém o bloco condicional."))
    if any(run.get("feature_encoding", {}).get("joint_encoding") for run in selected_runs(runs)):
        specifications.append(("src/fraudlab/joints.py", "pairs = zip(", 4,
            "Código: formar as oito tuplas fixas", ["As views já foram validadas; repr preserva os pares sem ambiguidade"],
            "As duas representações são reunidas em uma tupla e convertidas em uma categoria textual. "
            "Não há leitura de alvo, procura de pares ou estatística aprendida nessa extração."))
    if any(run.get("status") == "complete" and run.get("model") == "neural_offset" for run in runs):
        specifications.append(("src/fraudlab/neural_offset.py", "logits = offsets[selected]", 4,
            "Código: a perda avalia GLM mais correção neural", ["A margem congelada é somada à saída da rede antes da perda"],
            "Este trecho soma a margem do GLM à correção neural e calcula a perda binária com logits. "
            "O GLM não é atualizado pelo AdamW; o otimizador recebe os parâmetros da rede residual."))
    if any("strings.py" in run.get("source_hashes", {}) for run in selected_runs(runs)):
        specifications.append(("src/fraudlab/strings.py", "unique, inverse = np.unique", 2,
            "Código: compartilhar textos mantendo a sequência", ["Conversão dos valores distintos; índices inversos restauram as posições"],
            "A conversão para string acontece nos valores distintos. Os índices inversos reconstroem o array "
            "com dtype object e referências compartilhadas. Essa mudança de armazenamento preserva os valores."))
    return specifications


def method_code_analogy(relative):
    labels = {
        "src/fraudlab/bpe.py": ("No dia a dia: ler partes de uma etiqueta", "Consultar partes de uma etiqueta oferece outra maneira de agrupar os itens; não transforma o código em uma medida de risco."),
        "src/fraudlab/frequency.py": ("No dia a dia: consultar o caderno de exemplos", "Usamos a contagem guardada no caderno de treino. Não acrescentamos as páginas da prova antes de responder."),
        "src/fraudlab/offset.py": ("No dia a dia: somar orçamento e correções", "O total precisa conter o orçamento inicial e os ajustes. Guardar somente os ajustes perderia o ponto de partida."),
        "src/fraudlab/pooled.py": ("No dia a dia: começar pela referência da rede", "Uma loja com pouca evidência pode começar pela avaliação compartilhada antes de confiar na sua própria média."),
        "src/fraudlab/joints.py": ("No dia a dia: preencher rua e número", "Juntar dois campos preserva a combinação específica. O número da casa sozinho não informa o endereço completo."),
        "src/fraudlab/neural_offset.py": ("No dia a dia: avaliar o orçamento corrigido", "A avaliação olha o orçamento inicial junto das correções. Medir apenas a correção perderia o total que será usado."),
        "src/fraudlab/strings.py": ("No dia a dia: várias fichas para um verbete", "As fichas podem apontar para um mesmo verbete sem mudar sua ordem nem o significado. Economizar cópias não acrescenta informação."),
    }
    return labels.get(relative)
