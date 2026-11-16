"""Presentation content derived from experiment and competition evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
import re
from typing import Any
from zoneinfo import ZoneInfo

from .external import components_from, gpt2_slides


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


def is_refit(run: dict) -> bool:
    return run.get("phase") == "full_data_refit"


def run_label(run: dict) -> str:
    return str(run.get("run_name", run.get("model", "modelo")))


def observation_time(value: str | None, spoken: bool = False) -> str:
    if not value:
        return "não verificada"
    local = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ZoneInfo("America/Sao_Paulo"))
    if not spoken:
        return local.strftime("%d/%m/%Y às %Hh%M (São Paulo)")
    months = ("janeiro", "fevereiro", "março", "abril", "maio", "junho",
              "julho", "agosto", "setembro", "outubro", "novembro", "dezembro")
    return (f"{local.day} de {months[local.month - 1]} de {local.year}, "
            f"às {local.hour} horas e {local.minute} minutos, no horário de São Paulo")


def submission_matches(evidence: dict, selected: dict) -> bool:
    recorded = (evidence.get("run") or {}).get("submission_sha256")
    expected = selected.get("submission_sha256")
    return isinstance(recorded, str) and bool(re.fullmatch(r"[a-f0-9]{64}", recorded)) and recorded == expected


def validate_evidence(evidence: dict, runs: list[dict], draft: bool, holdout: dict | None = None,
                      historical_holdout: bool = False) -> list[str]:
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
    if runs and not submission_matches(evidence, runs[0]):
        missing.append("run.submission_sha256 matching selected metrics")
    for run in runs:
        auc = run.get("roc_auc")
        if is_refit(run):
            auc = run.get("selection", {}).get("development_roc_auc")
        if not isinstance(auc, (float, int)) or isinstance(auc, bool) or not math.isfinite(auc) or not 0 <= auc <= 1:
            missing.append("experiment.roc_auc")
        if run.get("status") != "complete":
            missing.append("experiment.status complete")
        if not run.get("data_hashes"):
            missing.append("experiment.data_hashes")
        elif runs and run["data_hashes"] != runs[0].get("data_hashes"):
            missing.append("experiment compatible data_hashes")
        if is_refit(run) and not any(
            not is_refit(other) and other.get("model") == run.get("model") and
            other.get("roc_auc") == run.get("selection", {}).get("development_roc_auc")
            for other in runs
        ):
            missing.append("refit development source metrics")
    selected_components = components_from(runs[:1])
    needs_holdout = (runs and is_refit(runs[0]) and not selected_components and
                     runs[0].get("validation", {}).get("independent_holdout") is not False)
    if needs_holdout and (not holdout or historical_holdout):
        missing.append("independent holdout report")
    if selected_components and holdout and not historical_holdout:
        missing.append("old holdout must be historical for the external extension")
    for component in selected_components:
        fields = ("trained_by", "training_code_url", "training_code_commit", "code_license", "predictions_source_url")
        if any(not component.get(field) for field in fields) or not component.get("predictions_license", component.get("data_license")):
            missing.append("external component provenance and licenses")
    if holdout:
        score = holdout.get("roc_auc")
        if holdout.get("status") != "complete" or holdout.get("kind") != "independent_holdout":
            missing.append("completed independent holdout report")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            missing.append("holdout.roc_auc")
        if runs and holdout.get("data_hashes") != runs[0].get("data_hashes"):
            missing.append("holdout compatible data_hashes")
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


def create_slides(evidence: dict, runs: list[dict], draft: bool = False, holdout: dict | None = None,
                  historical_holdout: bool = False) -> list[Slide]:
    missing = validate_evidence(evidence, runs, draft, holdout, historical_holdout)
    selected = runs[0] if runs else {}
    development_runs = [run for run in runs if not is_refit(run)]
    evaluation_run = next((run for run in development_runs if run.get("model") == selected.get("model")),
                          development_runs[0] if development_runs else selected)
    neural = next((run for run in runs if run.get("model") == "mlp"), {})
    details = neural.get("model_details", {})
    architecture = details.get("architecture", {})
    results = evidence.get("results", {}).copy()
    if not submission_matches(evidence, selected):
        for key in ("public_auc", "ranking", "total_teams", "submission_id", "observed_at"):
            results[key] = None
    if holdout:
        results["holdout_auc"] = holdout.get("roc_auc")
        results["holdout_observed_at"] = holdout.get("evaluated_at")
    rows = selected.get("rows", {})
    features = selected.get("features", {})
    raw_run = next((run for run in runs if run.get("model_details", {}).get("feature_set") == "raw"), {})
    raw_count = raw_run.get("features", {}).get("count")
    validation = evaluation_run.get("validation", {})
    reserved_run = next((run for run in runs if run.get("validation", {}).get("holdout_fraction", 0) > 0), {})
    all_names = {name for run in runs for name in run.get("features", {}).get("names", [])}
    target_runs = [run for run in runs if run.get("model_details", {}).get("target_encoder")]
    external_components = components_from(runs)
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
        (f"Colunas originais: {number(raw_count)}; entradas após representação: {number(features.get('count'))}"
         if raw_count is not None else f"Entradas após representação: {number(features.get('count'))}"),
        "Taxa positiva no treino: " + (number(selected.get("target", {}).get("positive_rate", 0) * 100, 3) + "%" if selected.get("target", {}).get("positive_rate") is not None else "não verificada"),
        "Identificadores e alvo são tratados separadamente das entradas",
    ], "Antes de escolher uma arquitetura, é preciso conhecer os arquivos. "
       + (f"Neste experimento foram registrados {number(rows.get('train'))} exemplos de treino "
          f"e {number(rows.get('test'))} exemplos de teste. "
          + (f"O arquivo possui {number(raw_count)} colunas originais de entrada. " if raw_count is not None else "")
          + f"Depois da representação selecionada, o modelo usa {number(features.get('count'))} entradas. " if selected else
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
    method = {"stratified": "estratificado", "StratifiedKFold": "estratificado",
              "group": "por grupos", "time": "temporal"}.get(
                  validation.get("method"), "não verificado")
    validation_note = (f"{number(validation.get('n_splits'))} folds sobre todos os rótulos; sem nova reserva independente. "
                       "Os parâmetros dos modelos próprios permaneceram fixos."
                       if validation.get("independent_holdout") is False else
                       "A justificativa técnica da divisão permanece registrada nas métricas da execução.")
    add("02 / Os dados", "A divisão precisa representar a pergunta", [
        f"Estratégia registrada: {method}",
        f"Divisões registradas: {number(validation.get('n_splits'))}",
        validation_note,
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
        reserve_title = ("Protocolo inicial: uma reserva separada" if historical_holdout
                         else "Uma reserva fora da seleção de modelos")
        reserve_note = ("Após a extensão neural, essa reserva deixou de ser independente para a nova composição. "
                        if historical_holdout else "")
        pair("03 / A avaliação", reserve_title, [
            f"Reserva registrada: {number(fraction * 100, 0)}% do treino ({number(reserved.get('holdout'))} linhas)",
            f"Desenvolvimento: {number(reserved.get('development'))} linhas para comparar configurações",
            "A reserva não participa dos folds usados na escolha",
            "Uma avaliação registrada é necessária antes de citar seu score",
        ], "O processo de desenvolvimento também separou uma reserva. "
           "Na fórmula, D é o conjunto de treino completo, desenvolvimento "
           "é a parte usada para escolher os modelos e reserva é a parte "
           "mantida fora desse processo. A interseção é vazia: a mesma linha "
           "não pertence às duas partes. Essa reserva só cumpre seu papel "
           "se não for consultada repetidamente para escolher configurações. " + reserve_note,
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
    if any(name.endswith("__log1p") for name in all_names):
        pair("04 / Representar os dados", "Logs: outra visão da escala", [
            "A transformação preserva o sinal e comprime magnitudes grandes",
            "O valor original continua disponível para o modelo",
            "É uma hipótese de representação; o ganho precisa ser medido",
        ], "Para algumas entradas contínuas, criamos uma segunda visão com "
           "log de um mais o valor absoluto, preservando o sinal. Isso "
           "reduz diferenças de escala entre valores muito grandes, "
           "sem apagar o valor original. A transformação não descobre "
           "fraude por si só. Sua utilidade depende da comparação com "
           "o experimento que não a utiliza.",
           r"u(x)=\mathrm{sgn}(x)\log(1+|x|)",
           "Nova visão = sinal de x vezes log de um mais o valor absoluto de x",
           "No dia a dia: olhar uma diferença de perto e de longe", [
               "Uma régua comum destaca diferenças absolutas",
               "Uma escala comprimida destaca diferenças relativas",
               "Manter as duas visões permite testar qual ajuda",
           ], "Imagine comparar o tamanho de objetos pequenos e muito grandes. "
              "Uma única régua pode dificultar enxergar os menores. Uma "
              "escala comprimida dá outra perspectiva. No experimento, "
              "o modelo recebe essa perspectiva junto da medida original.")
    if any(name.endswith("__sin") for name in all_names):
        pair("04 / Representar os dados", "Hora e dia formam ciclos", [
            "Hora do dia usa período 24; dia da semana usa período 7",
            "Seno e cosseno colocam a posição em um círculo",
            "A passagem do último valor ao primeiro fica contínua",
        ], "Uma hora do dia não é apenas um número em uma reta. "
           "Depois de vinte e três horas vem zero, e esses momentos "
           "são próximos. Representar a posição por seno e cosseno "
           "mantém essa proximidade. O período T é vinte e quatro "
           "para a hora e sete para o dia da semana.",
           r"v(t)=\left(\sin\frac{2\pi t}{T},\cos\frac{2\pi t}{T}\right)",
           "Posição de t no ciclo = seno e cosseno de dois pi t dividido pelo período",
           "No dia a dia: um relógio redondo", [
               "23 horas e 0 hora estão próximas no relógio",
               "Numa reta, os números parecem distantes",
               "A representação circular preserva essa vizinhança",
           ], "Pense no mostrador de um relógio. O fim e o começo "
              "do dia ficam lado a lado. Se desenharmos as horas "
              "apenas numa reta, essa vizinhança desaparece. Seno "
              "e cosseno levam a informação de volta para o círculo.")
    if "amount_balance_ratio" in all_names:
        pair("04 / Representar os dados", "Relacionar valor e saldo", [
            "A razão combina duas entradas disponíveis",
            "O módulo e o acréscimo de 1 evitam divisão por zero",
            "Esta construção não define uma regra operacional de bloqueio",
        ], "Uma quantia pode ter significados diferentes conforme o saldo. "
           "Testamos a razão entre o valor da transação e o módulo "
           "do saldo acrescido de um. Essa escolha evita divisão "
           "por zero. É uma variável construída para o experimento, "
           "não uma medida bancária universal de risco ou de utilização.",
           r"r=\frac{\mathrm{amount}}{|\mathrm{balance}|+1}",
           "Razão = valor da transação dividido pelo módulo do saldo mais um",
           "No dia a dia: o mesmo gasto em bolsos diferentes", [
               "Um gasto de 100 pesa de formas diferentes em dois orçamentos",
               "A relação entre valores pode trazer contexto",
               "O experimento decide se esse contexto melhora a previsão",
           ], "Imagine gastar cem reais com dez mil disponíveis "
              "ou com cento e poucos disponíveis. O número do gasto "
              "é o mesmo, mas a relação com o orçamento muda. "
              "A variável testa essa ideia sem transformar a analogia "
              "em uma regra de fraude.")
    if any(name.endswith("__cents_key") for name in all_names):
        pair("04 / Representar os dados", "Dígitos: investigar a geração sintética", [
            "São criadas categorias para valores arredondados e suas partes",
            "Centavos e dígitos iniciais oferecem outras visões do mesmo número",
            "Uma associação neste dataset não comprova uma regra de fraude real",
        ], "A pesquisa de soluções públicas sugeriu testar padrões de "
           "arredondamento e partes dos valores numéricos. Uma das "
           "visões arredonda cem vezes o valor absoluto e usa "
           "o resto da divisão por cem, isolando os centavos. "
           "Outras usam o valor arredondado e dígitos iniciais. "
           "São hipóteses sobre dados sintéticos. A associação "
           "precisa sobreviver à validação e não deve ser "
           "interpretada como lei de comportamento bancário.",
           r"c(x)=\mathrm{round}(100|x|)\;\mathrm{mod}\;100",
           "Categoria de centavos = arredondamento de cem vezes módulo de x, módulo cem",
           "No dia a dia: perceber o padrão de uma impressora", [
               "Uma máquina pode repetir marcas que passam despercebidas no valor completo",
               "Separar as partes ajuda a investigar essas marcas",
               "Um padrão da máquina pode desaparecer fora daquele processo",
           ], "Imagine uma impressora que deixa uma marca recorrente "
              "em seus documentos. Separar e observar pequenos detalhes "
              "pode revelar essa marca. Mas a marca diz algo sobre "
              "a impressora, não sobre todos os documentos do mundo. "
              "Os padrões numéricos de uma distribuição sintética "
              "exigem o mesmo cuidado.")
    if target_runs:
        encoders = [run["model_details"]["target_encoder"] for run in target_runs]
        smooth_values = ", ".join(sorted({str(encoder.get("smooth")) for encoder in encoders}))
        pair("04 / Representar os dados", "Target encoding com suavização", [
            "Cada categoria recebe uma estimativa baseada no alvo de treino",
            "A média da categoria é combinada com a média disponível no treino",
            "Suavização registrada: " + smooth_values,
            "Categoria desconhecida usa a média de referência do treino",
        ], "O target encoding representa uma categoria por uma "
           "estimativa do alvo baseada nos dados de treino. Para "
           "evitar confiar demais em categorias com poucos exemplos, "
           "a estimativa combina a média da categoria com a média "
           "de referência. O peso w depende da suavização. Quando "
           "o parâmetro é auto, a biblioteca usa uma estimativa "
           "empírica de Bayes. Isso é diferente de considerar "
           "toda média de categoria igualmente confiável.",
           r"\mathrm{TE}(c)=w_c\bar y_c+(1-w_c)\bar y,\quad 0\leq w_c\leq1",
           "Codificação da categoria = peso vezes média da categoria, mais um menos peso vezes média de referência",
           "No dia a dia: avaliar uma loja com poucas opiniões", [
               "Uma única avaliação excelente ainda traz pouca evidência",
               "A média geral funciona como uma referência inicial",
               "Com mais evidência, a avaliação específica ganha influência",
           ], "Pense em uma loja com uma única avaliação excelente. "
              "Não convém confiar nela como se fossem milhares "
              "de avaliações. A suavização mistura a informação "
              "específica com uma referência geral. Quanto confiar "
              "em cada parte é controlado pela regra de suavização.")
        inner_folds = sorted({encoder["internal_folds"] for encoder in encoders if "internal_folds" in encoder})
        pair("04 / Representar os dados", "Cross-fitting também dentro do treino", [
            "Divisões internas registradas: " + ", ".join(str(value) for value in inner_folds),
            "O próprio rótulo da linha não constrói sua codificação de treino",
            "Validação externa e teste usam mapeamentos ajustados no treino externo",
        ], "Codificar uma linha usando uma média que inclui sua "
           "própria resposta pode deixar a resposta escapar para "
           "a entrada, principalmente em categorias raras. O "
           "cross-fitting interno evita isso. O treino externo "
           "é dividido novamente, e cada parte recebe codificações "
           "calculadas nas demais. A validação externa e o teste "
           "continuam fora desse ajuste.",
           r"\mathrm{TE}_i=g_{D_{\mathrm{treino}}\setminus I_j}(c_i),\quad i\in I_j",
           "Codificação da linha i usa mapeamento ajustado no treino menos a divisão interna j que contém i",
           "No dia a dia: a resposta não pode estar na pista", [
               "Uma pista útil ajuda a resolver a pergunta",
               "Uma pista que contém a própria resposta vicia o teste",
               "Outra parte do treino precisa fornecer essa informação",
           ], "Imagine um exercício cuja pista revela a própria "
              "resposta. O aluno parece acertar, mas não precisou "
              "aprender. O cross-fitting impede esse atalho na "
              "codificação de treino: a pista de uma linha é "
              "construída sem consultar sua própria resposta.")
        if any(encoder.get("log_odds") for encoder in encoders):
            pair("04 / Representar os dados", "Transformar risco codificado em logit", [
                "A probabilidade codificada vira uma razão de chances em log",
                "A implementação limita valores extremos para evitar log de zero",
                "Os logits são escalonados no treino antes do modelo seguinte",
            ], "Alguns experimentos transformam a probabilidade "
               "codificada em logit: o log da razão entre p e um "
               "menos p. Valores próximos de zero ou um precisam "
               "ser limitados para evitar infinitos. Depois, esses "
               "logits são escalonados usando apenas estatísticas "
               "de treino. A transformação altera a representação, "
               "não cria novos rótulos.",
               r"\mathrm{logit}(p)=\log\frac{p}{1-p}",
               "Logit de p = log da razão entre p e um menos p",
               "No dia a dia: comparar duas possibilidades", [
                   "Uma chance de 80% corresponde a 4 para 1",
                   "O log transforma essa razão em uma escala aditiva",
                   "Uma representação diferente pode ajudar um modelo simples",
               ], "Imagine comparar a chance de acontecer algo "
                  "com a de não acontecer. Com oitenta por cento, "
                  "a relação é quatro para um. O logit coloca "
                  "essa razão em uma escala que pode ser somada "
                  "com outras pistas pelo modelo seguinte.")
    if any(run.get("model") in ("logistic", "glm") for run in runs):
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
    lightgbm_runs = [run for run in runs if run.get("model") == "lightgbm"]
    if lightgbm_runs:
        leaves = sorted({run["model_details"]["num_leaves"] for run in lightgbm_runs
                         if "num_leaves" in run.get("model_details", {})})
        pair("04 / Os modelos", "LightGBM: limitar as folhas de cada árvore", [
            "Limites de folhas registrados: " + ", ".join(str(value) for value in leaves),
            "Cada folha atribui uma contribuição aos exemplos de sua região",
            "Muitas árvores pequenas podem modelar relações sem árvores muito profundas",
        ], "O LightGBM também combina árvores em sequência. Cada "
           "árvore divide o espaço de características em regiões e "
           "atribui um valor a cada folha. O número de folhas limita "
           "sua complexidade. Quatro folhas não significam quatro "
           "camadas nem uma profundidade fixa. A escolha é avaliada "
           "contra outras configurações, em vez de presumir que "
           "uma árvore maior será melhor.",
           r"h(x)=\sum_{\ell=1}^{L}v_{\ell}\,\mathbf{1}[x\in R_{\ell}]",
           "Valor da árvore = soma do valor de cada folha multiplicado pelo indicador de que x pertence à sua região",
           "No dia a dia: uma triagem com poucas saídas", [
               "Uma sequência de perguntas encaminha cada caso a uma saída",
               "Cada saída tem uma contribuição para a avaliação",
               "Limitar as saídas impede uma regra diferente para cada detalhe",
           ], "Imagine uma triagem com algumas perguntas e poucas "
              "saídas possíveis. Cada saída acrescenta uma avaliação. "
              "Se criarmos uma saída para cada detalhe observado, "
              "podemos decorar os exemplos. Limitar as folhas testa "
              "uma forma mais simples de fazer essa triagem.")
    if any(run.get("model") == "glm" for run in runs):
        pair("04 / Os modelos", "GLM: combinar logits e categorias originais", [
            "Uma regressão logística recebe múltiplas representações das entradas",
            "Logits codificados e one-hot têm pesos aprendidos distintos",
            "O total de pesos depende da codificação, não só do número de colunas brutas",
        ], "O experimento GLM usa regressão logística sobre entradas "
           "numéricas e representações das categorias. As categorias "
           "codificadas por alvo passam pela transformação logit e "
           "pelo escalonamento. As categorias originais também têm "
           "uma representação one-hot. A combinação continua sendo "
           "uma soma ponderada antes da sigmoid. O total de pesos "
           "não é simplesmente o número de colunas do arquivo.",
           r"z=b+w_x^{\top}x+w_t^{\top}\mathrm{logit}(\mathrm{TE}(c))+w_o^{\top}\mathrm{OHE}(c)",
           "Logit final = viés mais contribuições das entradas numéricas, logits codificados e one-hot",
           "No dia a dia: uma ficha com mais de uma perspectiva", [
               "A mesma informação pode aparecer como medida, rótulo e histórico agregado",
               "A ficha aprende quanto cada perspectiva deve pesar",
               "Continuamos medindo se essas perspectivas melhoram a previsão",
           ], "Pense em uma ficha que reúne medidas, o tipo de item "
              "e um resumo do histórico de itens parecidos. Cada "
              "perspectiva pode ajudar de uma forma. A regressão "
              "aprende os pesos dessa ficha, e a validação decide "
              "se o conjunto trouxe ganho.")
        glm_details = next(run["model_details"] for run in runs if run.get("model") == "glm")
        pair("04 / Os modelos", "GLM: controlar o tamanho dos pesos", [
            f"C registrado: {glm_details.get('C', 'não registrado')}",
            "C menor representa maior força de regularização",
            "A penalidade L2 atua sobre os pesos; λ representa sua intensidade",
        ], "A regressão logística usa regularização para evitar "
           "pesos desnecessariamente grandes. A expressão mostra "
           "uma perda somada com uma penalidade quadrática. "
           "Lambda representa a intensidade da penalidade. No "
           "scikit-learn, C controla inversamente a força de "
           "regularização. A equivalência numérica exata depende "
           "da normalização da perda; não igualamos automaticamente "
           "lambda a um sobre C numa perda média.",
           r"\mathcal{J}(w)=\sum_i\ell(y_i,\hat p_i)+\lambda\|w\|_2^2",
           "Objetivo = soma das perdas mais lambda vezes norma quadrática dos pesos",
           "No dia a dia: evitar ajustes exagerados", [
               "Uma pequena evidência não deveria provocar uma correção enorme",
               "A penalidade desestimula pesos excessivos",
               "A força desse controle também precisa ser escolhida no desenvolvimento",
           ], "Imagine ajustar uma receita depois de provar "
              "uma colher. Um pequeno sinal de diferença não "
              "deveria levar a despejar um pacote inteiro de "
              "tempero. A regularização desestimula ajustes "
              "exagerados nos pesos sem impedir o modelo de aprender.")
        if glm_details.get("solver") == "newton-cholesky":
            pair("04 / Os modelos", "GLM: o passo de Newton com Cholesky", [
                "g é o gradiente; H é a matriz de curvaturas da perda",
                "Cholesky fatoriza H para resolver o sistema do passo Δ",
                "A atualização do GLM é diferente do AdamW usado na rede",
            ], "O solver registrado do GLM é Newton Cholesky. "
               "Além do gradiente, ele utiliza a Hessiana, que "
               "reúne informações de curvatura da função de perda. "
               "O passo é obtido resolvendo H vezes delta igual "
               "ao gradiente. A fatoração de Cholesky permite "
               "resolver esse sistema por fatores triangulares. "
               "Os parâmetros são então atualizados com o passo "
               "de Newton. Esse solver não é o AdamW da MLP.",
               r"H=RR^{\top},\quad H\Delta=g,\quad\theta_{t+1}=\theta_t-\Delta",
               "Hessiana = R vezes R transposta; Hessiana vezes passo = gradiente; novo parâmetro = parâmetro menos passo",
               "No dia a dia: considerar a inclinação e a curva", [
                   "A inclinação sugere para onde descer",
                   "A curvatura ajuda a dimensionar o próximo passo",
                   "O sistema matemático combina essas informações",
               ], "Imagine descer até o fundo de uma superfície "
                  "curva. A inclinação aponta a direção, mas "
                  "a forma da curva ajuda a decidir o tamanho "
                  "do passo. Newton usa informações equivalentes "
                  "de inclinação e curvatura. Cholesky é a "
                  "maneira de resolver o sistema que calcula esse passo.")
    if neural:
        hidden = architecture.get("hidden_dims", [])
        hidden_text = " → ".join(str(size) for size in hidden)
        neural_title = ("MLP-TE: a arquitetura treinada" if architecture.get("categorical_transform") == "target"
                        else "MLP: a arquitetura treinada")
        add("05 / A rede neural", neural_title, [
            f"Valores numéricos de entrada da rede: {number(architecture.get('numeric_features'))}",
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
        if any(run.get("model") == "mlp" and "standard scaling" in
               str(run.get("model_details", {}).get("preprocessing", "")).lower() for run in runs):
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
        if any(run.get("model_details", {}).get("numeric_transform") == "quantile" for run in runs):
            pair("05 / A rede neural", "Quantis: outra preparação numérica", [
                "A posição acumulada é estimada no treino de cada divisão",
                "Essa posição é mapeada para quantis de uma distribuição normal",
                "A transformação preserva a ordem, mas muda as distâncias",
            ], "Outra variante usa transformação por quantis. "
               "Primeiro estimamos a posição acumulada de cada "
               "valor na distribuição do treino. Depois aplicamos "
               "a função inversa da distribuição normal. A fórmula "
               "resume esse caminho. É uma transformação não linear "
               "e valores fora da faixa observada são limitados "
               "pelas regras da implementação.",
               r"x'_j=\Phi^{-1}\left(\widehat F_{j,\mathrm{treino}}(x_j)\right)",
               "Entrada transformada = quantil normal correspondente à posição acumulada estimada no treino",
               "No dia a dia: comparar a posição numa fila", [
                   "Em vez do valor absoluto, observamos a posição relativa",
                   "Essa posição é transferida para outra escala de referência",
                   "A ordem fica; a distância entre os valores pode mudar",
               ], "Imagine comparar a posição de uma pessoa em "
                  "uma fila em vez de sua distância exata da porta. "
                  "Depois colocamos essa posição numa outra escala. "
                  "A primeira continua antes da segunda, mas a "
                  "distância entre as duas pode mudar.")
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
        if any(run.get("model") == "mlp" and run.get("model_details", {}).get("architecture", {}).get("categorical_cardinalities")
               for run in runs):
            pair("05 / A rede neural", "Embeddings para categorias", [
                "Usados nas variantes com embeddings; a MLP-TE usa logits codificados",
                "Cada categoria é associada a um vetor aprendido",
                "A dimensão vem da configuração do treinamento",
            ], "As variantes iniciais com embeddings atribuem "
               "vetores aprendidos às categorias. A MLP com target "
               "encoding usa logits codificados no lugar desses embeddings. "
               "Uma categoria, como uma classe de estabelecimento quando essa variável "
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
        if not any(run.get("model") in ("logistic", "glm") for run in runs):
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
    for component in external_components:
        slides.extend(Slide(**specification) for specification in gpt2_slides(component))
    if selected.get("model") == "blend":
        weights = selected.get("model_details", {}).get("weights", [])
        source_names = [str(path).replace("\\", "/").rsplit("/", 1)[-1]
                        for path in selected.get("model_details", {}).get("sources", [])]
        composition = "; ".join(f"{number(weight * 100)}% {name}" for weight, name in zip(weights, source_names))
        effective = (evidence.get("run") or {}).get("effective_weights", {})
        if effective:
            labels = {"glm": "GLM", "lightgbm": "LightGBM", "external_gpt2": "GPT-2 externo"}
            composition = "; ".join(f"{number(weight * 100, 2)}% {labels.get(name, name)}"
                                    for name, weight in effective.items())
        pair("07 / Os experimentos", "Combinar previsões", [
            "A combinação usa previsões compatíveis por id e divisão",
            "Pesos não negativos somam 1",
            composition or "O ganho deve ser comparado com os modelos isolados",
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
        own_source = next((run for run in development_runs if run_label(run) in source_names and
                           run.get("model") == "blend" and not components_from([run])), None)
        if components_from([selected]) and own_source:
            improvement = selected["roc_auc"] - own_source["roc_auc"]
            paired = list(zip(selected.get("fold_scores", []), own_source.get("fold_scores", [])))
            improved = sum(candidate > baseline for candidate, baseline in paired)
            add("07 / Os experimentos", "A fonte complementar trouxe ganho medido?", [
                f"Base própria: {number(own_source['roc_auc'], 5)}; mistura: {number(selected['roc_auc'], 5)}",
                f"Ganho OOF observado: {number(improvement, 5)}",
                f"Melhora em {improved} de {len(paired)} folds alinhados",
                "É evidência local usada na seleção, sem uma reserva independente nova",
            ], "A composição foi comparada com a base própria "
               "no mesmo protocolo. A base teve AUC " + number(own_source["roc_auc"], 5) +
               ", e a mistura teve " + number(selected["roc_auc"], 5) + ". "
               "O ganho local foi " + number(improvement, 5) + ", com melhora em " +
               str(improved) + " dos " + str(len(paired)) + " folds. "
               "Essa consistência motivou a escolha, mas os "
               "folds também foram usados para selecionar "
               "a composição. Portanto não chamamos esse "
               "resultado de avaliação independente final.")
    protocols = {}
    for run in development_runs:
        contract = run.get("validation", {})
        key = (str(contract.get("method", "não verificado")).lower().replace("stratifiedkfold", "stratified"),
               contract.get("n_splits"), run.get("seed"), run.get("rows", {}).get("oof"), contract.get("holdout_fraction", 0))
        protocols.setdefault(key, []).append(run)
    if not protocols:
        protocols[("não verificado", None, None, None, None)] = []
    for protocol, members in protocols.items():
        scores = [f"{run_label(run)}{' [fonte externa]' if run.get('model') == 'external_gpt2' else ''}: AUC OOF {number(run.get('roc_auc'), 5)}"
                  for run in members]
        for start in range(0, max(1, len(scores)), 5):
            group = scores[start:start + 5]
            title = f"Resultados: {number(protocol[1])} folds, {number(protocol[3])} OOF" if members else "Resultados locais registrados"
            add("07 / Os experimentos", title, group or ["Nenhum resultado de treinamento verificado nesta versão"],
                "Os números vêm dos arquivos de métricas fornecidos. " +
                (". ".join(group).replace("-", " ") + ". " if group else "Não há resultados neste rascunho. ") +
                "As comparações são agrupadas por protocolo. AUC de três folds "
                "com uma reserva separada e AUC de cinco folds sobre todos "
                "os rótulos não são a mesma avaliação. A fonte externa foi "
                "auditada e sua métrica reproduzida, sem executar seu treinamento.")
    raw_logistic = next((run for run in development_runs if run.get("model") == "logistic" and
                         run.get("model_details", {}).get("feature_set", "raw") == "raw"), None)
    basic_neural = next((run for run in development_runs if run.get("model") == "mlp" and
                         run.get("model_details", {}).get("feature_set", "raw") == "raw" and
                         run.get("model_details", {}).get("numeric_transform", "standard") == "standard"), None)
    if raw_logistic and basic_neural:
        difference = basic_neural["roc_auc"] - raw_logistic["roc_auc"]
        add("07 / Os experimentos", "A rede básica superou a referência?", [
            f"Regressão logística: {number(raw_logistic['roc_auc'], 5)}",
            f"MLP básica: {number(basic_neural['roc_auc'], 5)}",
            "Diferença MLP − referência: " + number(difference, 5),
            "Não superou nesta execução" if difference <= 0 else "Superou nesta execução",
        ], "A rede básica e a referência simples precisam ser comparadas "
           "com honestidade. A regressão logística obteve " + number(raw_logistic["roc_auc"], 5) +
           ", e a MLP básica obteve " + number(basic_neural["roc_auc"], 5) + ". "
           + ("A rede não superou a referência neste experimento. " if difference <= 0 else
              "A rede superou a referência neste experimento. ") +
           "Uma diferença pequena, diante da variação entre divisões, "
           "não prova superioridade geral de uma família de modelos. "
           "Mais flexibilidade não garante melhor ordenação, e este "
           "experimento não isolou uma causa única para a diferença.")
    raw_tree = next((run for run in development_runs if run.get("model") == "lightgbm" and
                     run.get("model_details", {}).get("feature_set") == "raw"), None)
    digits_tree = next((run for run in development_runs if run.get("model") == "lightgbm" and
                        run.get("model_details", {}).get("feature_set") == "digits"), None)
    if raw_tree and digits_tree:
        delta = digits_tree["roc_auc"] - raw_tree["roc_auc"]
        pair("07 / Os experimentos", "Ablação: a visão de dígitos ajudou?", [
            f"LightGBM raw: {number(raw_tree['roc_auc'], 5)}",
            f"LightGBM digits: {number(digits_tree['roc_auc'], 5)}",
            f"Diferença observada: {number(delta, 5)}",
            "A interpretação depende de manter os demais fatores da comparação",
        ], "Uma ablação pergunta o que muda ao alterar uma parte "
           "do procedimento. Os registros raw e digits do LightGBM "
           "permitem examinar a representação adicionada. A "
           "diferença de AUC observada foi " + number(delta, 5) + ". "
           "Esse número só pode ser atribuído à representação "
           "quando dados, divisões e demais configurações são "
           "mantidos na comparação. Ele não demonstra uma "
           "regra universal sobre os dígitos das transações.",
           r"\Delta\mathrm{AUC}=\mathrm{AUC}_{\mathrm{digits}}-\mathrm{AUC}_{\mathrm{raw}}",
           "Diferença de AUC = AUC com dígitos menos AUC sem essa representação",
           "No dia a dia: mudar uma coisa na receita", [
               "Para testar um ingrediente, preserve o restante da receita",
               "Se vários ingredientes mudarem, o resultado tem causas misturadas",
               "Variações combinadas precisam ser descritas como combinadas",
           ], "Imagine testar se um ingrediente melhora uma receita. "
              "Se você também muda o forno e o tempo, não sabe "
              "qual mudança trouxe o efeito. A ablação procura "
              "controlar essa comparação. Uma variante que troca "
              "representações e modelo ao mesmo tempo continua "
              "sendo útil, mas não isola o efeito de uma única parte.")
    if selected and not is_refit(selected):
        add("07 / Os experimentos", "O resultado selecionado e seus limites", [
            f"Modelo selecionado: {selected.get('model', 'não verificado')}",
            f"AUC OOF: {number(selected.get('roc_auc'), 5)}; desvio entre divisões: {number(selected.get('fold_std'), 5)}",
            f"Tempo desta etapa registrada: {number(selected.get('elapsed_seconds'), 1)} segundos",
            "Score local e score público medem amostras diferentes",
        ], "A execução selecionada é " + str(selected.get("model")) +
           ". Seu resultado local é uma estimativa sobre as divisões usadas. "
           "O desvio entre divisões descreve a variação observada, não "
           "um intervalo de confiança automático. Em uma composição, "
           "o tempo dessa etapa mede combinar artefatos e não "
           "treinar todas as fontes. Uma pontuação maior no "
           "ranking público pode não se repetir na parte privada do teste.")
    if is_refit(selected):
        final_composition = ("Um GLM + um LightGBM, com composição congelada" if
                             selected.get("model") == "blend" and
                             {run.get("model") for run in runs if is_refit(run)}.issuperset({"glm", "lightgbm"})
                             else "A configuração foi escolhida e congelada antes do refit")
        pair("07 / Os experimentos", "Refit final: aprender novamente com todos os rótulos", [
            f"Registros usados no refit: {number(rows.get('train'))}",
            final_composition,
            "Os pesos são aprendidos novamente; os hiperparâmetros ficam definidos",
            "Este refit não produz uma nova AUC OOF ou independente",
        ], "Depois da seleção e da conferência registradas, o "
           "refit usa todos os rótulos disponíveis. A configuração "
           "é mantida e os parâmetros do modelo são aprendidos "
           "novamente. Quando existe um número de épocas ou "
           "árvores a fixar, ele vem da regra registrada no "
           "desenvolvimento. A reserva avaliou médias dos checkpoints "
           "de desenvolvimento. Este arquivo usa modelos aprendidos "
           "novamente com todos os rótulos. Por isso nem a AUC OOF "
           "nem a AUC da reserva são apresentadas como uma nova "
           "medição deste arquivo. Sua avaliação externa vem do Kaggle.",
           r"\theta_{\mathrm{final}}=\mathrm{Train}(D_{\mathrm{todo}};h^*)",
           "Parâmetros finais = treinamento com todos os dados rotulados sob hiperparâmetros já escolhidos",
           "No dia a dia: seguir a receita escolhida", [
               "Os testes ajudaram a escolher a receita",
               "Agora usamos os ingredientes disponíveis seguindo essa escolha",
               "Preparar uma porção maior não cria uma nova avaliação independente",
           ], "Pense em escolher uma receita depois de alguns "
              "testes e uma conferência separada. Na produção "
              "seguinte, você usa os ingredientes disponíveis "
              "e segue a escolha definida. Isso não transforma "
              "o resultado dos testes anteriores em uma nova "
              "avaliação independente da porção maior.")
    selection_reason = (evidence.get("run") or {}).get("selection_reason")
    if selection_reason:
        add("07 / Os experimentos", "Por que esta solução foi submetida", [
            f"Execução selecionada: {run_label(selected)}",
            str(selection_reason),
            "A justificativa é vinculada aos experimentos e à seleção registrada",
        ], "A solução submetida foi escolhida pelo seguinte motivo: "
           + str(selection_reason) + " A explicação deve ser lida junto "
           "das métricas e das limitações, incluindo os métodos "
           "que não melhoraram a referência. A presença de uma "
           "rede neural no estudo não justifica declarar um "
           "ganho que não foi observado.")
    if results.get("holdout_auc") is not None:
        held_rows = (holdout or {}).get("rows")
        positive_rows = (holdout or {}).get("positive_rows")
        held_sources = {str(path).replace("\\", "/").rsplit("/", 1)[-1]
                        for path in (holdout or {}).get("selection", {}).get("sources", [])}
        checkpoint_counts = {run.get("validation", {}).get("n_splits") for run in development_runs
                             if run_label(run) in held_sources}
        checkpoint_note = (f"Média de {next(iter(checkpoint_counts))} checkpoints de cada família" if
                           len(checkpoint_counts) == 1 else "Média dos checkpoints de cada família")
        held_title = "Reserva histórica do protocolo inicial" if historical_holdout else "O que exatamente a reserva avaliou"
        held_limit = ("Não é independente para o GPT-2 nem para a nova composição" if historical_holdout
                      else "Esse score não é uma avaliação do refit final")
        held_explanation = ("Este resultado pertence ao protocolo inicial. O componente "
                            "neural externo usou parte dessas linhas no treinamento "
                            "de seus folds. A antiga reserva não oferece uma avaliação "
                            "independente da nova composição. O protocolo novo é "
                            "descrito por seus cinco folds alinhados e pelo Kaggle. "
                            if historical_holdout else "")
        held_intro = ("No protocolo inicial, foi registrada uma avaliação "
                      "da reserva. " if historical_holdout else
                      "Além das métricas usadas no desenvolvimento, foi fornecido "
                      "um registro de avaliação da reserva. ")
        pair("07 / Os experimentos", held_title, [
            f"Reserva: {number(held_rows)} linhas; {number(positive_rows)} positivas",
            f"AUC do protocolo de desenvolvimento: {number(results['holdout_auc'], 5)}",
            checkpoint_note + "; composição congelada antes da consulta",
            held_limit,
        ], held_intro + "Ela contém "
           + number(held_rows) + " linhas, com " + number(positive_rows) + " positivas. "
           "A AUC do protocolo foi " + number(results["holdout_auc"], 5) + ". "
           + checkpoint_note + " de desenvolvimento. "
           "Essas médias foram combinadas pelos pesos congelados. "
           "Naquele procedimento, a reserva não entrou no ajuste. O refit "
           "posterior usa modelos diferentes, treinados com todos "
           "os rótulos, e não recebeu uma nova avaliação nessa reserva. " + held_explanation,
           r"\hat p_{\mathrm{reserva}}=\sum_jw_j\left(\frac{1}{K_j}\sum_{k=1}^{K_j}p_{j,k}\right)",
           "Previsão da reserva = soma dos pesos de cada família vezes a média de seus checkpoints",
           "No dia a dia: conferir a equipe do ensaio", [
               "A conferência observou uma equipe que combinava tentativas dos ensaios",
               "Depois, uma nova execução segue a receita escolhida",
               "O resultado da conferência não muda de dono automaticamente",
           ], "Imagine conferir o trabalho de uma equipe que "
              "combina várias tentativas feitas nos ensaios. "
              "Depois uma nova execução segue a receita definida. "
              "A conferência anterior continua sendo evidência "
              "sobre o processo testado. Ela não se transforma "
              "automaticamente em uma avaliação daquela nova execução.")
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
        f"Consulta: {observation_time(results.get('observed_at'))}",
    ], "A prova de participação é a submissão processada pela plataforma. "
       + (f"O registro fornecido informa AUC pública de {number(results.get('public_auc'), 5)} "
          f"e posição {number(results.get('ranking'))}, observadas em {observation_time(results.get('observed_at'), spoken=True)}. "
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
       "As conclusões se referem a estes dados e a este procedimento "
       "de avaliação. A comparação entre modelos, o efeito observado "
       "das representações e a posição datada permitem avaliar "
       "o trabalho sem atribuir ao modelo uma capacidade que "
       "o experimento não mediu.")
    sources = SOURCES.copy()
    if target_runs:
        sources.append(("TargetEncoder", "https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.TargetEncoder.html"))
    if any(run.get("model") == "glm" for run in runs):
        sources.append(("Regressão logística e Newton-Cholesky", "https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html"))
    if lightgbm_runs:
        sources.append(("LightGBM", "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html"))
    if any(run.get("model_details", {}).get("numeric_transform") == "quantile" for run in runs):
        sources.append(("QuantileTransformer", "https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.QuantileTransformer.html"))
    if any(name.endswith("__cents_key") for name in all_names):
        sources.append(("Pesquisa e atribuição das ideias", "docs/research.md no repositório"))
    if external_components:
        sources.extend([
            ("GPT-2 e atenção causal", "https://huggingface.co/docs/transformers/model_doc/gpt2"),
            ("Transformer", "https://arxiv.org/abs/1706.03762"),
            ("GPT-2: arquitetura e normalização", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
        ])
        for component in external_components:
            sources.append(("Código externo: " + str(component.get("trained_by")), str(component.get("training_code_url"))))
            sources.append(("Previsões externas e licença", str(component.get("predictions_source_url"))))
        sources.extend([
            ("GReaT: Borisov e coautores", "https://arxiv.org/abs/2210.06280"),
            ("Receita target-first: Chris Deotte", "https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754683"),
            ("Licença das previsões externas", "https://creativecommons.org/licenses/by/4.0/"),
        ])
    for start in range(0, len(sources), 3):
        group = sources[start:start + 3]
        add("Referências", "Fontes e próximos pontos de inspeção", [name + ": " + url for name, url in group],
            "Este grupo reúne as referências de " + "; ".join(name for name, _ in group) +
            ". Os links completos estão no slide e no repositório.")
    ensure_pairs(slides)
    return slides
