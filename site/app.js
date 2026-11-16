"use strict";

const select = (selector) => document.querySelector(selector);
const text = (selector, value) => { const element = select(selector); if (element) element.textContent = value; };
const validScore = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
const score = (value) => value.toLocaleString("pt-BR", { minimumFractionDigits: 5, maximumFractionDigits: 5 });

function dateLabel(value, includeTime = false) {
  if (!value) return null;
  const date = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T12:00:00Z` : value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("pt-BR", {
    timeZone: "America/Sao_Paulo", day: "2-digit", month: "2-digit", year: "numeric",
    ...(includeTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  }).format(date);
}

function safeLink(value) {
  if (typeof value !== "string" || !value.trim() || value.startsWith("//")) return null;
  try {
    const link = new URL(value, location.href);
    if (!(["http:", "https:"].includes(link.protocol))) return null;
    if (link.origin !== location.origin && link.protocol !== "https:") return null;
    return link;
  } catch { return null; }
}

function setLinks(links) {
  document.querySelectorAll("[data-link]").forEach((anchor) => {
    const link = safeLink(links[anchor.dataset.link]);
    if (!link) return;
    anchor.href = link.href;
    anchor.hidden = false;
    if (link.origin !== location.origin) {
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
    }
  });
  if (safeLink(links.github)) select("[data-code-status]").hidden = true;
  if (safeLink(links.presentation)) text("[data-slides-status]", "Slides disponíveis para consulta.");
  if (safeLink(links.video)) text("[data-video-status]", "Vídeo narrado disponível para consulta.");
  const video = safeLink(links.video);
  if (video && /\.mp4$/i.test(video.pathname)) {
    select("[data-video]").src = video.href;
    select("[data-video-container]").hidden = false;
    const captions = safeLink(links.captions);
    if (captions) select("[data-captions]").src = captions.href;
    else select("[data-captions]").remove();
  }
}

function listItems(selector, items, verified = []) {
  const list = select(selector);
  list.replaceChildren();
  items.forEach((content, index) => {
    const item = document.createElement("li");
    item.textContent = content;
    if (verified[index]) item.className = "verified";
    list.append(item);
  });
}

function renderFolds(folds) {
  if (!Array.isArray(folds) || !folds.length || !folds.every(validScore)) return;
  const chart = select("[data-fold-chart]");
  chart.replaceChildren();
  folds.forEach((value, index) => {
    const row = document.createElement("div");
    row.className = "fold-row";
    const label = document.createElement("span");
    label.textContent = `Fold ${index + 1}`;
    const track = document.createElement("div");
    track.className = "fold-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("div");
    bar.className = "fold-bar";
    bar.style.width = `${value * 100}%`;
    track.append(bar);
    const number = document.createElement("span");
    number.className = "fold-value";
    number.textContent = score(value);
    row.append(label, track, number);
    chart.append(row);
  });
  const axis = document.createElement("div");
  axis.className = "fold-axis";
  const start = document.createElement("span");
  start.textContent = "0";
  const end = document.createElement("span");
  end.textContent = "1 · ROC-AUC";
  axis.append(start, end);
  chart.append(axis);
  select("[data-fold-panel]").hidden = false;
}

function render(evidence) {
  const competition = evidence.competition || {};
  const checks = evidence.checks || {};
  const results = evidence.results || {};
  const run = evidence.run || {};
  const links = { ...(evidence.links || {}) };
  if (!links.kaggle) links.kaggle = competition.url;
  setLinks(links);
  const materials = evidence.materials || {};
  if (materials.presentation_status === "draft" && safeLink(links.presentation)) {
    text('[data-link="presentation"]', "Abrir prévia dos slides ↗");
    text("[data-slides-status]", "Prévia de conteúdo · experimentos e resultados pendentes.");
  }
  if (materials.video_status === "review" && safeLink(links.video)) {
    text("[data-video-status]", "Vídeo gerado · revisão audiovisual pendente.");
  }
  if (competition.title) text("[data-competition-title]", competition.title.replace(/\s*[·—]\s*Playground.*$/i, ""));
  const deadline = dateLabel(competition.deadline_utc, true);
  if (deadline) text("[data-deadline]", `${deadline} · SP`);
  const stages = {
    planning: "Estruturação do experimento", implementation: "Implementação em andamento",
    data: "Preparação dos dados", training: "Experimentos em andamento",
    evaluated: "Validação local registrada", submitted: "Submissão registrada",
    published: "Materiais publicados", blocked: "Execução com pendência de acesso",
  };
  text("[data-stage]", stages[evidence.stage] || "Experimento em desenvolvimento");
  const updated = dateLabel(evidence.updated_at);
  text("[data-updated]", updated ? `Registro atualizado em ${updated}` : "Registro sem data informada");

  if (checks.training_verified === true && validScore(results.local_auc)) {
    text("[data-local-auc]", score(results.local_auc));
    const validation = typeof run.validation === "string" ? run.validation : run.validation?.method;
    const folds = typeof run.validation?.n_splits === "number" ? `${run.validation.n_splits} folds` : null;
    text("[data-local-context]", [run.model, validation, folds].filter(Boolean).join(" · ") || "Validação local registrada");
    renderFolds(results.fold_scores);
  }
  if (checks.submission_verified === true && validScore(results.public_auc)) {
    text("[data-public-auc]", score(results.public_auc));
    text("[data-public-context]", results.submission_id ? `Submissão ${results.submission_id}` : "Avaliação pública registrada");
  }
  if (checks.submission_verified === true && Number.isInteger(results.ranking) && results.ranking > 0 && results.observed_at) {
    text("[data-ranking]", `${results.ranking.toLocaleString("pt-BR")}º`);
    const observed = dateLabel(results.observed_at, true);
    const teams = Number.isInteger(results.total_teams) && results.total_teams >= results.ranking ? `Entre ${results.total_teams.toLocaleString("pt-BR")} equipes` : "Ranking público provisório";
    text("[data-ranking-context]", `${teams}${observed ? ` · ${observed} (SP)` : ""}`);
  }
  const checkEntries = [
    [checks.data_verified === true, "Dados oficiais conferidos", "Dados oficiais: verificação pendente"],
    [checks.training_verified === true, "Treinamento e validação conferidos", "Treinamento e validação: pendentes"],
    [checks.submission_verified === true, "Submissão real conferida no Kaggle", "Submissão real no Kaggle: pendente"],
  ];
  listItems("[data-checks]", checkEntries.map(([done, yes, no]) => done ? yes : no), checkEntries.map(([done]) => done));
  const limitations = Array.isArray(evidence.limitations) ? evidence.limitations.filter((item) => typeof item === "string" && item.trim()) : [];
  listItems("[data-limitations]", [...limitations, "A pontuação pública e a posição podem mudar até a classificação final."]);
}

fetch("evidence.json", { cache: "no-store" })
  .then((response) => { if (!response.ok) throw new Error("Evidência indisponível"); return response.json(); })
  .then(render)
  .catch(() => {
    text("[data-stage]", "Evidências indisponíveis");
    text("[data-updated]", "Registro não carregado");
    listItems("[data-checks]", ["O arquivo de evidências não pôde ser consultado."]);
    select("[data-load-error]").hidden = false;
  });
