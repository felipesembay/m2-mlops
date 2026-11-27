const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(d).replace(".", ","));
const MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
const mesLabel = (p) => { const [a, m] = p.split("-"); return `${MESES[+m - 1]}/${a.slice(2)}`; };
const mesLongo = (p) => { const [a, m] = p.split("-"); return `${MESES[+m - 1]}/${a}`; };
const reais = (milhares) => `R$ ${fmt(milhares / 1e9, 2)} tri`; // SGS 27810 em R$ mil
const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); });

Chart.defaults.font.family = "system-ui, sans-serif";
Chart.defaults.color = css("--muted");
Chart.defaults.borderColor = css("--line");

const H = [1, 3, 6, 12];
const CORES_H = { 1: css("--c1"), 3: "#d9822b", 6: "#2a9d6f", 12: "#8e5bd6" };
const NOMES = { STL_ETS: "STL + ETS", RF_Bayesian: "Random Forest (OB)", SARIMA_300_011: "SARIMA", ETS_ANA: "ETS(A,N,A)", Media_Sazonal: "Média sazonal" };

const estado = { serie: [], prev: null, mon: null, mods: null, modelo: "melhor", meses: 12, periodo: "tudo", hHist: 1, hMon: 1 };
const charts = {};
function desenha(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart(document.getElementById(id), cfg); }

function tabs(id, opcoes, ativo, onClick) {
  const box = document.getElementById(id);
  box.innerHTML = opcoes.map(([v, t]) => `<button data-v="${v}" class="${String(v) === String(ativo) ? "ativo" : ""}">${t}</button>`).join("");
  box.querySelectorAll("button").forEach((b) => (b.onclick = () => onClick(b.dataset.v)));
}

// =================================================================== início
async function main() {
  const [serie, prev, mods, mon] = await Promise.all([
    get("/api/serie"), get("/api/previsoes"), get("/api/modelos"), get("/api/monitoramento"),
  ]);
  Object.assign(estado, { serie, prev, mods, mon });
  statusGeral();
  controlesProjecao();
  await renderProjecao();
  renderSerie();
  renderHist();
  renderMon();
  renderModelos();
  renderEventos();
}

function statusGeral() {
  const estados = Object.values(estado.mon.atual).map((a) => a.status);
  const geral = estados.includes("RETREINO") ? "RETREINO" : estados.includes("ALERTA") ? "ALERTA" : estados.length ? "OK" : "—";
  const pill = document.getElementById("status-geral");
  pill.textContent = `Monitoramento: ${geral}`;
  pill.className = `pill ${geral}`;
}

// ================================================================= projeção
function controlesProjecao() {
  const sel = document.getElementById("sel-modelo");
  const inp = document.getElementById("inp-meses");
  sel.onchange = () => { estado.modelo = sel.value; renderProjecao(); };
  inp.onchange = () => {
    estado.meses = Math.max(1, Math.min(24, +inp.value || 12));
    inp.value = estado.meses;
    tabsMeses();
    renderProjecao();
  };
  tabsMeses();
}

function tabsMeses() {
  tabs("tabs-meses", [1, 3, 6, 12, 24].map((m) => [m, m]), estado.meses, (v) => {
    estado.meses = +v;
    document.getElementById("inp-meses").value = v;
    tabsMeses();
    renderProjecao();
  });
}

const EXPLICA = {
  melhor: "Cada mês usa o modelo que venceu no artigo para aquela distância: mês 1 → STL+ETS (h=1); meses 2–3 → Random Forest (h=3); meses 4–6 → SARIMA (h=6); meses 7+ → Random Forest (h=12).",
  1: "Usa o STL+ETS (campeão de h=1) em todos os meses. Ele foi validado só para 1 mês à frente.",
  3: "Usa o Random Forest com os hiperparâmetros de h=3, reajustado para cada distância (previsão direta).",
  6: "Usa o SARIMA(3,0,0)(0,1,1)12 (campeão de h=6) em todos os meses.",
  12: "Usa o Random Forest com os hiperparâmetros de h=12, reajustado para cada distância (previsão direta).",
  benchmark: "Média histórica do mesmo mês do calendário nos últimos 120 meses: o benchmark do artigo.",
};

async function renderProjecao() {
  const box = document.getElementById("proj-resumo");
  box.innerHTML = "<p class='nota'>Calculando…</p>";
  let p;
  try {
    p = await get(`/api/projecao?modelo=${estado.modelo}&meses=${estado.meses}`);
  } catch (e) {
    box.innerHTML = `<p class='nota'>Erro ao projetar: ${e.message}</p>`;
    return;
  }
  const ult = p.projecao[p.projecao.length - 1];
  document.getElementById("proj-origem").textContent =
    `Último dado observado: ${mesLongo(p.origem)} (M2 = ${reais(p.nivel_origem)}). Projeção de ${mesLongo(p.projecao[0].data_alvo)} a ${mesLongo(ult.data_alvo)}.`;
  document.getElementById("proj-explica").textContent =
    EXPLICA[estado.modelo] + (estado.meses > 12 ? " Meses além de 12 ficam fora do que foi validado no artigo." : "");

  box.innerHTML = `
    <div class="card"><div class="h">Crescimento acumulado</div><div class="v">${fmt(p.crescimento_acumulado_pct)}%</div>
      <div class="m">em ${p.meses} ${p.meses === 1 ? "mês" : "meses"}</div></div>
    <div class="card"><div class="h">Média mensal</div><div class="v">${fmt(p.media_mensal_pct)}%</div>
      <div class="m">média das ${p.meses} previsões</div></div>
    <div class="card"><div class="h">M2 projetado em ${mesLabel(ult.data_alvo)}</div><div class="v">${reais(ult.nivel_m2)}</div>
      <div class="m">hoje: ${reais(p.nivel_origem)}</div></div>
    <div class="card"><div class="h">Próximo mês (${mesLabel(p.projecao[0].data_alvo)})</div><div class="v">${fmt(p.projecao[0].previsto)}%</div>
      <div class="m">faixa ${fmt(p.projecao[0].li)} a ${fmt(p.projecao[0].ls)}%</div></div>`;

  // gráfico: 36 meses observados + projeção com faixa
  const obs = estado.serie.slice(-36);
  const labels = [...obs.map((d) => d.data), ...p.projecao.map((x) => x.data_alvo)];
  const nObs = obs.length;
  const pad = (arr) => [...Array(nObs - 1).fill(null), obs[nObs - 1].m2_growth, ...arr];
  desenha("chart-proj", {
    type: "line",
    data: {
      labels: labels.map(mesLabel),
      datasets: [
        { label: "Observado", data: [...obs.map((d) => d.m2_growth), ...p.projecao.map(() => null)],
          borderColor: css("--real"), borderWidth: 1.8, pointRadius: 2 },
        { label: "Limite inferior", data: pad(p.projecao.map((x) => x.li)), borderWidth: 0, pointRadius: 0 },
        { label: "Faixa indicativa 95%", data: pad(p.projecao.map((x) => x.ls)), borderWidth: 0, pointRadius: 0,
          fill: "-1", backgroundColor: css("--band") },
        { label: "Projeção", data: pad(p.projecao.map((x) => x.previsto)), borderColor: css("--c1"), borderWidth: 2.2,
          pointRadius: (c) => (c.dataIndex >= nObs ? 4 : 0),
          pointBackgroundColor: (c) => { const x = p.projecao[c.dataIndex - nObs]; return x && x.h_modelo ? CORES_H[x.h_modelo] : css("--c1"); } },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { y: { title: { display: true, text: "% m/m" } }, x: { ticks: { maxTicksLimit: 16 } } },
      plugins: {
        legend: { labels: { filter: (i) => i.text !== "Limite inferior" } },
        tooltip: {
          filter: (i) => i.raw !== null && i.dataset.label !== "Limite inferior",
          callbacks: {
            label: (c) => `${c.dataset.label}: ${fmt(c.raw)}%`,
            afterBody: (items) => {
              const x = p.projecao[items[0].dataIndex - nObs];
              return x ? [`Modelo: ${x.modelo_nome}${x.h_modelo ? ` (h=${x.h_modelo})` : ""}`] : [];
            },
          },
        },
      },
    },
  });

  document.querySelector("#tab-proj tbody").innerHTML = p.projecao.map((x) => `<tr>
    <td>${mesLongo(x.data_alvo)}</td><td class="num">${fmt(x.previsto)}%</td>
    <td class="num">${fmt(x.li)} a ${fmt(x.ls)}%</td>
    <td>${x.modelo_nome}${x.h_modelo ? ` <span class="nota">h=${x.h_modelo}</span>` : ""}${x.fora_validacao ? ' <span class="nota">(fora da validação)</span>' : ""}</td>
    <td class="num">${fmt(x.acumulado_pct)}%</td><td class="num">${reais(x.nivel_m2)}</td></tr>`).join("");
}

// ==================================================================== série
function renderSerie() {
  const s = estado.serie;
  tabs("tabs-periodo", [["5", "5 anos"], ["10", "10 anos"], ["tudo", "Tudo"]], estado.periodo, (v) => { estado.periodo = v; renderSerie(); });
  const dados = estado.periodo === "tudo" ? s : s.slice(-12 * +estado.periodo);
  const media = dados.reduce((a, d) => a + d.m2_growth, 0) / dados.length;
  document.getElementById("serie-nota").textContent =
    `${mesLongo(s[0].data)} a ${mesLongo(s[s.length - 1].data)} · ${s.length} observações · média no período exibido: ${fmt(media)}% ao mês`;
  desenha("chart-serie", {
    type: "line",
    data: {
      labels: dados.map((d) => mesLabel(d.data)),
      datasets: [
        { label: "Crescimento mensal do M2", data: dados.map((d) => d.m2_growth), borderColor: css("--real"), borderWidth: 1.3, pointRadius: 0 },
        { label: "Média do período", data: dados.map(() => media), borderColor: css("--c2"), borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0 },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { y: { title: { display: true, text: "% m/m" } }, x: { ticks: { maxTicksLimit: 16 } } },
      plugins: { tooltip: { callbacks: {
        label: (c) => `${c.dataset.label}: ${fmt(c.raw)}%`,
        afterLabel: (c) => (c.datasetIndex === 0 ? `M2: ${reais(dados[c.dataIndex].valor)}` : ""),
      } } },
    },
  });
}

// ================================================================= backtest
function renderHist() {
  tabs("tabs-h", H.map((h) => [h, `h=${h}`]), estado.hHist, (v) => { estado.hHist = +v; renderHist(); });
  const d = estado.prev.historico.filter((r) => r.horizonte === estado.hHist && r.real !== null);
  desenha("chart-hist", {
    type: "line",
    data: {
      labels: d.map((r) => mesLabel(r.data_alvo)),
      datasets: [
        { label: "Realizado", data: d.map((r) => r.real), borderColor: css("--real"), borderWidth: 2, pointRadius: 2 },
        { label: "Campeão", data: d.map((r) => r.champion), borderColor: CORES_H[estado.hHist], borderWidth: 1.8, pointRadius: 2 },
        { label: "Média sazonal", data: d.map((r) => r.benchmark), borderColor: css("--c2"), borderDash: [5, 4], borderWidth: 1.4, pointRadius: 0 },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { y: { title: { display: true, text: "% m/m" } }, x: { ticks: { maxTicksLimit: 14 } } },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw)}%` } } },
    },
  });
}

// ============================================================ monitoramento
// pinta o fundo: verde abaixo do limiar (modelo melhor), vermelho acima (pior)
const zonas = {
  id: "zonas",
  beforeDraw(chart, _a, opts) {
    const { ctx, chartArea: a, scales: { y } } = chart;
    if (!a) return;
    const yl = Math.max(a.top, Math.min(a.bottom, y.getPixelForValue(opts.limiar)));
    ctx.save();
    ctx.fillStyle = css("--zona-ruim"); ctx.fillRect(a.left, a.top, a.width, yl - a.top);
    ctx.fillStyle = css("--zona-boa"); ctx.fillRect(a.left, yl, a.width, a.bottom - yl);
    ctx.font = "12px system-ui"; ctx.fillStyle = css("--muted");
    ctx.fillText("modelo pior que a média sazonal", a.left + 8, a.top + 16);
    ctx.fillText("modelo melhor que a média sazonal", a.left + 8, a.bottom - 8);
    ctx.restore();
  },
};

function renderMon() {
  const { mon, mods } = estado;
  const r = mon.regra;
  document.getElementById("mon-janela").textContent = r.janela_meses;
  document.getElementById("mon-regra").innerHTML =
    `quando <b>(A)</b> R fica acima de ${fmt(r.gatilho_a.rmse_relativo_max, 1)} por ${r.gatilho_a.meses_consecutivos} meses seguidos ` +
    `(o modelo perde para o benchmark de forma persistente) <b>ou (B)</b> o erro (RMSE) passa de ${fmt(r.gatilho_b.fator_rmse_validacao)}× ` +
    `o erro que o modelo teve na validação do artigo (piora absoluta, mesmo que o benchmark também piore). ` +
    `O retreino re-otimiza os hiperparâmetros e pode trocar o modelo campeão.`;

  document.getElementById("mon-cards").innerHTML = H.map((h) => {
    const a = mon.atual[String(h)];
    if (!a) return `<div class="card"><div class="h">h = ${h}</div><div class="m">sem dados suficientes</div></div>`;
    const dif = (1 - a.rmse_rel) * 100;
    const frase = Math.abs(dif) < 0.5 ? "empata com a média sazonal"
      : dif > 0 ? `erra <b>${fmt(dif, 0)}% menos</b> que a média sazonal` : `erra <b>${fmt(-dif, 0)}% mais</b> que a média sazonal`;
    const okA = a.meses_acima < r.gatilho_a.meses_consecutivos;
    const okB = a.rmse <= a.limite_b;
    return `<div class="card ${estado.hMon === h ? "sel" : ""}" data-h="${h}">
      <div class="h">h = ${h} ${h === 1 ? "mês" : "meses"} · até ${mesLabel(a.data_ref)}</div>
      <div class="v">R = ${fmt(a.rmse_rel)}</div>
      <div class="m">${frase} (RMSE ${fmt(a.rmse, 3)} vs ${fmt(a.rmse_bench, 3)}, ${a.n} meses)</div>
      <ul class="gatilhos">
        <li class="${okA ? "ok" : "bad"}">A: ${a.meses_acima} de ${r.gatilho_a.meses_consecutivos} meses seguidos com R &gt; ${fmt(r.gatilho_a.rmse_relativo_max, 1)}</li>
        <li class="${okB ? "ok" : "bad"}">B: RMSE ${fmt(a.rmse, 3)} ${okB ? "≤" : ">"} limite ${fmt(a.limite_b, 3)}</li>
      </ul>
      <span class="pill ${a.status}">${a.status}</span>
    </div>`;
  }).join("");
  document.querySelectorAll("#mon-cards .card[data-h]").forEach((c) => (c.onclick = () => { estado.hMon = +c.dataset.h; renderMon(); }));
  tabs("tabs-mon", H.map((h) => [h, `h=${h}`]), estado.hMon, (v) => { estado.hMon = +v; renderMon(); });

  const h = estado.hMon;
  const s = mon.serie.filter((x) => x.horizonte === h);
  const corStatus = { OK: css("--ok"), ALERTA: css("--warn"), RETREINO: css("--bad") };
  desenha("chart-mon", {
    type: "line",
    data: {
      labels: s.map((x) => mesLabel(x.data_ref)),
      datasets: [
        { label: `R (h=${h}) = RMSE campeão / RMSE média sazonal, últimos 12 meses`, data: s.map((x) => x.rmse_rel),
          borderColor: CORES_H[h], borderWidth: 2, pointRadius: 3, pointBackgroundColor: s.map((x) => corStatus[x.status]),
          pointBorderColor: s.map((x) => corStatus[x.status]) },
        { label: "Limiar (R = 1)", data: s.map(() => r.gatilho_a.rmse_relativo_max), borderColor: css("--bad"),
          borderDash: [6, 4], borderWidth: 1.2, pointRadius: 0 },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { y: { title: { display: true, text: "R (RMSE relativo)" } }, x: { ticks: { maxTicksLimit: 16 } } },
      plugins: {
        zonas: { limiar: r.gatilho_a.rmse_relativo_max },
        tooltip: { callbacks: {
          label: (c) => (c.datasetIndex === 0 ? `R = ${fmt(c.raw)}` : null),
          afterBody: (it) => {
            const x = s[it[0].dataIndex];
            return [`RMSE campeão: ${fmt(x.rmse, 3)}`, `RMSE média sazonal: ${fmt(x.rmse_bench, 3)}`, `Status: ${x.status}`, x.motivo || ""].filter(Boolean);
          },
        } },
      },
    },
    plugins: [zonas],
  });
  const nRet = s.filter((x) => x.status === "RETREINO").length;
  document.getElementById("mon-legenda").innerHTML =
    `Cada ponto avalia as previsões de h=${h} cujos meses-alvo caíram nos 12 meses anteriores. Cor do ponto: ` +
    `<span class="OK">● OK</span> · <span class="ALERTA">● alerta</span> (R &gt; 1 por pouco tempo ou viés alto) · ` +
    `<span class="RETREINO">● retreino</span> (gatilho A ou B ativo). Nesta série: ${nRet} de ${s.length} meses em retreino. ` +
    `O histórico até a implantação é um backtest com os mesmos modelos. ` +
    `O RMSE da média sazonal é igual nos quatro horizontes porque, até 12 meses, ela usa os mesmos meses do ` +
    `calendário em qualquer origem (o mesmo ocorre na Tabela 11 do artigo).`;
}

// =================================================================== tabelas
function renderModelos() {
  const tb = document.querySelector("#tab-modelos tbody");
  const reg = estado.mods.registro;
  if (!reg.length) { tb.innerHTML = "<tr><td colspan=5>MLflow sem modelos registrados.</td></tr>"; return; }
  tb.innerHTML = reg.map((r) => `<tr>
    <td>${r.horizonte}</td><td><code>${r.modelo}@champion</code></td><td>${r.modelo_nome}</td>
    <td class="num">v${r.versao}${r.challenger ? ` <span class="nota">(challenger v${r.challenger[0]})</span>` : ""}</td>
    <td>${r.origem ? mesLabel(r.origem) : "—"}</td></tr>`).join("");
}

function renderEventos() {
  const ev = estado.mon.eventos;
  const tb = document.querySelector("#tab-eventos tbody");
  if (!ev.length) { tb.innerHTML = "<tr><td colspan=5>Nenhum retreino disparado.</td></tr>"; return; }
  tb.innerHTML = ev.map((e) => `<tr title="${e.motivo ?? ""}">
    <td>${e.criado_em.slice(0, 10)}</td><td>${e.horizonte}</td>
    <td>${NOMES[e.campeao_antigo] ?? e.campeao_antigo ?? "—"} → ${NOMES[e.campeao_novo] ?? e.campeao_novo ?? "—"}</td>
    <td class="num">${fmt(e.rmse_antigo, 3)} → ${fmt(e.rmse_novo, 3)}</td><td>${e.status}</td></tr>`).join("");
}

main().catch((e) => { document.getElementById("status-geral").textContent = "Erro: " + e.message; console.error(e); });
