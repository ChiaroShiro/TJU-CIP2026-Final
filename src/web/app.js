/* =========================================================================
   Deep Research Agent · GUI 前端逻辑（原生 JS，无框架）
   - 左侧 rail 路由：选模式 → 只渲染该模式工作区
   - SSE 流式：streamPost 解析 data: 帧，按模式渲染不同形态的实时反馈
   ========================================================================= */
"use strict";

const $ = (id) => document.getElementById(id);
const EDGE_COLORS = { builds_on: "#38bdf8", compares_with: "#fb7185", similar_to: "#94a3b8" };
const NODE_COLOR = "#2dd4bf";
const NODE_PLACEHOLDER = "#f59e0b";

/* ------------------------------ 基础工具 ------------------------------ */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}
function safeUrl(u) {
  const s = String(u || "").trim();
  const scheme = s.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):/);
  if (!scheme) return s;
  return /^(https?|mailto)$/i.test(scheme[1]) ? s : "#";
}
function sanitizeSvg(svg) {
  if (!svg) return "";
  const tpl = document.createElement("template");
  tpl.innerHTML = String(svg);
  tpl.content.querySelectorAll("script, foreignObject").forEach((n) => n.remove());
  tpl.content.querySelectorAll("*").forEach((el) => {
    [...el.attributes].forEach((a) => {
      const n = a.name.toLowerCase();
      if (n.startsWith("on")) el.removeAttribute(a.name);
      if ((n === "href" || n === "xlink:href") && /^\s*javascript:/i.test(a.value)) el.removeAttribute(a.name);
    });
  });
  return tpl.innerHTML;
}
function setStatus(id, text, kind = "") {
  const el = $(id);
  if (!el) return;
  el.textContent = text || "";
  el.className = "status" + (kind ? " " + kind : "");
}
function toast(message, kind = "info") {
  const host = $("toast-host");
  if (!host) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<div class="bar"></div><div>${escapeHtml(message)}</div>`;
  host.appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 340); }, 3600);
}
function busy(btn, on, label) {
  if (!btn) return;
  if (on) {
    btn.dataset.label = btn.dataset.label || btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>${label || "处理中…"}`;
  } else {
    btn.disabled = false;
    if (btn.dataset.label) { btn.innerHTML = btn.dataset.label; delete btn.dataset.label; }
  }
}
function animateNumber(el, target) {
  if (el == null) return;
  const value = Number(target);
  if (!isFinite(value)) { el.textContent = target == null ? "–" : String(target); return; }
  const start = Number(el.dataset.val || 0);
  el.dataset.val = String(value);
  const t0 = performance.now();
  function frame(now) {
    const p = Math.min(1, (now - t0) / 650);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(start + (value - start) * eased).toLocaleString();
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
function skeleton(container, lines = 3, block = false) {
  if (!container) return;
  let html = "";
  if (block) html += `<div class="skeleton sk-block"></div>`;
  for (let i = 0; i < lines; i++) html += `<div class="skeleton sk-line" style="width:${(70 + Math.random() * 30) | 0}%"></div>`;
  container.innerHTML = html;
}
function emptyState(icon, text, hint) {
  return `<div class="empty"><div class="ic">${icon}</div><div>${escapeHtml(text)}</div>` +
    (hint ? `<div class="hint">${escapeHtml(hint)}</div>` : "") + `</div>`;
}

/* ------------------------------ 主题 ------------------------------ */
const THEME_KEY = "dr-theme";
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  const btn = $("theme-toggle");
  if (btn) btn.textContent = theme === "dark" ? "☀" : "☾";
}
function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  applyTheme(saved);
}

/* ------------------------------ Markdown ------------------------------ */
function renderInline(text) {
  let html = escapeHtml(text || "");
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (m, a, u) => `<img alt="${a}" src="${safeUrl(u)}" />`);
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, t, u) => `<a href="${safeUrl(u)}" target="_blank" rel="noopener noreferrer">${t}</a>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  html = html.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
  return html;
}
function isTableSep(line) { return /^\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$/.test((line || "").trim()); }
function splitRow(line) { return (line || "").trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => renderInline(c.trim())); }
function renderMarkdown(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) { i++; continue; }
    if (raw.startsWith("```")) {
      const fence = raw.slice(3).trim();
      const code = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) { code.push(lines[i]); i++; }
      if (i < lines.length) i++;
      blocks.push(`<pre><code class="${escapeHtml(fence)}">${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    if (/^#{1,6}\s+/.test(line)) {
      const level = line.match(/^#+/)[0].length;
      blocks.push(`<h${level}>${renderInline(line.slice(level).trim())}</h${level}>`); i++; continue;
    }
    if (/^---+$/.test(line) || /^\*\*\*+$/.test(line)) { blocks.push("<hr />"); i++; continue; }
    if (lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const headers = splitRow(lines[i]); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().includes("|")) { rows.push(splitRow(lines[i])); i++; }
      blocks.push("<table><thead><tr>" + headers.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    if (/^>\s?/.test(line)) {
      const q = [];
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) { q.push(renderInline(lines[i].trim().replace(/^>\s?/, ""))); i++; }
      blocks.push(`<blockquote>${q.join("<br />")}</blockquote>`); continue;
    }
    if (/^[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) { items.push(`<li>${renderInline(lines[i].trim().replace(/^[-*+]\s+/, ""))}</li>`); i++; }
      blocks.push(`<ul>${items.join("")}</ul>`); continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) { items.push(`<li>${renderInline(lines[i].trim().replace(/^\d+\.\s+/, ""))}</li>`); i++; }
      blocks.push(`<ol>${items.join("")}</ol>`); continue;
    }
    const para = [];
    while (i < lines.length) {
      const cur = lines[i], curT = cur.trim();
      if (!curT || cur.startsWith("```") || /^#{1,6}\s+/.test(curT) || /^>\s?/.test(curT) ||
        /^[-*+]\s+/.test(curT) || /^\d+\.\s+/.test(curT) || /^---+$/.test(curT) || /^\*\*\*+$/.test(curT) ||
        (cur.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) break;
      para.push(renderInline(curT)); i++;
    }
    blocks.push(`<p>${para.join("<br />")}</p>`);
  }
  return blocks.join("\n");
}
function renderMarkdownWithAssets(markdown, baseDir) {
  const host = document.createElement("div");
  host.innerHTML = renderMarkdown(markdown);
  if (baseDir) {
    host.querySelectorAll("img").forEach((img) => {
      const src = (img.getAttribute("src") || "").trim();
      if (!src || /^(https?:|data:|\/api\/)/i.test(src)) return;
      if (/^([a-zA-Z]:[\\/]|[\\/])/.test(src)) { img.setAttribute("src", "/api/asset?path=" + encodeURIComponent(src)); return; }
      const rel = src.replace(/^\.\//, "");
      const joined = baseDir.replace(/[\\/]+$/, "") + "/" + rel;
      img.setAttribute("src", "/api/asset?path=" + encodeURIComponent(joined));
    });
  }
  return host.innerHTML;
}

/* ------------------------------ 网络 ------------------------------ */
async function apiGet(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({ ok: false, error: "invalid JSON" }));
  if (!res.ok || !data.ok) throw new Error(data.error || `请求失败 (${res.status})`);
  return data.data;
}
async function apiPost(url, payload) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}) });
  const data = await res.json().catch(() => ({ ok: false, error: "invalid JSON" }));
  if (!res.ok || !data.ok) throw new Error(data.error || `请求失败 (${res.status})`);
  return data.data;
}
/** SSE：POST 一个 job，逐帧解析 data: 事件并回调 onEvent。 */
async function streamPost(url, payload, onEvent) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}) });
  if (!res.ok || !res.body) throw new Error(`请求失败 (${res.status})`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let errorText = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const dataLines = frame.split("\n").filter((l) => l.startsWith("data:"));
      if (!dataLines.length) continue;
      const jsonStr = dataLines.map((l) => l.slice(5).replace(/^ /, "")).join("\n");
      let evt; try { evt = JSON.parse(jsonStr); } catch (e) { continue; }
      if (evt && evt.type === "error") errorText = evt.text || "处理失败";
      try { onEvent(evt); } catch (e) { console.error("onEvent error", e); }
    }
  }
  // 服务端以 200 + {type:"error"} 上报业务失败：流结束后抛出，避免调用方误判为成功
  if (errorText) throw new Error(errorText);
}

/* ------------------------------ Activity 面板 ------------------------------ */
function makeActivity(host) {
  host.hidden = false;
  host.innerHTML =
    `<div class="phase-bar">
       <div class="phase-label"><span class="pl-text">准备中…</span><span class="pct">0%</span></div>
       <div class="phase-track"><div class="phase-fill"></div></div>
     </div>
     <div class="log-stream"></div>`;
  const fill = host.querySelector(".phase-fill");
  const plText = host.querySelector(".pl-text");
  const pctEl = host.querySelector(".pct");
  const logStream = host.querySelector(".log-stream");
  return {
    phase(label, p) {
      if (label != null) plText.textContent = label;
      if (typeof p === "number") {
        const v = Math.max(0, Math.min(100, p));
        fill.style.width = v + "%"; pctEl.textContent = Math.round(v) + "%";
      }
    },
    log(level, text) {
      const el = document.createElement("div");
      el.className = "log-line " + (level || "info");
      el.innerHTML = `<span class="lt">●</span><span class="lx">${escapeHtml(text)}</span>`;
      logStream.appendChild(el);
      logStream.scrollTop = logStream.scrollHeight;
    },
    done(label) { if (label) plText.textContent = label; fill.style.width = "100%"; pctEl.textContent = "100%"; },
  };
}

/* ------------------------------ 路由 ------------------------------ */
const ROUTES = ["search", "discover", "evaluate", "analyze", "survey", "research", "chat", "graph", "notes"];
const DEFAULT_ROUTE = "research";
const loadedOnce = {};

function showView(name) {
  if (!ROUTES.includes(name)) name = DEFAULT_ROUTE;
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === name));
  document.querySelectorAll(".rail-btn").forEach((b) => b.classList.toggle("active", b.dataset.route === name));
  if (location.hash !== "#/" + name) history.replaceState(null, "", "#/" + name);
  if (window.innerWidth <= 760) $("app").classList.remove("rail-open");
  onEnterView(name);
}
function onEnterView(name) {
  if (name === "graph" && !loadedOnce.graph) { loadedOnce.graph = true; loadGraph(""); }
  if (name === "notes" && !loadedOnce.notes) { loadedOnce.notes = true; loadNotes(""); }
}
function routeFromHash() { return (location.hash || "").replace(/^#\//, "") || DEFAULT_ROUTE; }

/* ------------------------------ 指标 / 能力 ------------------------------ */
async function refreshStats() {
  try {
    const s = await apiGet("/api/stats");
    animateNumber($("m-nodes"), s.paper_nodes ?? 0);
    animateNumber($("m-edges"), s.paper_edges ?? 0);
    animateNumber($("m-episodes"), s.episodes ?? 0);
    animateNumber($("m-skills"), s.skills ?? 0);
    animateNumber($("m-vectors"), s.vectors ?? 0);
  } catch (err) { /* 不阻塞 */ }
}
async function refreshCapabilities() {
  try {
    const c = await apiGet("/api/capabilities");
    const badge = $("llm-badge");
    if (badge) badge.textContent = c.llm ? "LLM ✓" : "无 Key";
    if (badge) badge.style.color = c.llm ? "var(--good)" : "var(--warn)";
  } catch (err) { /* ignore */ }
}

/* ------------------------------ 搜索 ------------------------------ */
function renderSearchResults(host, items, onFill) {
  if (!items || !items.length) { host.innerHTML = emptyState("🔍", "没有搜到结果", "换个英文关键词试试"); return; }
  host.innerHTML = items.map((it, i) => `
    <div class="list-item">
      <div class="li-title">${escapeHtml(it.title || "")}</div>
      <div class="li-meta">${escapeHtml((it.authors || []).slice(0, 4).join(", "))}${it.published ? " · " + escapeHtml(String(it.published).slice(0, 4)) : ""}</div>
      <div class="li-meta" style="margin-top:6px;">${escapeHtml((it.abstract || "").slice(0, 220))}…</div>
      <div class="li-actions">
        <button class="btn btn-ghost btn-sm js-fill" data-i="${i}">填入精读</button>
        <button class="btn btn-ghost btn-sm js-open" data-url="${escapeHtml(safeUrl(it.url || ""))}">打开链接</button>
      </div>
    </div>`).join("");
  host.querySelectorAll(".js-fill").forEach((b) => b.addEventListener("click", () => onFill(items[+b.dataset.i])));
  host.querySelectorAll(".js-open").forEach((b) => b.addEventListener("click", () => { if (b.dataset.url && b.dataset.url !== "#") window.open(b.dataset.url, "_blank"); }));
}
async function runSearch(query) {
  const q = (query != null ? query : $("s-q").value).trim();
  if (!q) { setStatus("s-status", "请先输入关键词", "bad"); return; }
  $("s-q").value = q;
  setStatus("s-status", "搜索中…");
  skeleton($("s-results"), 3);
  busy($("s-run"), true, "搜索中…");
  try {
    const data = await apiPost("/api/search", { query: q });
    renderSearchResults($("s-results"), data, (it) => {
      $("a-source").value = it.url || "";
      if (!$("a-title").value) $("a-title").value = it.title || "";
      showView("analyze"); toast("已填入精读", "good");
    });
    setStatus("s-status", `共 ${data.length} 条结果`, "good");
  } catch (err) {
    setStatus("s-status", String(err.message || err), "bad");
    $("s-results").innerHTML = emptyState("⚠️", "搜索失败", String(err.message || err));
  } finally { busy($("s-run"), false); }
}

/* ------------------------------ 发现 · 代码 ------------------------------ */
async function runDiscover() {
  const topic = $("d-q").value.trim();
  if (!topic) { toast("请先输入研究主题", "bad"); return; }
  const n = parseInt($("d-n").value, 10) || 10;
  const act = makeActivity($("d-activity"));
  $("d-result").hidden = true;
  busy($("d-run"), true, "发现中…");
  try {
    await streamPost("/api/discover", { topic, max_papers: n }, (ev) => {
      if (ev.type === "phase") act.phase(ev.label, ev.pct);
      else if (ev.type === "log") act.log(ev.level, ev.text);
      else if (ev.type === "result") renderDiscover(ev.payload);
      else if (ev.type === "error") act.log("error", ev.text);
    });
    act.done("完成");
  } catch (err) { toast(String(err.message || err), "bad"); }
  finally { busy($("d-run"), false); }
}
function renderDiscover(p) {
  $("d-summary").textContent = `${p.count} 篇 · ${p.with_code} 篇有公开代码`;
  const rows = (p.papers || []).map((it, i) => `
    <tr class="${it.has_code ? "has-code" : ""}">
      <td>${i + 1}</td>
      <td>${escapeHtml(it.year || "?")}</td>
      <td><a href="${escapeHtml(safeUrl(it.url || ""))}" target="_blank" rel="noopener">${escapeHtml(it.title || "")}</a></td>
      <td>${it.has_code
          ? `<span class="code-badge yes">${(it.code_confidence || 0).toFixed(2)}</span>`
          : `<span class="code-badge no">无</span>`}</td>
      <td>${it.code_url ? `<a href="${escapeHtml(safeUrl(it.code_url))}" target="_blank" rel="noopener">GitHub ↗</a>` : "-"}</td>
    </tr>`).join("");
  $("d-table").innerHTML =
    `<thead><tr><th>#</th><th>年份</th><th>论文</th><th>代码置信度</th><th>仓库</th></tr></thead><tbody>${rows}</tbody>`;
  $("d-result").hidden = false;
}

/* ------------------------------ 方向评估 ------------------------------ */
async function runEvaluate() {
  const direction = $("e-q").value.trim();
  if (!direction) { toast("请先输入研究方向", "bad"); return; }
  const act = makeActivity($("e-activity"));
  $("e-result").hidden = true;
  busy($("e-run"), true, "评估中…");
  try {
    await streamPost("/api/evaluate", { direction }, (ev) => {
      if (ev.type === "phase") act.phase(ev.label, ev.pct);
      else if (ev.type === "log") act.log(ev.level, ev.text);
      else if (ev.type === "result") renderEvaluate(ev.payload);
      else if (ev.type === "error") act.log("error", ev.text);
    });
    act.done("完成");
  } catch (err) { toast(String(err.message || err), "bad"); }
  finally { busy($("e-run"), false); }
}
function ringColor(v) { return v >= 0.66 ? "var(--good)" : v >= 0.4 ? "var(--warn)" : "var(--bad)"; }
function ring(name, v) {
  const val = Math.max(0, Math.min(1, Number(v) || 0));
  const deg = Math.round(val * 360);
  const color = ringColor(val);
  return `<div class="ring">
    <div class="ring-circle" style="background:conic-gradient(${color} ${deg}deg, var(--surface-2) ${deg}deg);">
      <span class="ring-val">${val.toFixed(2)}</span>
    </div>
    <span class="ring-name">${name}</span>
  </div>`;
}
function chips(label, arr) {
  if (!arr || !arr.length) return "";
  return `<div><div class="extra-label">${label}</div><div class="chip-row">${
    arr.map((x) => `<span class="soft-chip">${escapeHtml(typeof x === "string" ? x : JSON.stringify(x))}</span>`).join("")
  }</div></div>`;
}
function renderEvaluate(p) {
  $("e-rings").innerHTML = ring("可行性", p.feasibility) + ring("新颖性", p.novelty) + ring("影响力", p.impact);
  $("e-analysis").innerHTML = renderMarkdown(p.analysis || "（无详细分析）");
  let extra = "";
  if (p.recommendations && p.recommendations.length) {
    extra += `<div><div class="extra-label">研究建议</div><ul class="panel-box" style="list-style:none;padding-left:16px;">${
      p.recommendations.map((r) => `<li style="margin:4px 0;">• ${escapeHtml(r)}</li>`).join("")}</ul></div>`;
  }
  extra += chips("相关 Benchmark", p.benchmarks);
  extra += chips("相关主题", p.related_topics);
  extra += chips("检索词", p.search_queries);
  if (p.papers && p.papers.length) {
    extra += `<div><div class="extra-label">参考论文</div><ul class="panel-box" style="padding-left:18px;">${
      p.papers.map((x) => `<li><a href="${escapeHtml(safeUrl(x.url))}" target="_blank" rel="noopener">${escapeHtml(x.title)}</a></li>`).join("")}</ul></div>`;
  }
  $("e-extra").innerHTML = extra;
  $("e-result").hidden = false;
}

/* ------------------------------ 论文精读 ------------------------------ */
function fmtList(items, emptyText) {
  if (!items || !items.length) return `<li class="li-empty">${escapeHtml(emptyText)}</li>`;
  return items.map((it) => `<li>${escapeHtml(typeof it === "string" ? it : JSON.stringify(it))}</li>`).join("");
}
function renderAnalysis(data) {
  const r = data.result || {};
  const meta = `<div class="meta-row">
      <span class="tag">${escapeHtml(r._analysis_mode || "—")}</span>
      <span class="tag">来源: ${escapeHtml(r._source || "—")}</span>
      ${data.local ? '<span class="tag">本地 PDF</span>' : ""}
      ${r._num_chunks ? `<span class="tag">${r._num_chunks} 章节块</span>` : ""}
      ${(r.tags || []).slice(0, 6).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}
    </div>`;
  const sections = [
    ["一句话总结", r.tldr], ["核心问题", r.problem], ["研究动机", r.motivation],
    ["方法概述", r.method_summary], ["实验结果", r.results], ["消融分析", r.ablations], ["未来方向", r.future_work],
  ].filter(([, v]) => v).map(([h, v]) => `### ${h}\n\n${v}`).join("\n\n");
  const noteLine = r._note_path ? `\n\n> 笔记已保存：\`${r._note_path}\`` : "";
  $("a-main").innerHTML = `${meta}<h2 style="margin-top:2px;">${escapeHtml(data.title || "")}</h2>` +
    `<div class="markdown-body">${renderMarkdown(sections + noteLine)}</div>`;

  $("a-contribs").innerHTML = fmtList(r.contributions || [], "暂无");
  const formulas = r.formulas || [];
  $("a-formulas").innerHTML = formulas.length
    ? formulas.map((f) => `<div class="formula"><div class="fname">${escapeHtml(f.name || "公式")}</div>` +
        `<pre>${escapeHtml(f.latex || "")}</pre>` +
        (f.meaning ? `<div class="fmean">${escapeHtml(f.meaning)}</div>` : "") + `</div>`).join("")
    : `<div class="li-empty" style="font-style:italic;color:var(--faint);">暂无公式</div>`;
  $("a-datasets").innerHTML = fmtList((r.datasets || []).map((it) =>
    typeof it === "string" ? it : (it.name || "Unknown") + (it.used_for ? " / " + it.used_for : "") + (it.size ? " / " + it.size : "")), "暂无");
  $("a-weak").innerHTML = fmtList(r.weaknesses || r.limitations || [], "暂无");
  const repro = r.reproducibility || {};
  const reproRows = [["提供代码", repro.code], ["提供权重", repro.weights], ["训练细节充分", repro.details_sufficient], ["数据可公开", repro.data_public]];
  $("a-repro").innerHTML = Object.keys(repro).length
    ? reproRows.map(([k, v]) => `<div class="repro-item"><span class="box ${v ? "on" : ""}">${v ? "✓" : ""}</span>${escapeHtml(k)}</div>`).join("")
    : `<div class="li-empty" style="font-style:italic;color:var(--faint);">暂无</div>`;
  $("a-memory").innerHTML = fmtList((r.memory_connections || []).map((it) =>
    (it.title || "Unknown") + " · " + (it.relation || "") + " · " + Number(it.confidence || 0).toFixed(2)), "暂无高置信记忆连接");
  $("a-related").innerHTML = fmtList((r.cited_similar_work || []).map((it) =>
    typeof it === "string" ? it : (it.title || "Unknown") + (it.category ? " [" + it.category + "]" : "")), "暂无");
}
async function runAnalyze() {
  const source = $("a-source").value.trim();
  if (!source) { toast("请先输入论文来源", "bad"); return; }
  const act = makeActivity($("a-activity"));
  $("a-result").hidden = true;
  busy($("a-run"), true, "精读中…");
  try {
    await streamPost("/api/analyze", {
      source, title: $("a-title").value.trim(), focus: $("a-focus").value.trim(), mode: $("a-mode").value,
    }, (ev) => {
      if (ev.type === "phase") act.phase(ev.label, ev.pct);
      else if (ev.type === "log") act.log(ev.level, ev.text);
      else if (ev.type === "result") { renderAnalysis(ev.payload); $("a-result").hidden = false; }
      else if (ev.type === "error") act.log("error", ev.text);
    });
    act.done("完成");
    toast("精读完成 ✓", "good");
    refreshStats();
    if (loadedOnce.notes) loadNotes($("n-q").value.trim());
    loadedOnce.graph = false;
  } catch (err) { toast(String(err.message || err), "bad"); }
  finally { busy($("a-run"), false); }
}

/* ------------------------------ 综述 ------------------------------ */
function buildFigureCard(fig) {
  const card = document.createElement("div");
  card.className = "figure-card";
  const kind = fig.kind || "svg";
  const body = document.createElement("div");
  body.className = "fc-body";
  let zoomable = true;
  if (kind === "image") {
    const src = (fig.image && (fig.image.data_uri || fig.image.url)) || "";
    body.innerHTML = `<img alt="${escapeHtml(fig.title || "")}" src="${escapeHtml(src)}" />`;
  } else if (kind === "latex") {
    const rendered = fig.latex && fig.latex.rendered_svg;
    if (rendered) body.innerHTML = sanitizeSvg(rendered);
    else { body.style.cursor = "default"; zoomable = false; body.innerHTML = `<pre class="latex-source">${escapeHtml((fig.latex && fig.latex.source) || "")}</pre>`; }
  } else body.innerHTML = sanitizeSvg(fig.svg);

  const foot = document.createElement("div");
  foot.className = "fc-foot";
  if (fig.download_path) foot.innerHTML = `<a class="btn btn-ghost btn-sm" href="/api/asset?path=${encodeURIComponent(fig.download_path)}" download>下载文件</a>`;
  if (zoomable) {
    const zb = document.createElement("button"); zb.className = "btn btn-ghost btn-sm"; zb.textContent = "放大查看";
    zb.addEventListener("click", () => openLightbox(body.innerHTML)); foot.appendChild(zb);
    body.addEventListener("click", () => openLightbox(body.innerHTML));
  }
  card.innerHTML = `<div class="fc-head"><span class="fc-title">${escapeHtml(fig.title || "图")}</span><span class="kind-badge kind-${kind}">${kind}</span></div>`;
  card.appendChild(body); card.appendChild(foot);
  if (fig.meta && fig.meta.note) { const note = document.createElement("div"); note.className = "fc-note"; note.textContent = fig.meta.note; card.appendChild(note); }
  return card;
}
function renderFigures(container, figures) { container.innerHTML = ""; (figures || []).forEach((f) => container.appendChild(buildFigureCard(f))); }
async function runSurvey() {
  const topic = $("sv-topic").value.trim();
  if (!topic) { setStatus("sv-status", "请先输入综述主题", "bad"); return; }
  const n = parseInt($("sv-n").value, 10) || 10;
  setStatus("sv-status", "正在检索论文并生成综述与图，可能需要 30s–2min…");
  $("sv-figures").innerHTML = `<div class="skeleton sk-block"></div><div class="skeleton sk-block"></div>`;
  $("sv-report").hidden = true;
  busy($("sv-run"), true, "生成中…");
  try {
    const data = await apiPost("/api/survey", { topic, max_papers: n });
    renderFigures($("sv-figures"), data.figures);
    $("sv-report").hidden = false;
    $("sv-report").innerHTML = `<div class="markdown-body">${renderMarkdownWithAssets(data.report_md || "", data.output_dir)}</div>`;
    setStatus("sv-status", `已生成：${(data.papers || []).length} 篇论文 · 2 张 SVG 图`, "good");
    toast("综述与图已生成 ✓", "good");
  } catch (err) {
    setStatus("sv-status", String(err.message || err), "bad");
    $("sv-figures").innerHTML = emptyState("⚠️", "生成失败", String(err.message || err));
  } finally { busy($("sv-run"), false); }
}

/* ------------------------------ 深度研究 ------------------------------ */
const research = { raw: "", timer: null, steps: 0, tokens: 0, maxSteps: 30, maxTokens: 200000, mode: "auto" };
function stepCard(ev) {
  const cls = ev.status === "ok" ? "ok" : ev.status === "err" ? "err" : "think";
  const statusTxt = ev.status === "ok" ? "OK" : ev.status === "err" ? "ERR" : "···";
  const tool = ev.tool || "(thinking)";
  const el = document.createElement("div");
  el.className = "step-card " + cls;
  el.innerHTML =
    `<div class="step-top"><span class="step-idx">#${(ev.idx ?? 0) + 1}</span>` +
    `<span class="step-tool">${escapeHtml(tool)}</span>` +
    `<span class="step-status ${cls}">${statusTxt}</span></div>` +
    (ev.args_brief ? `<div class="step-args">${escapeHtml(ev.args_brief)}</div>` : "") +
    (ev.thought ? `<div class="step-thought">${escapeHtml(ev.thought)}</div>` : "") +
    `<div class="step-meta">${ev.tokens || 0} tok · ${ev.ms || 0} ms${ev.error ? " · " + escapeHtml(ev.error) : ""}</div>`;
  return el;
}
function traceLogLine(level, text) {
  const el = document.createElement("div");
  el.className = "log-line " + (level || "info");
  el.innerHTML = `<span class="lt">●</span><span class="lx">${escapeHtml(text)}</span>`;
  return el;
}
function updateBudget() {
  if (research.mode !== "auto") return;
  const bar = $("r-budget"); bar.hidden = false;
  const stepPct = research.maxSteps ? research.steps / research.maxSteps : 0;
  const tokPct = research.maxTokens ? research.tokens / research.maxTokens : 0;
  const pct = Math.min(1, Math.max(stepPct, tokPct)) * 100;
  const fill = $("r-budget-fill");
  fill.style.width = pct + "%";
  fill.classList.toggle("warn", pct > 70);
  $("r-budget-meta").textContent = `步 ${research.steps}/${research.maxSteps} · token ${research.tokens.toLocaleString()}/${research.maxTokens.toLocaleString()}`;
}
function renderReportThrottled() {
  if (research.timer) return;
  research.timer = setTimeout(() => {
    research.timer = null;
    const el = $("r-report");
    el.classList.add("typing");
    el.innerHTML = `<div class="markdown-body">${renderMarkdown(research.raw)}</div>`;
    el.scrollTop = el.scrollHeight;
  }, 90);
}
async function runResearch() {
  const topic = $("r-topic").value.trim();
  if (!topic) { setStatus("r-status", "请先输入研究主题", "bad"); return; }
  research.mode = $("r-mode").value;
  research.maxSteps = parseInt($("r-steps").value, 10) || 30;
  research.maxTokens = parseInt($("r-tokens").value, 10) || 200000;
  research.raw = ""; research.steps = 0; research.tokens = 0;
  $("r-trace").innerHTML = "";
  $("r-trace-count").textContent = "进行中…";
  $("r-report").innerHTML = `<div class="empty"><div class="ic">⏳</div><div>研究启动中…</div></div>`;
  $("r-stats").hidden = true;
  $("r-budget").hidden = research.mode !== "auto";
  if (research.mode === "auto") { $("r-budget-fill").style.width = "0%"; $("r-budget-meta").textContent = ""; }
  setStatus("r-status", "研究进行中，请保持页面打开…");
  busy($("r-run"), true, "研究中…");
  const trace = $("r-trace");
  let firstToken = true;
  try {
    await streamPost("/api/research", {
      topic, mode: research.mode, max_steps: research.maxSteps, max_tokens: research.maxTokens,
    }, (ev) => {
      if (ev.type === "step") {
        research.steps += 1; research.tokens += ev.tokens || 0;
        trace.appendChild(stepCard(ev)); trace.scrollTop = trace.scrollHeight;
        $("r-trace-count").textContent = `${research.steps} 步`;
        updateBudget();
      } else if (ev.type === "phase") {
        setStatus("r-status", ev.label + (typeof ev.pct === "number" ? ` · ${Math.round(ev.pct)}%` : ""));
      } else if (ev.type === "log") {
        trace.appendChild(traceLogLine(ev.level, ev.text)); trace.scrollTop = trace.scrollHeight;
      } else if (ev.type === "token" && ev.pane === "report") {
        if (firstToken) { research.raw = ""; firstToken = false; }
        research.raw += ev.text || ""; renderReportThrottled();
      } else if (ev.type === "result") {
        finalizeResearch(ev.payload);
      } else if (ev.type === "error") {
        trace.appendChild(traceLogLine("error", ev.text));
      }
    });
    setStatus("r-status", "研究完成", "good");
  } catch (err) { setStatus("r-status", String(err.message || err), "bad"); toast(String(err.message || err), "bad"); }
  finally {
    busy($("r-run"), false);
    if (research.timer) { clearTimeout(research.timer); research.timer = null; }
    $("r-report").classList.remove("typing");
    refreshStats(); loadedOnce.graph = false;
  }
}
function finalizeResearch(p) {
  if (research.timer) { clearTimeout(research.timer); research.timer = null; }
  const md = p.report_md || research.raw || "（无报告内容）";
  $("r-report").classList.remove("typing");
  $("r-report").innerHTML = `<div class="markdown-body">${renderMarkdown(md)}</div>`;
  const st = p.stats || {};
  const bits = [];
  if (p.mode === "auto") {
    bits.push(`步数 ${st.total_steps ?? "-"}`, `tokens ${(st.total_tokens ?? 0).toLocaleString()}`,
      `耗时 ${st.elapsed_ms ? (st.elapsed_ms / 1000).toFixed(1) + "s" : "-"}`,
      `状态 ${p.finished ? "完成" : "未完成(" + (p.finish_reason || "") + ")"}`);
  } else {
    bits.push(`论文 ${st.paper_count ?? "-"}`, `子任务 ${st.task_count ?? "-"}`,
      `Critic ${st.critic_score != null ? Number(st.critic_score).toFixed(2) : "-"}`, `修改 ${st.revision_count ?? 0} 次`);
  }
  if (p.report_path) bits.push(`已保存`);
  const stats = $("r-stats"); stats.hidden = false;
  stats.innerHTML = bits.map((b) => `<span>${escapeHtml(b)}</span>`).join("");
}

/* ------------------------------ 对话 ------------------------------ */
function addChatMessage(role) {
  const host = $("c-messages");
  const first = host.querySelector(".empty");
  if (first) first.remove();
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  wrap.innerHTML = role === "assistant"
    ? `<div class="msg-intent" hidden></div><div class="msg-body"></div><div class="msg-progress" hidden></div>`
    : `<div class="msg-body"></div>`;
  host.appendChild(wrap); host.scrollTop = host.scrollHeight;
  return wrap;
}
async function sendChat() {
  const input = $("c-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = ""; input.style.height = "auto";
  const userMsg = addChatMessage("user");
  userMsg.querySelector(".msg-body").textContent = message;
  const bot = addChatMessage("assistant");
  const intentEl = bot.querySelector(".msg-intent");
  const bodyEl = bot.querySelector(".msg-body");
  const progEl = bot.querySelector(".msg-progress");
  bodyEl.classList.add("typing");
  let replyRaw = "", firstReply = true;
  busy($("c-send"), true, "…");
  try {
    await streamPost("/api/chat", { message }, (ev) => {
      if (ev.type === "intent") {
        intentEl.hidden = false;
        const q = (ev.queries || []).length ? " · " + escapeHtml(ev.queries.join(" / ")) : "";
        intentEl.innerHTML = `意图 <span class="badge">${escapeHtml(ev.action || "")}</span>${q}`;
      } else if (ev.type === "phase" || ev.type === "log") {
        progEl.hidden = false;
        const line = document.createElement("div");
        line.className = "log-line " + (ev.level || "info");
        line.innerHTML = `<span class="lt">●</span><span class="lx">${escapeHtml(ev.label || ev.text || "")}</span>`;
        progEl.appendChild(line);
      } else if (ev.type === "token" && ev.pane === "reply") {
        if (firstReply) { replyRaw = ""; firstReply = false; }
        replyRaw += ev.text || "";
        bodyEl.innerHTML = `<div class="markdown-body">${renderMarkdown(replyRaw)}</div>`;
        $("c-messages").scrollTop = $("c-messages").scrollHeight;
      } else if (ev.type === "result") {
        const reply = (ev.payload && ev.payload.reply) || replyRaw;
        bodyEl.innerHTML = `<div class="markdown-body">${renderMarkdown(reply)}</div>`;
      } else if (ev.type === "error") {
        bodyEl.innerHTML = `<span style="color:var(--bad);">${escapeHtml(ev.text)}</span>`;
      }
    });
  } catch (err) {
    bodyEl.innerHTML = `<span style="color:var(--bad);">${escapeHtml(String(err.message || err))}</span>`;
  } finally {
    bodyEl.classList.remove("typing");
    busy($("c-send"), false);
    $("c-messages").scrollTop = $("c-messages").scrollHeight;
    refreshStats();
  }
}
async function clearChat() {
  try { await apiPost("/api/chat-clear", {}); } catch (e) { /* ignore */ }
  $("c-messages").innerHTML = emptyState("🤖", "直接描述你的研究想法或问题，我会判断该怎么帮你。");
  toast("对话已清空", "good");
}

/* ------------------------------ 图谱（d3 力导向 + 降级） ------------------------------ */
const graphState = { sim: null, zoom: null, svg: null };
function edgeColor(type) { return EDGE_COLORS[type] || EDGE_COLORS.similar_to; }
function nodeColor(n) { return n.metadata && n.metadata.placeholder ? NODE_PLACEHOLDER : NODE_COLOR; }
function nodeLabel(n) { return n.method_name || n.title || n.paper_id || ""; }
function showNodeDetail(n) {
  const el = $("g-detail"); el.className = "node-detail";
  el.innerHTML = `<span class="pill">${escapeHtml(n.method_name || "Paper")}</span>
    <div class="nd-title">${escapeHtml(n.title || n.paper_id || "")}</div>
    <div class="li-meta" style="margin-top:6px;">id: ${escapeHtml(n.paper_id || "")}</div>
    <div class="li-meta" style="margin-top:6px;">${escapeHtml(n.problem || n.tldr || "暂无摘要")}</div>
    ${n.note_path ? `<div class="li-actions"><button class="btn btn-ghost btn-sm" id="nd-open">打开笔记</button></div>` : ""}`;
  const open = $("nd-open");
  if (open) open.addEventListener("click", () => { showView("notes"); openNote(n.note_path).catch((e) => toast(String(e.message || e), "bad")); });
}
function setGraphStats(snapshot, nNodes, nEdges) {
  const s = snapshot.stats || {};
  $("g-stats").textContent = [
    `显示节点: ${nNodes}`, `显示边: ${nEdges}`,
    `已存论文节点: ${s.paper_nodes ?? "-"}`, `已存论文边: ${s.paper_edges ?? "-"}`,
    `episodes: ${s.episodes ?? "-"}`, `vectors: ${s.vectors ?? "-"}`,
  ].join("\n");
}
function renderGraphD3(snapshot) {
  const d3 = window.d3;
  const svgEl = $("graph-stage");
  const w = svgEl.clientWidth || 760, h = svgEl.clientHeight || 560;
  if (graphState.sim) { graphState.sim.stop(); graphState.sim = null; }
  const svg = d3.select(svgEl); svg.selectAll("*").remove();
  const nodes = (snapshot.nodes || []).map((n) => ({ ...n, id: n.paper_id }));
  const idset = new Set(nodes.map((n) => n.id));
  const links = (snapshot.edges || [])
    .filter((e) => idset.has(e.src_paper_id) && idset.has(e.dst_paper_id))
    .map((e) => ({ source: e.src_paper_id, target: e.dst_paper_id, type: e.relation_type, strength: e.relation_strength || 0.5 }));
  setGraphStats(snapshot, nodes.length, links.length);
  if (!nodes.length) {
    svg.append("text").attr("x", w / 2).attr("y", h / 2).attr("text-anchor", "middle")
      .attr("fill", "var(--faint)").attr("font-size", 15).text("暂无图谱数据，先分析几篇论文");
    return;
  }
  const deg = {};
  links.forEach((l) => { deg[l.source] = (deg[l.source] || 0) + 1; deg[l.target] = (deg[l.target] || 0) + 1; });
  const radius = (n) => 16 + Math.min(14, (deg[n.id] || 0) * 3);
  const root = svg.append("g");
  const zoom = d3.zoom().scaleExtent([0.3, 3]).on("zoom", (ev) => root.attr("transform", ev.transform));
  svg.call(zoom).on("dblclick.zoom", null);
  graphState.svg = svg; graphState.zoom = zoom;
  const linkSel = root.append("g").selectAll("line").data(links).join("line")
    .attr("class", "gedge").attr("stroke", (d) => edgeColor(d.type)).attr("stroke-width", (d) => 1.2 + d.strength * 2.2);
  const linkLabel = root.append("g").selectAll("text").data(links).join("text")
    .attr("class", "gedge-label").attr("text-anchor", "middle").text((d) => d.type || "");
  const nodeSel = root.append("g").selectAll("g.gnode").data(nodes).join("g").attr("class", "gnode").style("cursor", "pointer");
  nodeSel.append("circle").attr("class", "halo").attr("r", (d) => radius(d) + 7).attr("fill", "none")
    .attr("stroke", (d) => nodeColor(d)).attr("stroke-opacity", 0.18).attr("stroke-width", 8);
  nodeSel.append("circle").attr("class", "core").attr("r", radius).attr("fill", (d) => nodeColor(d)).attr("fill-opacity", 0.95);
  nodeSel.append("text").attr("text-anchor", "middle").attr("dy", (d) => radius(d) + 15)
    .text((d) => { const t = nodeLabel(d); return t.length > 16 ? t.slice(0, 15) + "…" : t; });
  const adj = {};
  links.forEach((l) => { (adj[l.source] = adj[l.source] || new Set()).add(l.target); (adj[l.target] = adj[l.target] || new Set()).add(l.source); });
  nodeSel.on("mouseenter", (ev, d) => {
    nodeSel.classed("dim", (o) => o.id !== d.id && !(adj[d.id] && adj[d.id].has(o.id)));
    linkSel.classed("dim", (o) => o.source.id !== d.id && o.target.id !== d.id);
    linkLabel.classed("dim", (o) => o.source.id !== d.id && o.target.id !== d.id);
  }).on("mouseleave", () => { nodeSel.classed("dim", false); linkSel.classed("dim", false); linkLabel.classed("dim", false); })
    .on("click", (ev, d) => { ev.stopPropagation(); showNodeDetail(d); });
  nodeSel.call(d3.drag()
    .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
    .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance((d) => 90 + (1 - d.strength) * 70))
    .force("charge", d3.forceManyBody().strength(-340))
    .force("center", d3.forceCenter(w / 2, h / 2))
    .force("collide", d3.forceCollide((d) => radius(d) + 16))
    .on("tick", () => {
      linkSel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      linkLabel.attr("x", (d) => (d.source.x + d.target.x) / 2).attr("y", (d) => (d.source.y + d.target.y) / 2 - 5);
      nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
  graphState.sim = sim;
}
function renderGraphFallback(snapshot) {
  const svgEl = $("graph-stage");
  const w = svgEl.clientWidth || 760, h = svgEl.clientHeight || 560;
  const nodes = snapshot.nodes || [];
  setGraphStats(snapshot, nodes.length, (snapshot.edges || []).length);
  if (!nodes.length) { svgEl.innerHTML = `<text x="${w / 2}" y="${h / 2}" text-anchor="middle" fill="#94a3b8">暂无图谱数据</text>`; return; }
  const pos = {}, cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.36;
  nodes.forEach((n, idx) => { const a = (idx / nodes.length) * Math.PI * 2; pos[n.paper_id] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) }; });
  const edges = (snapshot.edges || []).map((e) => {
    const a = pos[e.src_paper_id], b = pos[e.dst_paper_id]; if (!a || !b) return "";
    return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${edgeColor(e.relation_type)}" stroke-width="1.6" stroke-opacity="0.6"/>`;
  }).join("");
  const circles = nodes.map((n) => {
    const p = pos[n.paper_id], t = nodeLabel(n);
    return `<g class="gnode" data-id="${escapeHtml(n.paper_id)}" transform="translate(${p.x},${p.y})" style="cursor:pointer;">
      <circle r="20" fill="${nodeColor(n)}" fill-opacity="0.95" stroke="#fff" stroke-width="3"/>
      <text text-anchor="middle" dy="34" font-size="11" fill="#94a3b8">${escapeHtml(t.length > 16 ? t.slice(0, 15) + "…" : t)}</text></g>`;
  }).join("");
  svgEl.innerHTML = `<g>${edges}${circles}</g>`;
  svgEl.querySelectorAll(".gnode").forEach((g) => g.addEventListener("click", () => { const n = nodes.find((x) => x.paper_id === g.dataset.id); if (n) showNodeDetail(n); }));
}
async function loadGraph(query = "") {
  $("graph-engine").textContent = window.d3 ? "d3 force" : "fallback layout";
  try {
    const snap = await apiGet("/api/graph" + (query ? "?q=" + encodeURIComponent(query) : ""));
    if (window.d3) renderGraphD3(snap); else renderGraphFallback(snap);
  } catch (err) { $("g-stats").textContent = "加载失败: " + (err.message || err); toast("图谱加载失败", "bad"); }
}
function zoomBy(factor) { if (graphState.svg && graphState.zoom) graphState.svg.transition().duration(250).call(graphState.zoom.scaleBy, factor); }
function resetGraph() { if (graphState.svg && graphState.zoom && window.d3) graphState.svg.transition().duration(300).call(graphState.zoom.transform, window.d3.zoomIdentity); }

/* ------------------------------ 笔记 ------------------------------ */
function renderNotes(items) {
  const host = $("n-list");
  if (!items || !items.length) { host.innerHTML = emptyState("📭", "没有匹配到笔记", "先精读一篇论文"); return; }
  host.innerHTML = items.map((it) => `
    <div class="list-item">
      <div class="li-title">${escapeHtml(it.title || "")}</div>
      <div class="li-meta">${escapeHtml(it.path || "")}</div>
      <div class="li-meta" style="margin-top:6px;">${escapeHtml((it.preview || "").slice(0, 160))}</div>
      <div class="li-actions"><button class="btn btn-ghost btn-sm js-note" data-path="${escapeHtml(it.path || "")}">查看笔记</button></div>
    </div>`).join("");
  host.querySelectorAll(".js-note").forEach((b) => b.addEventListener("click", () => openNote(b.dataset.path || "").catch((e) => toast(String(e.message || e), "bad"))));
}
async function loadNotes(query = "") {
  try { renderNotes(await apiGet("/api/notes" + (query ? "?q=" + encodeURIComponent(query) : ""))); }
  catch (err) { $("n-list").innerHTML = emptyState("⚠️", "加载笔记失败", String(err.message || err)); }
}
async function openNote(path) {
  if (!path) throw new Error("笔记路径为空");
  const data = await apiPost("/api/open-note", { path });
  const baseDir = data.dir || (data.path || "").replace(/[\\/][^\\/]*$/, "");
  $("n-viewer").innerHTML = `<div class="markdown-body">${renderMarkdownWithAssets(data.content || "", baseDir)}</div>`;
  $("n-viewer").scrollTop = 0;
  toast("笔记已加载", "good");
}

/* ------------------------------ 浮层 ------------------------------ */
function openLightbox(innerHtml) { $("lightbox-inner").innerHTML = innerHtml; $("lightbox").classList.add("open"); }
function closeLightbox() { $("lightbox").classList.remove("open"); $("lightbox-inner").innerHTML = ""; }
function openCmdk() { $("cmdk").classList.add("open"); setTimeout(() => $("cmdk-input").focus(), 30); }
function closeCmdk() { $("cmdk").classList.remove("open"); }

/* ------------------------------ 事件绑定 ------------------------------ */
function wireEvents() {
  // rail 路由
  document.querySelectorAll(".rail-btn").forEach((b) => b.addEventListener("click", () => showView(b.dataset.route)));
  window.addEventListener("hashchange", () => showView(routeFromHash()));

  $("theme-toggle").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  $("rail-collapse").addEventListener("click", () => {
    if (window.innerWidth <= 760) $("app").classList.toggle("rail-open");
    else $("app").classList.toggle("rail-collapsed");
  });

  // 搜索
  $("s-run").addEventListener("click", () => runSearch());
  $("s-q").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  // 发现
  $("d-run").addEventListener("click", runDiscover);
  $("d-q").addEventListener("keydown", (e) => { if (e.key === "Enter") runDiscover(); });
  // 评估
  $("e-run").addEventListener("click", runEvaluate);
  // 精读
  $("a-run").addEventListener("click", runAnalyze);
  // 综述
  $("sv-run").addEventListener("click", runSurvey);
  $("sv-topic").addEventListener("keydown", (e) => { if (e.key === "Enter") runSurvey(); });
  // 研究
  $("r-run").addEventListener("click", runResearch);
  $("r-mode").addEventListener("change", () => {
    const auto = $("r-mode").value === "auto";
    document.querySelectorAll("[data-auto-only]").forEach((el) => el.style.display = auto ? "" : "none");
  });
  // 对话
  $("c-send").addEventListener("click", sendChat);
  $("c-clear").addEventListener("click", clearChat);
  $("c-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
  $("c-input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = Math.min(160, e.target.scrollHeight) + "px"; });
  // 图谱
  $("g-run").addEventListener("click", () => loadGraph($("g-q").value.trim()));
  $("g-q").addEventListener("keydown", (e) => { if (e.key === "Enter") loadGraph($("g-q").value.trim()); });
  $("g-zoom-in").addEventListener("click", () => zoomBy(1.3));
  $("g-zoom-out").addEventListener("click", () => zoomBy(1 / 1.3));
  $("g-reset").addEventListener("click", resetGraph);
  // 笔记
  $("n-run").addEventListener("click", () => loadNotes($("n-q").value.trim()));
  $("n-q").addEventListener("keydown", (e) => { if (e.key === "Enter") loadNotes($("n-q").value.trim()); });

  // 浮层
  $("open-cmdk").addEventListener("click", openCmdk);
  $("cmdk-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { const v = e.target.value.trim(); closeCmdk(); if (v) { $("s-q").value = v; showView("search"); runSearch(v); } }
    if (e.key === "Escape") closeCmdk();
  });
  $("cmdk").addEventListener("click", (e) => { if (e.target.id === "cmdk") closeCmdk(); });
  $("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") closeLightbox(); });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openCmdk(); }
    if (e.key === "Escape") { closeCmdk(); closeLightbox(); }
  });
  window.addEventListener("resize", () => {
    clearTimeout(window.__graphResize);
    window.__graphResize = setTimeout(() => { if (document.querySelector('.view[data-view="graph"]').classList.contains("active")) loadGraph($("g-q").value.trim()); }, 300);
  });
}

async function boot() {
  initTheme();
  wireEvents();
  showView(routeFromHash());
  await refreshStats();
  refreshCapabilities();
}
document.addEventListener("DOMContentLoaded", () => { boot().catch((e) => toast(String(e.message || e), "bad")); });
