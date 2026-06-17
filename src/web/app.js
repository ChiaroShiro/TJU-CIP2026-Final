/* =========================================================================
   Deep Research Agent · GUI 前端逻辑（原生 JS，无框架）
   ========================================================================= */
"use strict";

const $ = (id) => document.getElementById(id);
const EDGE_COLORS = { builds_on: "#38bdf8", compares_with: "#fb7185", similar_to: "#94a3b8" };
const NODE_COLOR = "#2dd4bf";
const NODE_PLACEHOLDER = "#f59e0b";

/* ------------------------------ 基础工具 ------------------------------ */
async function apiGet(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({ ok: false, error: "invalid JSON" }));
  if (!res.ok || !data.ok) throw new Error(data.error || `请求失败 (${res.status})`);
  return data.data;
}
async function apiPost(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await res.json().catch(() => ({ ok: false, error: "invalid JSON" }));
  if (!res.ok || !data.ok) throw new Error(data.error || `请求失败 (${res.status})`);
  return data.data;
}
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}
/** 只放行 http/https/mailto 与相对链接，拦截 javascript: / data: 等危险协议 */
function safeUrl(u) {
  const s = String(u || "").trim();
  const scheme = s.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):/);
  if (!scheme) return s; // 无协议 = 相对链接，放行
  return /^(https?|mailto)$/i.test(scheme[1]) ? s : "#";
}
/** 注入前清洗 SVG：移除 script/foreignObject、on* 事件、javascript: 链接（零依赖，面向未来外部图源） */
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
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 320);
  }, 3600);
}
function busy(btn, on, label) {
  if (!btn) return;
  if (on) {
    btn.dataset.label = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>${label || "处理中…"}`;
  } else {
    btn.disabled = false;
    if (btn.dataset.label) btn.innerHTML = btn.dataset.label;
  }
}
function animateNumber(el, target) {
  if (el == null) return;
  const value = Number(target);
  if (!isFinite(value)) { el.textContent = target == null ? "–" : String(target); return; }
  const start = Number(el.dataset.val || 0);
  el.dataset.val = String(value);
  const t0 = performance.now();
  const dur = 650;
  function frame(now) {
    const p = Math.min(1, (now - t0) / dur);
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
  for (let i = 0; i < lines; i++) {
    html += `<div class="skeleton sk-line" style="width:${70 + Math.random() * 30 | 0}%"></div>`;
  }
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
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2" />');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, t, u) => `<a href="${safeUrl(u)}" target="_blank" rel="noopener noreferrer">${t}</a>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  html = html.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
  html = html.replace(/(?<!_)_([^_\n]+)_(?!_)/g, "<em>$1</em>");
  return html;
}
function isTableSep(line) {
  return /^\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$/.test((line || "").trim());
}
function splitRow(line) {
  return (line || "").trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => renderInline(c.trim()));
}
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
      const code = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) { code.push(lines[i]); i++; }
      if (i < lines.length) i++;
      blocks.push(`<pre><code class="${escapeHtml(fence)}">${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    if (/^#{1,6}\s+/.test(line)) {
      const level = line.match(/^#+/)[0].length;
      blocks.push(`<h${level}>${renderInline(line.slice(level).trim())}</h${level}>`);
      i++; continue;
    }
    if (/^---+$/.test(line) || /^\*\*\*+$/.test(line)) { blocks.push("<hr />"); i++; continue; }
    if (lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const headers = splitRow(lines[i]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().includes("|")) { rows.push(splitRow(lines[i])); i++; }
      blocks.push(
        "<table><thead><tr>" + headers.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    if (/^>\s?/.test(line)) {
      const q = [];
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) { q.push(renderInline(lines[i].trim().replace(/^>\s?/, ""))); i++; }
      blocks.push(`<blockquote>${q.join("<br />")}</blockquote>`);
      continue;
    }
    if (/^[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) { items.push(`<li>${renderInline(lines[i].trim().replace(/^[-*+]\s+/, ""))}</li>`); i++; }
      blocks.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) { items.push(`<li>${renderInline(lines[i].trim().replace(/^\d+\.\s+/, ""))}</li>`); i++; }
      blocks.push(`<ol>${items.join("")}</ol>`);
      continue;
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
/** 把渲染后 HTML 里的相对图片路径改写为 /api/asset，使本地笔记图能显示 */
function renderMarkdownWithAssets(markdown, baseDir) {
  const host = document.createElement("div");
  host.innerHTML = renderMarkdown(markdown);
  if (baseDir) {
    host.querySelectorAll("img").forEach((img) => {
      const src = (img.getAttribute("src") || "").trim();
      if (!src || /^(https?:|data:|\/api\/)/i.test(src)) return; // 外链 / data / 已是接口
      if (/^([a-zA-Z]:[\\/]|[\\/])/.test(src)) {
        // 绝对路径：直接交给后端 /api/asset（_resolve_within 会做 workspace 边界校验）
        img.setAttribute("src", "/api/asset?path=" + encodeURIComponent(src));
        return;
      }
      // 相对路径：仅去掉一个前导 ./，保留 ../ 语义交给后端兜底
      const rel = src.replace(/^\.\//, "");
      const joined = baseDir.replace(/[\\/]+$/, "") + "/" + rel;
      img.setAttribute("src", "/api/asset?path=" + encodeURIComponent(joined));
    });
  }
  return host.innerHTML;
}

/* ------------------------------ 指标 ------------------------------ */
async function refreshStats() {
  try {
    const s = await apiGet("/api/stats");
    animateNumber($("m-nodes"), s.paper_nodes ?? 0);
    animateNumber($("m-edges"), s.paper_edges ?? 0);
    animateNumber($("m-episodes"), s.episodes ?? 0);
    animateNumber($("m-skills"), s.skills ?? 0);
    animateNumber($("m-vectors"), s.vectors ?? 0);
  } catch (err) { /* 指标失败不阻塞页面 */ }
}

/* ------------------------------ 搜索 ------------------------------ */
function renderSearchResults(items) {
  const host = $("search-results");
  if (!items || !items.length) { host.innerHTML = emptyState("🔍", "没有搜到结果", "换个英文关键词试试"); return; }
  host.innerHTML = items.map((it) => `
    <div class="list-item">
      <div class="li-title">${escapeHtml(it.title || "")}</div>
      <div class="li-meta">${escapeHtml((it.authors || []).slice(0, 4).join(", "))}${it.published ? " · " + escapeHtml(String(it.published).slice(0, 4)) : ""}</div>
      <div class="li-meta" style="margin-top:6px;">${escapeHtml((it.abstract || "").slice(0, 220))}…</div>
      <div class="li-actions">
        <button class="btn btn-ghost btn-sm js-fill" data-url="${escapeHtml(it.url || "")}" data-title="${escapeHtml(it.title || "")}">填入分析框</button>
        <button class="btn btn-ghost btn-sm js-open" data-url="${escapeHtml(it.url || "")}">打开链接</button>
      </div>
    </div>`).join("");
  host.querySelectorAll(".js-fill").forEach((b) => b.addEventListener("click", () => {
    $("an-source").value = b.dataset.url || "";
    if (!$("an-title").value) $("an-title").value = b.dataset.title || "";
    toast("已填入分析框", "good");
  }));
  host.querySelectorAll(".js-open").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.url) window.open(b.dataset.url, "_blank");
  }));
}
async function runSearch(query) {
  const q = (query != null ? query : $("search-q").value).trim();
  if (!q) { setStatus("st-search", "请先输入关键词", "bad"); return; }
  $("search-q").value = q;
  setStatus("st-search", "搜索中…");
  skeleton($("search-results"), 3);
  busy($("btn-search"), true, "搜索中…");
  try {
    const data = await apiPost("/api/search", { query: q });
    renderSearchResults(data);
    setStatus("st-search", `共 ${data.length} 条结果`, "good");
  } catch (err) {
    setStatus("st-search", String(err.message || err), "bad");
    $("search-results").innerHTML = emptyState("⚠️", "搜索失败", String(err.message || err));
  } finally { busy($("btn-search"), false); }
}

/* ------------------------------ 分析 ------------------------------ */
function fmtList(items, emptyText) {
  if (!items || !items.length) return `<li class="li-empty">${escapeHtml(emptyText)}</li>`;
  return items.map((it) => `<li>${escapeHtml(typeof it === "string" ? it : JSON.stringify(it))}</li>`).join("");
}
function renderAnalysis(data) {
  const r = data.result || {};
  const meta = `
    <div class="meta-row">
      <span class="tag">${escapeHtml(r._analysis_mode || "—")}</span>
      <span class="tag">来源: ${escapeHtml(r._source || "—")}</span>
      ${data.local ? '<span class="tag">本地 PDF</span>' : ""}
    </div>`;
  // 用 Markdown 三级标题 + 原始文本，交给 renderMarkdown 统一转义渲染（避免字面标签/双重转义）
  const sections = [
    ["一句话总结", r.tldr],
    ["核心问题", r.problem],
    ["方法概述", r.method_summary],
    ["实验结果", r.results],
    ["未来方向", r.future_work],
  ].filter(([, v]) => v).map(([h, v]) => `### ${h}\n\n${v}`).join("\n\n");
  const noteLine = r._note_path ? `\n\n> 笔记已保存：\`${r._note_path}\`` : "";
  $("analysis-output").innerHTML = `<div class="markdown-body">${meta}<h2 style="margin-top:4px;">${escapeHtml(data.title || "")}</h2>${renderMarkdown(sections + noteLine)}</div>`;

  $("an-contribs").innerHTML = fmtList(r.contributions || [], "暂无");
  $("an-datasets").innerHTML = fmtList((r.datasets || []).map((it) => {
    if (typeof it === "string") return it;
    return (it.name || "Unknown") + (it.used_for ? " / " + it.used_for : "") + (it.size ? " / " + it.size : "");
  }), "暂无");
  $("an-memory").innerHTML = fmtList((r.memory_connections || []).map((it) =>
    (it.title || "Unknown") + " · " + (it.relation || "") + " · " + Number(it.confidence || 0).toFixed(2)), "暂无高置信记忆连接");
  $("an-related").innerHTML = fmtList((r.cited_similar_work || []).map((it) =>
    typeof it === "string" ? it : (it.title || "Unknown") + (it.category ? " [" + it.category + "]" : "")), "暂无");
}
async function runAnalyze() {
  const source = $("an-source").value.trim();
  if (!source) { setStatus("st-analyze", "请先输入论文来源", "bad"); return; }
  setStatus("st-analyze", "正在分析，可能需要几十秒到几分钟…");
  skeleton($("analysis-output"), 5, true);
  busy($("btn-analyze"), true, "分析中…");
  try {
    const data = await apiPost("/api/analyze", {
      source, title: $("an-title").value.trim(),
      focus: $("an-focus").value.trim(), mode: $("an-mode").value,
    });
    renderAnalysis(data);
    setStatus("st-analyze", "分析完成，笔记与记忆已更新", "good");
    toast("分析完成 ✓", "good");
    await refreshStats();
    await loadNotes($("notes-q").value.trim());
    if (!$("graph-q").value.trim()) $("graph-q").value = $("an-focus").value.trim() || $("an-title").value.trim() || source;
    await loadGraph($("graph-q").value.trim());
  } catch (err) {
    setStatus("st-analyze", String(err.message || err), "bad");
    $("analysis-output").innerHTML = emptyState("⚠️", "分析失败", String(err.message || err));
    toast("分析失败", "bad");
  } finally { busy($("btn-analyze"), false); }
}

/* ------------------------------ 笔记 ------------------------------ */
function renderNotes(items) {
  const host = $("note-results");
  if (!items || !items.length) { host.innerHTML = emptyState("📭", "没有匹配到笔记", "先分析一篇论文"); return; }
  host.innerHTML = items.map((it) => `
    <div class="list-item">
      <div class="li-title">${escapeHtml(it.title || "")}</div>
      <div class="li-meta">${escapeHtml(it.path || "")}</div>
      <div class="li-meta" style="margin-top:6px;">${escapeHtml((it.preview || "").slice(0, 180))}</div>
      <div class="li-actions"><button class="btn btn-ghost btn-sm js-note" data-path="${escapeHtml(it.path || "")}">查看笔记</button></div>
    </div>`).join("");
  host.querySelectorAll(".js-note").forEach((b) => b.addEventListener("click", () =>
    openNote(b.dataset.path || "").catch((e) => toast(String(e.message || e), "bad"))));
}
async function loadNotes(query = "") {
  try {
    const data = await apiGet("/api/notes" + (query ? "?q=" + encodeURIComponent(query) : ""));
    renderNotes(data);
  } catch (err) { $("note-results").innerHTML = emptyState("⚠️", "加载笔记失败", String(err.message || err)); }
}
async function openNote(path) {
  if (!path) throw new Error("笔记路径为空");
  const data = await apiPost("/api/open-note", { path });
  const baseDir = data.dir || (data.path || "").replace(/[\\/][^\\/]*$/, "");
  $("note-viewer").innerHTML = `<div class="markdown-body">${renderMarkdownWithAssets(data.content || "", baseDir)}</div>`;
  $("note-viewer").scrollTop = 0;
  $("note-viewer").scrollIntoView({ behavior: "smooth", block: "start" });
  toast("笔记已加载", "good");
}

/* ------------------------------ 图谱（d3 力导向 + 降级） ------------------------------ */
const graphState = { sim: null, zoom: null, svg: null };

function edgeColor(type) { return EDGE_COLORS[type] || EDGE_COLORS.similar_to; }
function nodeColor(n) { return n.metadata && n.metadata.placeholder ? NODE_PLACEHOLDER : NODE_COLOR; }
function nodeLabel(n) { return n.method_name || n.title || n.paper_id || ""; }

function showNodeDetail(n) {
  const el = $("graph-node-detail");
  el.className = "node-detail";
  el.innerHTML = `
    <span class="pill">${escapeHtml(n.method_name || "Paper")}</span>
    <div class="nd-title">${escapeHtml(n.title || n.paper_id || "")}</div>
    <div class="li-meta" style="margin-top:6px;">id: ${escapeHtml(n.paper_id || "")}</div>
    <div class="li-meta" style="margin-top:6px;">${escapeHtml(n.problem || n.tldr || "暂无摘要")}</div>
    ${n.note_path ? `<div class="li-actions"><button class="btn btn-ghost btn-sm" id="nd-open">打开笔记</button></div>` : ""}`;
  const open = $("nd-open");
  if (open) open.addEventListener("click", () => openNote(n.note_path).catch((e) => toast(String(e.message || e), "bad")));
}
function setGraphStats(snapshot, nNodes, nEdges) {
  const s = snapshot.stats || {};
  $("graph-stats").textContent = [
    `显示节点: ${nNodes}`,
    `显示边: ${nEdges}`,
    `已存论文节点: ${s.paper_nodes ?? "-"}`,
    `已存论文边: ${s.paper_edges ?? "-"}`,
    `episodes: ${s.episodes ?? "-"}`,
    `vectors: ${s.vectors ?? "-"}`,
  ].join("\n");
}

function renderGraphD3(snapshot) {
  const d3 = window.d3;
  const svgEl = $("graph-stage");
  const w = svgEl.clientWidth || 760;
  const h = svgEl.clientHeight || 560;

  if (graphState.sim) { graphState.sim.stop(); graphState.sim = null; }
  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();

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

  // 度数 → 半径
  const deg = {};
  links.forEach((l) => { deg[l.source] = (deg[l.source] || 0) + 1; deg[l.target] = (deg[l.target] || 0) + 1; });
  const radius = (n) => 16 + Math.min(14, (deg[n.id] || 0) * 3);

  const root = svg.append("g");
  const zoom = d3.zoom().scaleExtent([0.3, 3]).on("zoom", (ev) => root.attr("transform", ev.transform));
  svg.call(zoom).on("dblclick.zoom", null);
  graphState.svg = svg; graphState.zoom = zoom;

  const linkSel = root.append("g").selectAll("line").data(links).join("line")
    .attr("class", "gedge").attr("stroke", (d) => edgeColor(d.type))
    .attr("stroke-width", (d) => 1.2 + d.strength * 2.2);
  const linkLabel = root.append("g").selectAll("text").data(links).join("text")
    .attr("class", "gedge-label").attr("text-anchor", "middle").text((d) => d.type || "");

  const nodeSel = root.append("g").selectAll("g.gnode").data(nodes).join("g").attr("class", "gnode")
    .style("cursor", "pointer");
  nodeSel.append("circle").attr("class", "halo").attr("r", (d) => radius(d) + 7)
    .attr("fill", "none").attr("stroke", (d) => nodeColor(d)).attr("stroke-opacity", 0.18).attr("stroke-width", 8);
  nodeSel.append("circle").attr("class", "core").attr("r", radius).attr("fill", (d) => nodeColor(d)).attr("fill-opacity", 0.95);
  nodeSel.append("text").attr("text-anchor", "middle").attr("dy", (d) => radius(d) + 15)
    .text((d) => { const t = nodeLabel(d); return t.length > 16 ? t.slice(0, 15) + "…" : t; });

  // 邻接表 → hover 高亮
  const adj = {};
  links.forEach((l) => { (adj[l.source] = adj[l.source] || new Set()).add(l.target); (adj[l.target] = adj[l.target] || new Set()).add(l.source); });
  nodeSel.on("mouseenter", (ev, d) => {
    nodeSel.classed("dim", (o) => o.id !== d.id && !(adj[d.id] && adj[d.id].has(o.id)));
    linkSel.classed("dim", (o) => o.source.id !== d.id && o.target.id !== d.id);
    linkLabel.classed("dim", (o) => o.source.id !== d.id && o.target.id !== d.id);
  }).on("mouseleave", () => {
    nodeSel.classed("dim", false); linkSel.classed("dim", false); linkLabel.classed("dim", false);
  }).on("click", (ev, d) => { ev.stopPropagation(); showNodeDetail(d); });

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
  const pos = {};
  const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.36;
  nodes.forEach((n, idx) => { const a = (idx / nodes.length) * Math.PI * 2; pos[n.paper_id] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) }; });
  const edges = (snapshot.edges || []).map((e) => {
    const a = pos[e.src_paper_id], b = pos[e.dst_paper_id];
    if (!a || !b) return "";
    return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${edgeColor(e.relation_type)}" stroke-width="1.6" stroke-opacity="0.6"/>`;
  }).join("");
  const circles = nodes.map((n) => {
    const p = pos[n.paper_id]; const t = nodeLabel(n);
    return `<g class="gnode" data-id="${escapeHtml(n.paper_id)}" transform="translate(${p.x},${p.y})" style="cursor:pointer;">
      <circle r="20" fill="${nodeColor(n)}" fill-opacity="0.95" stroke="#fff" stroke-width="3"/>
      <text text-anchor="middle" dy="34" font-size="11" fill="#94a3b8">${escapeHtml(t.length > 16 ? t.slice(0, 15) + "…" : t)}</text></g>`;
  }).join("");
  svgEl.innerHTML = `<g>${edges}${circles}</g>`;
  svgEl.querySelectorAll(".gnode").forEach((g) => g.addEventListener("click", () => {
    const n = nodes.find((x) => x.paper_id === g.dataset.id); if (n) showNodeDetail(n);
  }));
}

async function loadGraph(query = "") {
  $("graph-engine").textContent = window.d3 ? "d3 force" : "fallback layout";
  try {
    const snap = await apiGet("/api/graph" + (query ? "?q=" + encodeURIComponent(query) : ""));
    if (window.d3) renderGraphD3(snap); else renderGraphFallback(snap);
  } catch (err) {
    $("graph-stats").textContent = "加载失败: " + (err.message || err);
    toast("图谱加载失败", "bad");
  }
}
function zoomBy(factor) {
  if (graphState.svg && graphState.zoom) graphState.svg.transition().duration(250).call(graphState.zoom.scaleBy, factor);
}
function resetGraph() {
  if (graphState.svg && graphState.zoom && window.d3) graphState.svg.transition().duration(300).call(graphState.zoom.transform, window.d3.zoomIdentity);
}

/* ------------------------------ 图表工作室 ------------------------------ */
function buildFigureCard(fig) {
  const card = document.createElement("div");
  card.className = "figure-card";
  const kind = fig.kind || "svg";
  const badge = `<span class="kind-badge kind-${kind}">${kind}</span>`;
  const body = document.createElement("div");
  body.className = "fc-body";
  let zoomable = true;

  if (kind === "image") {
    const src = (fig.image && (fig.image.data_uri || fig.image.url)) || "";
    body.innerHTML = `<img alt="${escapeHtml(fig.title || "")}" src="${escapeHtml(src)}" />`;
  } else if (kind === "latex") {
    const rendered = fig.latex && fig.latex.rendered_svg;
    if (rendered) { body.innerHTML = sanitizeSvg(rendered); }
    else { body.style.cursor = "default"; zoomable = false;
      body.innerHTML = `<pre class="latex-source">${escapeHtml((fig.latex && fig.latex.source) || "")}</pre>`; }
  } else {
    body.innerHTML = sanitizeSvg(fig.svg);
  }

  const foot = document.createElement("div");
  foot.className = "fc-foot";
  if (fig.download_path) {
    foot.innerHTML = `<a class="btn btn-ghost btn-sm" href="/api/asset?path=${encodeURIComponent(fig.download_path)}" download>下载文件</a>`;
  }
  if (zoomable) {
    const zb = document.createElement("button");
    zb.className = "btn btn-ghost btn-sm"; zb.textContent = "放大查看";
    zb.addEventListener("click", () => openLightbox(body.innerHTML));
    foot.appendChild(zb);
  }
  if (zoomable) body.addEventListener("click", () => openLightbox(body.innerHTML));

  card.innerHTML = `<div class="fc-head"><span class="fc-title">${escapeHtml(fig.title || "图")}</span>${badge}</div>`;
  card.appendChild(body);
  card.appendChild(foot);
  if (fig.meta && fig.meta.note) {
    const note = document.createElement("div"); note.className = "fc-note"; note.textContent = fig.meta.note;
    card.appendChild(note);
  }
  return card;
}
function renderFigures(container, figures) {
  container.innerHTML = "";
  (figures || []).forEach((f) => container.appendChild(buildFigureCard(f)));
}
async function runSurvey() {
  const topic = $("survey-topic").value.trim();
  if (!topic) { setStatus("st-survey", "请先输入综述主题", "bad"); return; }
  const n = parseInt($("survey-n").value, 10) || 10;
  setStatus("st-survey", "正在检索论文并生成综述与图，可能需要 30s–2min…");
  skeleton($("survey-figures"), 0, false);
  $("survey-figures").innerHTML = `<div class="skeleton sk-block"></div><div class="skeleton sk-block"></div>`;
  busy($("btn-survey"), true, "生成中…");
  try {
    const data = await apiPost("/api/survey", { topic, max_papers: n });
    renderFigures($("survey-figures"), data.figures);
    const report = $("survey-report");
    report.style.display = "block";
    report.innerHTML = `<div class="markdown-body">${renderMarkdownWithAssets(data.report_md || "", data.output_dir)}</div>`;
    setStatus("st-survey", `已生成：${(data.papers || []).length} 篇论文 · 2 张 SVG 图`, "good");
    toast("综述与图已生成 ✓", "good");
  } catch (err) {
    setStatus("st-survey", String(err.message || err), "bad");
    $("survey-figures").innerHTML = emptyState("⚠️", "生成失败", String(err.message || err));
    toast("综述生成失败", "bad");
  } finally { busy($("btn-survey"), false); }
}
async function runFigMode(mode, btn) {
  setStatus("st-figmode", `生成 ${mode} 图产物中…`);
  busy(btn, true, "生成中…");
  try {
    const fig = await apiPost("/api/figure", { mode, title: `${mode.toUpperCase()} 示例` });
    renderFigures($("figmode-out"), [fig]);
    setStatus("st-figmode", `已渲染 kind=${fig.kind} 的图产物`, "good");
  } catch (err) {
    setStatus("st-figmode", String(err.message || err), "bad");
  } finally { busy(btn, false); }
}

/* ------------------------------ Lightbox ------------------------------ */
function openLightbox(innerHtml) {
  $("lightbox-inner").innerHTML = innerHtml;
  $("lightbox").classList.add("open");
}
function closeLightbox() { $("lightbox").classList.remove("open"); $("lightbox-inner").innerHTML = ""; }

/* ------------------------------ Cmd-K ------------------------------ */
function openCmdk() { $("cmdk").classList.add("open"); setTimeout(() => $("cmdk-input").focus(), 30); }
function closeCmdk() { $("cmdk").classList.remove("open"); }

/* ------------------------------ 启动 ------------------------------ */
function wireEvents() {
  $("theme-toggle").addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

  $("btn-analyze").addEventListener("click", runAnalyze);
  $("btn-search").addEventListener("click", () => runSearch());
  $("search-q").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  $("btn-refresh-graph").addEventListener("click", () => loadGraph($("graph-q").value.trim()));
  $("btn-refresh-notes").addEventListener("click", () => loadNotes($("notes-q").value.trim()));
  $("btn-filter-notes").addEventListener("click", () => loadNotes($("notes-q").value.trim()));
  $("notes-q").addEventListener("keydown", (e) => { if (e.key === "Enter") loadNotes($("notes-q").value.trim()); });

  $("btn-query-graph").addEventListener("click", () => loadGraph($("graph-q").value.trim()));
  $("graph-q").addEventListener("keydown", (e) => { if (e.key === "Enter") loadGraph($("graph-q").value.trim()); });
  $("g-zoom-in").addEventListener("click", () => zoomBy(1.3));
  $("g-zoom-out").addEventListener("click", () => zoomBy(1 / 1.3));
  $("g-reset").addEventListener("click", resetGraph);

  document.querySelectorAll(".studio-tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".studio-tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".studio-pane").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("pane-" + tab.dataset.pane).classList.add("active");
  }));
  $("btn-survey").addEventListener("click", runSurvey);
  $("survey-topic").addEventListener("keydown", (e) => { if (e.key === "Enter") runSurvey(); });
  document.querySelectorAll("[data-figmode]").forEach((b) =>
    b.addEventListener("click", () => runFigMode(b.dataset.figmode, b)));

  $("open-cmdk").addEventListener("click", openCmdk);
  $("cmdk-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { const v = e.target.value.trim(); closeCmdk(); if (v) runSearch(v); }
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
    window.__graphResize = setTimeout(() => loadGraph($("graph-q").value.trim()), 300);
  });
}

async function boot() {
  initTheme();
  wireEvents();
  await refreshStats();
  await loadNotes("");
  await loadGraph("");
}
document.addEventListener("DOMContentLoaded", () => { boot().catch((e) => toast(String(e.message || e), "bad")); });
