import json
import threading
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .orchestrator import ResearchOrchestrator


def build_gui_app(orchestrator: ResearchOrchestrator) -> ThreadingHTTPServer:
    class GuiHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            try:
                if path == "/":
                    self._send_html(_build_index_html())
                    return
                if path == "/api/health":
                    self._send_json({"ok": True})
                    return
                if path == "/api/stats":
                    self._send_json({"ok": True, "data": orchestrator.memory_stats()})
                    return
                if path == "/api/notes":
                    keyword = (query.get("q", [""])[0] or "").strip()
                    notes = orchestrator.memory.find_paper_notes(keyword, top_k=50)
                    self._send_json({"ok": True, "data": notes})
                    return
                if path == "/api/note":
                    note_path = (query.get("path", [""])[0] or "").strip()
                    if not note_path:
                        self._send_json({"ok": False, "error": "missing path"}, status=400)
                        return
                    file_path = Path(note_path)
                    if not file_path.exists():
                        self._send_json({"ok": False, "error": "note not found"}, status=404)
                        return
                    content = file_path.read_text(encoding="utf-8")
                    self._send_json(
                        {
                            "ok": True,
                            "data": {
                                "path": str(file_path),
                                "name": file_path.name,
                                "content": content,
                            },
                        }
                    )
                    return
                if path == "/api/graph":
                    keyword = (query.get("q", [""])[0] or "").strip()
                    snapshot = orchestrator.memory.get_paper_graph_snapshot(
                        query=keyword,
                        node_limit=40,
                        edge_limit=120,
                        include_neighbors=True,
                    )
                    self._send_json({"ok": True, "data": snapshot})
                    return
                self._send_json({"ok": False, "error": f"unknown path: {path}"}, status=404)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            try:
                payload = self._read_json_body()

                if path == "/api/search":
                    query = str(payload.get("query", "")).strip()
                    if not query:
                        self._send_json({"ok": False, "error": "query is required"}, status=400)
                        return
                    papers = orchestrator.search_papers(query)
                    data = [
                        {
                            "paper_id": p.paper_id,
                            "title": p.title,
                            "authors": p.authors,
                            "abstract": p.abstract,
                            "url": p.url,
                            "published": p.published,
                        }
                        for p in papers
                    ]
                    self._send_json({"ok": True, "data": data})
                    return

                if path == "/api/analyze":
                    source = str(payload.get("source", "")).strip()
                    focus = str(payload.get("focus", "")).strip() or None
                    title = str(payload.get("title", "")).strip() or None
                    mode = str(payload.get("mode", "read-paper")).strip()

                    if not source:
                        self._send_json({"ok": False, "error": "source is required"}, status=400)
                        return

                    possible_file = Path(source).expanduser()
                    if possible_file.exists() and possible_file.is_file():
                        result = orchestrator.analyzer.analyze_local_pdf(
                            possible_file.resolve(),
                            title=title,
                            focus=focus,
                        )
                        display_title = title or orchestrator.analyzer.fetcher.infer_title_from_pdf(possible_file)
                        self._send_json(
                            {"ok": True, "data": {"title": display_title, "result": result, "local": True}}
                        )
                        return

                    paper_id = source.rstrip("/").split("/")[-1].replace(".pdf", "")
                    papers = orchestrator.arxiv.search(f"id:{paper_id}", max_results=1)
                    if not papers:
                        self._send_json({"ok": False, "error": "paper not found"}, status=404)
                        return

                    paper = papers[0]
                    result = (
                        orchestrator.analyze_paper_multimodal(paper, focus)
                        if mode == "read-paper"
                        else orchestrator.analyze_paper(paper, focus)
                    )
                    self._send_json(
                        {"ok": True, "data": {"title": paper.title, "result": result, "local": False}}
                    )
                    return

                self._send_json({"ok": False, "error": f"unknown path: {path}"}, status=404)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    return ThreadingHTTPServer(("127.0.0.1", 0), GuiHandler)


def launch_gui(orchestrator: ResearchOrchestrator, open_browser: bool = True) -> str:
    server = build_gui_app(orchestrator)
    host, port = server.server_address
    url = f"http://{host}:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return url


def _build_index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deep Research Agent GUI</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.86);
      --panel-strong: #fffdf8;
      --ink: #1e1c18;
      --muted: #70685f;
      --line: rgba(65, 52, 37, 0.16);
      --brand: #0f766e;
      --brand-2: #c2410c;
      --accent: #164e63;
      --good: #166534;
      --bad: #b91c1c;
      --shadow: 0 22px 50px rgba(59, 39, 23, 0.12);
      --radius: 22px;
      --mono: "JetBrains Mono", "Consolas", monospace;
      --sans: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 28%),
        linear-gradient(180deg, #f8f2e8 0%, #efe6d8 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1520px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .card {
      background: var(--panel);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-main {
      padding: 26px 28px;
      min-height: 170px;
      position: relative;
      overflow: hidden;
    }

    .hero-main::after {
      content: "";
      position: absolute;
      right: -40px;
      top: -40px;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(15, 118, 110, 0.22), transparent 70%);
    }

    h1 {
      margin: 0 0 10px;
      font-size: clamp(28px, 3vw, 46px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }

    .hero-main p {
      margin: 0;
      max-width: 760px;
      font-size: 15px;
      line-height: 1.7;
      color: var(--muted);
    }

    .hero-side {
      padding: 22px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-content: start;
    }

    .metric {
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(22, 78, 99, 0.10);
      border-radius: 18px;
      padding: 16px;
      min-height: 94px;
      transition: transform 180ms ease, background 180ms ease;
    }

    .metric:hover {
      transform: translateY(-2px);
      background: rgba(255,255,255,0.78);
    }

    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric .value {
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
    }

    .layout {
      display: grid;
      grid-template-columns: 440px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .stack {
      display: grid;
      gap: 18px;
    }

    .panel {
      padding: 18px;
    }

    .panel h2 {
      margin: 0 0 14px;
      font-size: 20px;
    }

    .panel h3 {
      margin: 8px 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    .controls {
      display: grid;
      gap: 12px;
    }

    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }

    input, textarea, select, button {
      font: inherit;
    }

    input, textarea, select {
      width: 100%;
      border: 1px solid rgba(30, 28, 24, 0.12);
      background: rgba(255, 255, 255, 0.82);
      border-radius: 14px;
      padding: 12px 14px;
      color: var(--ink);
      transition: border-color 160ms ease, box-shadow 160ms ease;
    }

    input:focus, textarea:focus, select:focus {
      outline: none;
      border-color: rgba(15, 118, 110, 0.52);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.10);
    }

    textarea {
      min-height: 86px;
      resize: vertical;
      line-height: 1.6;
    }

    .button-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    button {
      border: none;
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      transition: transform 160ms ease, opacity 160ms ease, box-shadow 160ms ease;
      box-shadow: 0 10px 18px rgba(15, 118, 110, 0.15);
    }

    button:hover {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.6;
      cursor: wait;
      transform: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--brand), var(--accent));
      color: white;
    }

    .btn-secondary {
      background: linear-gradient(135deg, #f59e0b, var(--brand-2));
      color: white;
    }

    .btn-ghost {
      background: rgba(255,255,255,0.8);
      color: var(--ink);
      box-shadow: none;
      border: 1px solid rgba(30, 28, 24, 0.1);
    }

    .status {
      min-height: 24px;
      font-size: 13px;
      color: var(--muted);
    }

    .status.good { color: var(--good); }
    .status.bad { color: var(--bad); }

    .results {
      display: grid;
      gap: 18px;
    }

    .paper-list, .note-list {
      display: grid;
      gap: 10px;
      max-height: 300px;
      overflow: auto;
      padding-right: 4px;
    }

    .paper-item, .note-item {
      border: 1px solid rgba(22, 78, 99, 0.10);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.62);
      transition: transform 160ms ease, background 160ms ease;
    }

    .paper-item:hover, .note-item:hover {
      transform: translateY(-1px);
      background: rgba(255,255,255,0.8);
    }

    .paper-item .title, .note-item .title {
      font-weight: 700;
      margin-bottom: 8px;
      line-height: 1.45;
    }

    .mini {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    .analysis-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      align-items: start;
    }

    .analysis-body {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      line-height: 1.8;
      white-space: pre-wrap;
      min-height: 520px;
      overflow: auto;
    }

    .analysis-side {
      display: grid;
      gap: 14px;
    }

    .side-box {
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(22, 78, 99, 0.10);
      border-radius: 18px;
      padding: 16px;
    }

    .side-box ul {
      margin: 0;
      padding-left: 18px;
    }

    .side-box li {
      margin: 6px 0;
      line-height: 1.6;
    }

    .graph-wrap {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 18px;
      align-items: start;
    }

    .graph-surface {
      position: relative;
      min-height: 680px;
      background:
        radial-gradient(circle at center, rgba(255,255,255,0.85), rgba(248, 244, 237, 0.92));
      border: 1px solid var(--line);
      border-radius: 20px;
      overflow: hidden;
    }

    .graph-stage {
      width: 100%;
      height: 680px;
      display: block;
      cursor: grab;
    }

    .graph-stage:active {
      cursor: grabbing;
    }

    .graph-toolbar {
      position: absolute;
      left: 14px;
      top: 14px;
      display: flex;
      gap: 8px;
      z-index: 2;
    }

    .graph-toolbar button {
      padding: 8px 12px;
      border-radius: 999px;
      box-shadow: none;
      background: rgba(255,255,255,0.88);
    }

    .graph-info {
      display: grid;
      gap: 12px;
      align-content: start;
    }

    .legend {
      display: grid;
      gap: 10px;
      font-size: 13px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
    }

    .dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      display: inline-block;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.10);
      color: var(--brand);
      font-size: 12px;
      font-weight: 600;
    }

    .codebox {
      font-family: var(--mono);
      font-size: 12px;
      white-space: pre-wrap;
      line-height: 1.7;
      background: rgba(255,255,255,0.62);
      border-radius: 14px;
      padding: 12px;
      border: 1px solid rgba(22, 78, 99, 0.08);
    }

    .empty {
      color: var(--muted);
      font-size: 14px;
      padding: 20px 0 8px;
    }

    @media (max-width: 1180px) {
      .layout, .analysis-grid, .graph-wrap, .hero {
        grid-template-columns: 1fr;
      }

      .graph-info {
        order: -1;
      }
    }

    @media (max-width: 760px) {
      .hero-side {
        grid-template-columns: 1fr 1fr;
      }

      .shell {
        padding: 18px 12px 28px;
      }

      .panel, .hero-main, .hero-side {
        padding: 16px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="card hero-main">
        <h1>Deep Research Agent<br/>Paper Workspace</h1>
        <p>把论文搜索、保守版分析、多模态细读、已读笔记浏览和论文图关系可视化放到同一个本地页面里。左侧负责输入和调用，右侧负责结果和图谱，让你更像是在操作一个“研究工作台”，而不是一堆零散命令。</p>
      </div>
      <div class="card hero-side">
        <div class="metric">
          <div class="label">Paper Nodes</div>
          <div class="value" id="metric-nodes">-</div>
        </div>
        <div class="metric">
          <div class="label">Paper Edges</div>
          <div class="value" id="metric-edges">-</div>
        </div>
        <div class="metric">
          <div class="label">Episodes</div>
          <div class="value" id="metric-episodes">-</div>
        </div>
        <div class="metric">
          <div class="label">Vectors</div>
          <div class="value" id="metric-vectors">-</div>
        </div>
      </div>
    </section>

    <section class="layout">
      <div class="stack">
        <div class="card panel">
          <h2>论文工作台</h2>
          <div class="controls">
            <div>
              <label for="source">论文来源</label>
              <input id="source" placeholder="本地 PDF 路径，或 arXiv ID / 链接" />
            </div>
            <div>
              <label for="title">本地 PDF 可选标题</label>
              <input id="title" placeholder="仅本地 PDF 需要时填写" />
            </div>
            <div>
              <label for="focus">关注点</label>
              <textarea id="focus" placeholder="例如：方法细节、实验设置、和已有 VLA 工作的关系"></textarea>
            </div>
            <div>
              <label for="mode">分析模式</label>
              <select id="mode">
                <option value="read-paper">read-paper 多模态/细读</option>
                <option value="analyze">analyze 保守版</option>
              </select>
            </div>
            <div class="button-row">
              <button class="btn-primary" id="btn-analyze">开始分析</button>
              <button class="btn-ghost" id="btn-load-graph">刷新图谱</button>
              <button class="btn-ghost" id="btn-load-notes">刷新笔记</button>
            </div>
            <div class="status" id="action-status"></div>
          </div>
        </div>

        <div class="card panel">
          <h2>论文搜索</h2>
          <div class="controls">
            <div>
              <label for="search-query">关键词</label>
              <input id="search-query" placeholder="例如：latent world model for robot policy" />
            </div>
            <div class="button-row">
              <button class="btn-secondary" id="btn-search">搜索论文</button>
            </div>
            <div class="status" id="search-status"></div>
            <div class="paper-list" id="search-results"></div>
          </div>
        </div>

        <div class="card panel">
          <h2>已读笔记</h2>
          <div class="controls">
            <div>
              <label for="notes-query">筛选关键词</label>
              <input id="notes-query" placeholder="留空则按最新笔记显示" />
            </div>
            <div class="button-row">
              <button class="btn-ghost" id="btn-filter-notes">筛选笔记</button>
            </div>
            <div class="note-list" id="note-results"></div>
          </div>
        </div>
      </div>

      <div class="stack results">
        <div class="card panel">
          <h2>分析结果</h2>
          <div class="analysis-grid">
            <div class="analysis-body" id="analysis-output">这里会显示论文分析、保存路径和正文摘要。</div>
            <div class="analysis-side">
              <div class="side-box">
                <h3>主要贡献</h3>
                <ul id="analysis-contribs"><li>等待分析结果</li></ul>
              </div>
              <div class="side-box">
                <h3>数据集</h3>
                <ul id="analysis-datasets"><li>等待分析结果</li></ul>
              </div>
              <div class="side-box">
                <h3>记忆连接</h3>
                <ul id="analysis-memory"><li>等待分析结果</li></ul>
              </div>
              <div class="side-box">
                <h3>引用中的相似工作</h3>
                <ul id="analysis-related"><li>等待分析结果</li></ul>
              </div>
            </div>
          </div>
        </div>

        <div class="card panel">
          <h2>论文图关系可视化</h2>
          <div class="controls" style="margin-bottom: 14px;">
            <div>
              <label for="graph-query">图谱检索词</label>
              <input id="graph-query" placeholder="例如：VLA / DPO / 世界模型，留空显示最近节点" />
            </div>
            <div class="button-row">
              <button class="btn-primary" id="btn-query-graph">查询图谱</button>
            </div>
          </div>

          <div class="graph-wrap">
            <div class="graph-surface">
              <div class="graph-toolbar">
                <button type="button" id="btn-zoom-in">放大</button>
                <button type="button" id="btn-zoom-out">缩小</button>
                <button type="button" id="btn-reset-graph">重置</button>
              </div>
              <svg class="graph-stage" id="graph-stage" viewBox="0 0 980 680"></svg>
            </div>
            <div class="graph-info">
              <div class="side-box">
                <h3>图谱说明</h3>
                <div class="legend">
                  <div class="legend-item"><span class="dot" style="background:#0f766e;"></span> 已读论文节点</div>
                  <div class="legend-item"><span class="dot" style="background:#b45309;"></span> 引用占位节点</div>
                  <div class="legend-item"><span class="dot" style="background:#164e63;"></span> `builds_on`</div>
                  <div class="legend-item"><span class="dot" style="background:#c2410c;"></span> `compares_with`</div>
                  <div class="legend-item"><span class="dot" style="background:#6b7280;"></span> `similar_to`</div>
                </div>
              </div>
              <div class="side-box">
                <h3>选中节点</h3>
                <div id="graph-node-detail" class="empty">点击图中的节点查看详情。</div>
              </div>
              <div class="side-box">
                <h3>图谱统计</h3>
                <div class="codebox" id="graph-stats">等待加载</div>
              </div>
            </div>
          </div>
        </div>

        <div class="card panel">
          <h2>笔记查看器</h2>
          <div class="analysis-body" id="note-viewer">这里会显示选中笔记的原始 Markdown。</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const state = {
      graph: { nodes: [], edges: [], zoom: 1, panX: 0, panY: 0, positions: {} },
      drag: null,
      graphDrag: null,
    };

    const $ = (id) => document.getElementById(id);

    function setStatus(id, text, kind = "") {
      const el = $(id);
      el.textContent = text || "";
      el.className = "status" + (kind ? " " + kind : "");
    }

    async function apiGet(url) {
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || "request failed");
      }
      return data.data;
    }

    async function apiPost(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || "request failed");
      }
      return data.data;
    }

    function escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text == null ? "" : String(text);
      return div.innerHTML;
    }

    function formatList(items, emptyText) {
      if (!items || !items.length) {
        return "<li>" + escapeHtml(emptyText) + "</li>";
      }
      return items.map((item) => "<li>" + escapeHtml(typeof item === "string" ? item : JSON.stringify(item)) + "</li>").join("");
    }

    function renderAnalysis(data) {
      const title = data.title || "";
      const result = data.result || {};
      const content = [
        "标题: " + title,
        "分析模式: " + (result._analysis_mode || ""),
        "来源: " + (result._source || ""),
        result._note_path ? "笔记路径: " + result._note_path : "",
        "",
        "一句话总结:",
        result.tldr || "",
        "",
        "核心问题:",
        result.problem || "",
        "",
        "方法概述:",
        result.method_summary || "",
        "",
        "实验结果:",
        result.results || "",
        "",
        "未来方向:",
        result.future_work || "",
      ].filter(Boolean).join("\\n");

      $("analysis-output").textContent = content;
      $("analysis-contribs").innerHTML = formatList(result.contributions || [], "暂无");
      $("analysis-datasets").innerHTML = formatList(
        (result.datasets || []).map((item) => {
          if (typeof item === "string") return item;
          const name = item.name || "Unknown";
          const usedFor = item.used_for ? " / " + item.used_for : "";
          const size = item.size ? " / " + item.size : "";
          return name + usedFor + size;
        }),
        "暂无"
      );
      $("analysis-memory").innerHTML = formatList(
        (result.memory_connections || []).map((item) => {
          return (item.title || "Unknown") + " | " + (item.relation || "") + " | 置信度 " + Number(item.confidence || 0).toFixed(2);
        }),
        "暂无高置信度记忆连接"
      );
      $("analysis-related").innerHTML = formatList(
        (result.cited_similar_work || []).map((item) => {
          if (typeof item === "string") return item;
          return (item.title || "Unknown") + (item.category ? " [" + item.category + "]" : "");
        }),
        "暂无"
      );
    }

    function renderSearchResults(items) {
      const host = $("search-results");
      if (!items.length) {
        host.innerHTML = '<div class="empty">没有搜到结果。</div>';
        return;
      }
      host.innerHTML = items.map((item) => `
        <div class="paper-item">
          <div class="title">${escapeHtml(item.title || "")}</div>
          <div class="mini">${escapeHtml((item.authors || []).slice(0, 4).join(", "))}</div>
          <div class="mini" style="margin-top:8px;">${escapeHtml((item.abstract || "").slice(0, 260))}</div>
          <div class="button-row" style="margin-top:10px;">
            <button class="btn-ghost" onclick="fillPaperSource(${JSON.stringify(item.url || "")}, ${JSON.stringify(item.title || "")})">填入阅读框</button>
            <button class="btn-ghost" onclick="window.open(${JSON.stringify(item.url || "")}, '_blank')">打开链接</button>
          </div>
        </div>
      `).join("");
    }

    function renderNotes(items) {
      const host = $("note-results");
      if (!items.length) {
        host.innerHTML = '<div class="empty">没有匹配到笔记。</div>';
        return;
      }
      host.innerHTML = items.map((item) => `
        <div class="note-item">
          <div class="title">${escapeHtml(item.title || "")}</div>
          <div class="mini">${escapeHtml(item.path || "")}</div>
          <div class="mini" style="margin-top:8px;">${escapeHtml((item.preview || "").slice(0, 220))}</div>
          <div class="button-row" style="margin-top:10px;">
            <button class="btn-ghost" onclick="openNote(${JSON.stringify(item.path || "")})">查看笔记</button>
          </div>
        </div>
      `).join("");
    }

    function circleColor(node) {
      return node.metadata && node.metadata.placeholder ? "#b45309" : "#0f766e";
    }

    function edgeColor(edge) {
      if (edge.relation_type === "builds_on") return "#164e63";
      if (edge.relation_type === "compares_with") return "#c2410c";
      return "#6b7280";
    }

    function summarizeGraphStats(stats, nodes, edges) {
      return [
        "显示节点: " + nodes.length,
        "显示边: " + edges.length,
        "已存论文节点: " + (stats.paper_nodes ?? "-"),
        "已存论文边: " + (stats.paper_edges ?? "-"),
        "episodes: " + (stats.episodes ?? "-"),
        "vectors: " + (stats.vectors ?? "-")
      ].join("\\n");
    }

    function wrapTitle(text, max = 18) {
      const source = text || "";
      const lines = [];
      for (let i = 0; i < source.length; i += max) {
        lines.push(source.slice(i, i + max));
      }
      return lines.slice(0, 3);
    }

    function computeGraphLayout(nodes, width, height) {
      const positions = {};
      if (!nodes.length) return positions;
      const cols = Math.max(2, Math.ceil(Math.sqrt(nodes.length)));
      const gapX = width / (cols + 1);
      const rows = Math.ceil(nodes.length / cols);
      const gapY = height / (rows + 1);
      nodes.forEach((node, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        positions[node.paper_id] = {
          x: (col + 1) * gapX,
          y: (row + 1) * gapY + ((col % 2) ? 18 : -18),
        };
      });
      return positions;
    }

    function renderGraph(snapshot) {
      const svg = $("graph-stage");
      const nodes = snapshot.nodes || [];
      const edges = snapshot.edges || [];
      state.graph.nodes = nodes;
      state.graph.edges = edges;
      if (!Object.keys(state.graph.positions).length || state.graph.lastQuery !== (snapshot.query || "")) {
        state.graph.positions = computeGraphLayout(nodes, 920, 620);
      }
      state.graph.lastQuery = snapshot.query || "";

      const positions = state.graph.positions;
      const zoom = state.graph.zoom;
      const panX = state.graph.panX;
      const panY = state.graph.panY;

      const edgeMarkup = edges.map((edge) => {
        const a = positions[edge.src_paper_id];
        const b = positions[edge.dst_paper_id];
        if (!a || !b) return "";
        const labelX = (a.x + b.x) / 2;
        const labelY = (a.y + b.y) / 2;
        return `
          <g>
            <line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
                  stroke="${edgeColor(edge)}" stroke-width="${1.4 + (edge.relation_strength || 0.5) * 1.6}"
                  stroke-opacity="0.72" />
            <text x="${labelX}" y="${labelY - 6}" text-anchor="middle"
                  font-size="10" fill="${edgeColor(edge)}">${escapeHtml(edge.relation_type || "")}</text>
          </g>
        `;
      }).join("");

      const nodeMarkup = nodes.map((node) => {
        const pos = positions[node.paper_id];
        if (!pos) return "";
        const lines = wrapTitle(node.method_name || node.title || node.paper_id);
        return `
          <g class="graph-node" data-node-id="${escapeHtml(node.paper_id)}" transform="translate(${pos.x}, ${pos.y})" style="cursor:pointer;">
            <circle r="28" fill="${circleColor(node)}" fill-opacity="0.92" stroke="white" stroke-width="3"></circle>
            <circle r="34" fill="none" stroke="${circleColor(node)}" stroke-opacity="0.18" stroke-width="10"></circle>
            ${lines.map((line, idx) => `
              <text x="0" y="${44 + idx * 13}" text-anchor="middle" font-size="11" fill="#1e1c18">${escapeHtml(line)}</text>
            `).join("")}
          </g>
        `;
      }).join("");

      svg.innerHTML = `
        <g transform="translate(${panX}, ${panY}) scale(${zoom})">
          ${edgeMarkup}
          ${nodeMarkup}
        </g>
      `;

      svg.querySelectorAll(".graph-node").forEach((el) => {
        el.addEventListener("click", (event) => {
          event.stopPropagation();
          const nodeId = el.getAttribute("data-node-id");
          const node = state.graph.nodes.find((item) => item.paper_id === nodeId);
          if (!node) return;
          $("graph-node-detail").innerHTML = `
            <div class="pill">${escapeHtml(node.method_name || "Paper")}</div>
            <div style="margin-top:10px;font-weight:700;line-height:1.5;">${escapeHtml(node.title || "")}</div>
            <div class="mini" style="margin-top:8px;">paper_id: ${escapeHtml(node.paper_id || "")}</div>
            <div class="mini" style="margin-top:8px;">${escapeHtml(node.problem || node.tldr || "暂无摘要")}</div>
            ${node.note_path ? `<div class="button-row" style="margin-top:10px;"><button class="btn-ghost" onclick="openNote(${JSON.stringify(node.note_path)})">打开笔记</button></div>` : ""}
          `;
        });
        el.addEventListener("mousedown", (event) => {
          event.stopPropagation();
          const nodeId = el.getAttribute("data-node-id");
          state.drag = {
            nodeId,
            startX: event.clientX,
            startY: event.clientY,
            original: { ...state.graph.positions[nodeId] },
          };
        });
      });

      $("graph-stats").textContent = summarizeGraphStats(snapshot.stats || {}, nodes, edges);
    }

    async function refreshStats() {
      const stats = await apiGet("/api/stats");
      $("metric-nodes").textContent = stats.paper_nodes ?? 0;
      $("metric-edges").textContent = stats.paper_edges ?? 0;
      $("metric-episodes").textContent = stats.episodes ?? 0;
      $("metric-vectors").textContent = stats.vectors ?? 0;
    }

    async function loadNotes(query = "") {
      const suffix = query ? "?q=" + encodeURIComponent(query) : "";
      const data = await apiGet("/api/notes" + suffix);
      renderNotes(data);
    }

    async function openNote(path) {
      const data = await apiGet("/api/note?path=" + encodeURIComponent(path));
      $("note-viewer").textContent = data.content || "";
    }
    window.openNote = openNote;

    function fillPaperSource(url, title) {
      $("source").value = url || "";
      if (!$("title").value) $("title").value = title || "";
    }
    window.fillPaperSource = fillPaperSource;

    async function queryGraph() {
      const q = $("graph-query").value.trim();
      const data = await apiGet("/api/graph" + (q ? "?q=" + encodeURIComponent(q) : ""));
      renderGraph(data);
    }

    async function runSearch() {
      const query = $("search-query").value.trim();
      if (!query) {
        setStatus("search-status", "请先输入搜索关键词。", "bad");
        return;
      }
      setStatus("search-status", "正在搜索论文...");
      $("btn-search").disabled = true;
      try {
        const data = await apiPost("/api/search", { query });
        renderSearchResults(data);
        setStatus("search-status", "搜索完成，共 " + data.length + " 条结果。", "good");
      } catch (err) {
        setStatus("search-status", String(err.message || err), "bad");
      } finally {
        $("btn-search").disabled = false;
      }
    }

    async function runAnalyze() {
      const source = $("source").value.trim();
      const title = $("title").value.trim();
      const focus = $("focus").value.trim();
      const mode = $("mode").value;
      if (!source) {
        setStatus("action-status", "请先输入论文来源。", "bad");
        return;
      }
      setStatus("action-status", "正在分析，这一步可能需要几十秒到几分钟...");
      $("btn-analyze").disabled = true;
      try {
        const data = await apiPost("/api/analyze", { source, title, focus, mode });
        renderAnalysis(data);
        setStatus("action-status", "分析完成，笔记和记忆已更新。", "good");
        await refreshStats();
        await loadNotes($("notes-query").value.trim());
        if (!$("graph-query").value.trim()) {
          $("graph-query").value = focus || title || source;
        }
        await queryGraph();
      } catch (err) {
        setStatus("action-status", String(err.message || err), "bad");
      } finally {
        $("btn-analyze").disabled = false;
      }
    }

    function wireGraphInteractions() {
      const svg = $("graph-stage");

      svg.addEventListener("mousedown", (event) => {
        if (event.target.closest(".graph-node")) return;
        state.graphDrag = {
          startX: event.clientX,
          startY: event.clientY,
          panX: state.graph.panX,
          panY: state.graph.panY,
        };
      });

      window.addEventListener("mousemove", (event) => {
        if (state.drag) {
          const dx = (event.clientX - state.drag.startX) / state.graph.zoom;
          const dy = (event.clientY - state.drag.startY) / state.graph.zoom;
          state.graph.positions[state.drag.nodeId] = {
            x: state.drag.original.x + dx,
            y: state.drag.original.y + dy,
          };
          renderGraph({ nodes: state.graph.nodes, edges: state.graph.edges, stats: {
            paper_nodes: $("metric-nodes").textContent,
            paper_edges: $("metric-edges").textContent,
            episodes: $("metric-episodes").textContent,
            vectors: $("metric-vectors").textContent,
          }, query: state.graph.lastQuery || "" });
          return;
        }
        if (state.graphDrag) {
          state.graph.panX = state.graphDrag.panX + (event.clientX - state.graphDrag.startX);
          state.graph.panY = state.graphDrag.panY + (event.clientY - state.graphDrag.startY);
          renderGraph({ nodes: state.graph.nodes, edges: state.graph.edges, stats: {
            paper_nodes: $("metric-nodes").textContent,
            paper_edges: $("metric-edges").textContent,
            episodes: $("metric-episodes").textContent,
            vectors: $("metric-vectors").textContent,
          }, query: state.graph.lastQuery || "" });
        }
      });

      window.addEventListener("mouseup", () => {
        state.drag = null;
        state.graphDrag = null;
      });

      svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        const delta = event.deltaY < 0 ? 1.08 : 0.92;
        state.graph.zoom = Math.max(0.45, Math.min(2.8, state.graph.zoom * delta));
        renderGraph({ nodes: state.graph.nodes, edges: state.graph.edges, stats: {
          paper_nodes: $("metric-nodes").textContent,
          paper_edges: $("metric-edges").textContent,
          episodes: $("metric-episodes").textContent,
          vectors: $("metric-vectors").textContent,
        }, query: state.graph.lastQuery || "" });
      }, { passive: false });
    }

    async function boot() {
      wireGraphInteractions();
      $("btn-search").addEventListener("click", runSearch);
      $("btn-analyze").addEventListener("click", runAnalyze);
      $("btn-load-notes").addEventListener("click", () => loadNotes($("notes-query").value.trim()).catch((err) => setStatus("action-status", err.message, "bad")));
      $("btn-filter-notes").addEventListener("click", () => loadNotes($("notes-query").value.trim()).catch((err) => setStatus("action-status", err.message, "bad")));
      $("btn-query-graph").addEventListener("click", () => queryGraph().catch((err) => setStatus("action-status", err.message, "bad")));
      $("btn-load-graph").addEventListener("click", () => queryGraph().catch((err) => setStatus("action-status", err.message, "bad")));
      $("btn-zoom-in").addEventListener("click", () => {
        state.graph.zoom = Math.min(2.8, state.graph.zoom * 1.12);
        renderGraph({ nodes: state.graph.nodes, edges: state.graph.edges, stats: {
          paper_nodes: $("metric-nodes").textContent,
          paper_edges: $("metric-edges").textContent,
          episodes: $("metric-episodes").textContent,
          vectors: $("metric-vectors").textContent,
        }, query: state.graph.lastQuery || "" });
      });
      $("btn-zoom-out").addEventListener("click", () => {
        state.graph.zoom = Math.max(0.45, state.graph.zoom * 0.9);
        renderGraph({ nodes: state.graph.nodes, edges: state.graph.edges, stats: {
          paper_nodes: $("metric-nodes").textContent,
          paper_edges: $("metric-edges").textContent,
          episodes: $("metric-episodes").textContent,
          vectors: $("metric-vectors").textContent,
        }, query: state.graph.lastQuery || "" });
      });
      $("btn-reset-graph").addEventListener("click", () => {
        state.graph.zoom = 1;
        state.graph.panX = 0;
        state.graph.panY = 0;
        state.graph.positions = computeGraphLayout(state.graph.nodes, 920, 620);
        renderGraph({ nodes: state.graph.nodes, edges: state.graph.edges, stats: {
          paper_nodes: $("metric-nodes").textContent,
          paper_edges: $("metric-edges").textContent,
          episodes: $("metric-episodes").textContent,
          vectors: $("metric-vectors").textContent,
        }, query: state.graph.lastQuery || "" });
      });

      await refreshStats();
      await loadNotes("");
      await queryGraph();
    }

    boot().catch((err) => {
      setStatus("action-status", String(err.message || err), "bad");
    });
  </script>
</body>
</html>
""".strip()
