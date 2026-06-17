"""本地 GUI 服务。

设计：
- 前端是 src/web 下的独立静态文件（index.html / styles.css / app.js / vendor/d3）。
- 后端只做一层薄薄的 JSON API + 静态资源分发，业务逻辑全部委托给 ResearchOrchestrator。
- 新增能力：
  * /api/asset   安全分发 workspace 内的本地图片/SVG（修复笔记本地图无法显示）。
  * /api/survey  调用 SurveyBuilder，把综述报告 + 算法演进图/海报 SVG 接入前端。
  * /api/figure  统一的「图产物」契约，支持 svg / latex / image 三种渲染模式，
                 为后续接入 LaTeX 描述图或图像生成模型预留扩展点。
"""

from __future__ import annotations

import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:  # 仅类型提示，避免在导入本模块时拉起重型依赖
    from .orchestrator import ResearchOrchestrator


WEB_DIR = Path(__file__).parent / "web"

# 只允许分发这些被显式登记的前端文件，杜绝任意路径读取
_STATIC_ROUTES: Dict[str, Tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/vendor/d3.v7.min.js": ("vendor/d3.v7.min.js", "application/javascript; charset=utf-8"),
}

# /api/asset 仅用于显示/下载图片与 SVG 产物，因此白名单收窄到图片类，缩小可读文件面
_ASSET_CONTENT_TYPES: Dict[str, str] = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _resolve_within(base: Path, target: str) -> Path:
    """把 target 解析为绝对路径，并确保它位于 base 之内（防止路径穿越）。"""
    base_resolved = base.expanduser().resolve()
    candidate = Path(target).expanduser()
    if not candidate.is_absolute():
        candidate = base_resolved / candidate
    candidate = candidate.resolve()
    if base_resolved != candidate and base_resolved not in candidate.parents:
        raise PermissionError("path escapes workspace")
    return candidate


def build_figure_stub(mode: str, title: str, spec: str = "") -> Dict[str, Any]:
    """构造统一的「图产物（Figure）」对象。

    前端只依据 kind ∈ {svg, latex, image} 分发渲染，永远不关心图是怎么生成的。
    这里给三种模式各提供一个可正确渲染的示例 / 占位，证明契约端到端可用；
    后续把对应分支替换成真实的 LaTeX 编译或图像模型调用即可，前端零改动。
    """
    mode = (mode or "svg").strip().lower()
    title = title or "示例图"

    if mode == "latex":
        source = spec.strip() or (
            "\\begin{tikzpicture}\n"
            "  \\node[draw,rounded corners] (a) {Encoder};\n"
            "  \\node[draw,rounded corners,right=of a] (b) {Policy};\n"
            "  \\draw[->] (a) -- (b);\n"
            "\\end{tikzpicture}"
        )
        return {
            "id": "figure-latex",
            "title": title,
            "kind": "latex",
            "latex": {"engine": "tikz", "source": source, "rendered_svg": None},
            "meta": {
                "source": "stub",
                "note": "LaTeX 渲染后端尚未配置：已返回源码，接入后端编译为 SVG 后即可矢量显示。",
            },
        }

    if mode == "image":
        # 不透明浅色面板，保证在深/浅两种卡片背景下文字都清晰可读
        placeholder = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='420'>"
            "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0' stop-color='#2dd4bf'/><stop offset='1' stop-color='#818cf8'/>"
            "</linearGradient></defs>"
            "<rect width='720' height='420' rx='18' fill='#eef2f7'/>"
            "<rect width='720' height='12' rx='6' fill='url(#g)'/>"
            "<rect x='1.5' y='1.5' width='717' height='417' rx='18' fill='none' stroke='#cbd5e1'/>"
            "<text x='360' y='200' text-anchor='middle' font-size='28' font-weight='800' "
            "fill='#1e293b' font-family='sans-serif'>图像生成占位</text>"
            "<text x='360' y='242' text-anchor='middle' font-size='15' "
            "fill='#475569' font-family='sans-serif'>image2 / 扩散模型接入后将替换为真实位图</text>"
            "</svg>"
        )
        data_uri = "data:image/svg+xml;base64," + base64.b64encode(placeholder.encode("utf-8")).decode("ascii")
        return {
            "id": "figure-image",
            "title": title,
            "kind": "image",
            "image": {"url": None, "data_uri": data_uri, "mime": "image/svg+xml", "alt": title},
            "meta": {
                "source": "stub",
                "note": "图像生成模型尚未配置：当前返回占位图，接入后返回真实 PNG/URL。",
            },
        }

    # 默认 svg：返回一张真实可缩放的矢量示例图
    sample_svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='360' viewBox='0 0 720 360'>"
        "<rect width='720' height='360' fill='#ffffff'/>"
        "<text x='40' y='52' font-size='24' font-weight='800' fill='#0f172a' "
        "font-family='sans-serif'>SVG 矢量示例</text>"
        "<line x1='60' y1='300' x2='680' y2='300' stroke='#94a3b8' stroke-width='2'/>"
        "<line x1='60' y1='90' x2='60' y2='300' stroke='#94a3b8' stroke-width='2'/>"
        + "".join(
            f"<rect x='{90 + i * 95}' y='{300 - h}' width='60' height='{h}' rx='6' "
            f"fill='{'#0f766e' if i % 2 == 0 else '#818cf8'}'/>"
            for i, h in enumerate([120, 168, 96, 204, 150, 186])
        )
        + "<text x='40' y='344' font-size='13' fill='#64748b' font-family='sans-serif'>"
        "矢量图：可无损缩放、可下载 .svg</text>"
        "</svg>"
    )
    return {
        "id": "figure-svg",
        "title": title,
        "kind": "svg",
        "svg": sample_svg,
        "meta": {"source": "stub", "note": "内置 SVG 示例（规则生成，无需 LLM）。"},
    }


def build_gui_app(orchestrator: "ResearchOrchestrator") -> ThreadingHTTPServer:
    workspace_dir = orchestrator.settings.workspace_dir

    class GuiHandler(BaseHTTPRequestHandler):
        # -------------------- 基础发送工具 -------------------- #
        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

        def log_message(self, format: str, *args) -> None:  # 静音访问日志
            return

        # -------------------- 资源分发 -------------------- #
        def _serve_static(self, path: str) -> bool:
            entry = _STATIC_ROUTES.get(path)
            if not entry:
                return False
            filename, content_type = entry
            file_path = (WEB_DIR / filename).resolve()
            if not file_path.exists() or not file_path.is_file():
                self._send_json({"ok": False, "error": f"missing static file: {filename}"}, status=404)
                return True
            self._send_bytes(file_path.read_bytes(), content_type)
            return True

        def _serve_asset(self, raw_path: str) -> None:
            if not raw_path:
                raise ValueError("missing asset path")
            target = _resolve_within(workspace_dir, raw_path)
            if not target.exists() or not target.is_file():
                raise FileNotFoundError("asset not found")
            content_type = _ASSET_CONTENT_TYPES.get(target.suffix.lower())
            if content_type is None:
                raise PermissionError(f"asset type not allowed: {target.suffix}")
            self._send_bytes(target.read_bytes(), content_type)

        def _read_note_payload(self, note_path: str) -> Dict[str, Any]:
            if not note_path:
                raise ValueError("missing path")
            file_path = _resolve_within(workspace_dir, note_path)
            if not file_path.exists():
                raise FileNotFoundError("note not found")
            return {
                "path": str(file_path),
                "name": file_path.name,
                "dir": str(file_path.parent),
                "content": file_path.read_text(encoding="utf-8"),
            }

        # -------------------- GET -------------------- #
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            try:
                if self._serve_static(path):
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
                    self._send_json({"ok": True, "data": self._read_note_payload(note_path)})
                    return
                if path == "/api/asset":
                    self._serve_asset((query.get("path", [""])[0] or "").strip())
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
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=404)
            except PermissionError as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=403)
            except ValueError as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

        # -------------------- POST -------------------- #
        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            try:
                payload = self._read_json_body()

                if path == "/api/search":
                    self._handle_search(payload)
                    return
                if path == "/api/analyze":
                    self._handle_analyze(payload)
                    return
                if path == "/api/survey":
                    self._handle_survey(payload)
                    return
                if path == "/api/figure":
                    mode = str(payload.get("mode", "svg")).strip()
                    title = str(payload.get("title", "")).strip()
                    spec = str(payload.get("spec", payload.get("prompt", "")) or "")
                    self._send_json({"ok": True, "data": build_figure_stub(mode, title, spec)})
                    return
                if path == "/api/open-note":
                    note_path = str(payload.get("path", "")).strip()
                    self._send_json({"ok": True, "data": self._read_note_payload(note_path)})
                    return

                self._send_json({"ok": False, "error": f"unknown path: {path}"}, status=404)
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=404)
            except PermissionError as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=403)
            except ValueError as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

        # -------------------- 业务处理 -------------------- #
        def _handle_search(self, payload: Dict[str, Any]) -> None:
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

        def _handle_analyze(self, payload: Dict[str, Any]) -> None:
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
                    possible_file.resolve(), title=title, focus=focus
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

        def _handle_survey(self, payload: Dict[str, Any]) -> None:
            topic = str(payload.get("topic", "")).strip()
            if not topic:
                self._send_json({"ok": False, "error": "topic is required"}, status=400)
                return
            try:
                max_papers = int(payload.get("max_papers", 12) or 12)
            except (TypeError, ValueError):
                max_papers = 12
            max_papers = max(3, min(30, max_papers))

            from .services.survey_builder import SurveyBuilder

            builder = SurveyBuilder(workspace_dir)
            artifact = builder.build(topic, max_papers=max_papers)

            timeline_svg = Path(artifact.timeline_file).read_text(encoding="utf-8")
            poster_svg = Path(artifact.poster_file).read_text(encoding="utf-8")
            report_md = Path(artifact.report_file).read_text(encoding="utf-8")

            figures = [
                {
                    "id": "timeline",
                    "title": "算法发展演进图",
                    "kind": "svg",
                    "svg": timeline_svg,
                    "download_path": artifact.timeline_file,
                    "meta": {"source": "survey_timeline", "note": "按发布日期排序，绿色节点表示发现公开代码。"},
                },
                {
                    "id": "poster",
                    "title": "综述海报",
                    "kind": "svg",
                    "svg": poster_svg,
                    "download_path": artifact.poster_file,
                    "meta": {"source": "survey_poster", "note": "证据优先、公开代码优先的综述海报。"},
                },
            ]
            papers = [
                {
                    "title": p.title,
                    "year": (p.published or p.updated or "")[:4],
                    "url": p.url,
                    "code_url": p.code_url,
                    "has_code": p.has_code,
                }
                for p in artifact.papers
            ]
            self._send_json(
                {
                    "ok": True,
                    "data": {
                        "topic": topic,
                        "output_dir": artifact.output_dir,
                        "report_md": report_md,
                        "report_path": artifact.report_file,
                        "raw_data_file": artifact.raw_data_file,
                        "papers": papers,
                        "figures": figures,
                    },
                }
            )

    return ThreadingHTTPServer(("127.0.0.1", 0), GuiHandler)


def launch_gui(orchestrator: "ResearchOrchestrator", open_browser: bool = True) -> str:
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
