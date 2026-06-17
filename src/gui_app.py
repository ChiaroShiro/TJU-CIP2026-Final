"""本地 GUI 服务。

设计：
- 前端是 src/web 下的独立静态文件（index.html / styles.css / app.js / vendor/d3）。
- 后端只做一层薄薄的 JSON / SSE API + 静态资源分发，业务逻辑全部委托给 ResearchOrchestrator。
- 同步 JSON 接口：search / survey / figure / note 等快速、确定性请求。
- SSE 流式接口（text/event-stream）：discover / evaluate / analyze / research / chat，
  逐条推送统一进度事件（phase / log / step / token / data / result / error / done），
  让前端按模式渲染不同形态的实时反馈。
"""

from __future__ import annotations

import base64
import json
import queue
import re
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
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


class _ClientGone(BaseException):
    """SSE 客户端断开时，用于中止后台 worker。

    继承 BaseException 而非 Exception：这样业务代码里随处可见的 `except Exception`
    （如 LLMClient 流式回调、BaseAgent loop）不会把它吞掉，客户端断开时能一路
    冒泡中止正在进行的 LLM 流，避免白白消耗 token；worker 仍以 `except _ClientGone`
    精确捕获，相关 `finally`（如释放研究锁）正常执行。
    """


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


def _extract_json_string_value(buf: str, key: str) -> Optional[str]:
    """从一段（可能尚不完整的）JSON 文本里增量提取某个字符串字段的当前值。

    用于自主研究 finish 工具的参数流式解析：finish 的参数形如
    {"output": "....(报告)...."}，工具参数是按 JSON 字符串逐块到达的，
    本函数把已到达部分里 "output" 的值解出来（含转义还原），实现报告打字机。
    未出现该字段或转义不完整时返回已解析部分 / None。
    """
    match = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', buf)
    if not match:
        return None
    i = match.end()
    out = []
    n = len(buf)
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}
    while i < n:
        ch = buf[i]
        if ch == "\\":
            if i + 1 >= n:
                break  # 转义不完整，等后续 chunk
            nxt = buf[i + 1]
            if nxt == "u":
                if i + 6 > n:
                    break
                try:
                    out.append(chr(int(buf[i + 2:i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append(escapes.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            break  # 字符串结束
        out.append(ch)
        i += 1
    return "".join(out)


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


def _paper_brief(paper) -> Dict[str, Any]:
    """把 PaperItem 序列化为前端友好的 dict。"""
    return {
        "paper_id": getattr(paper, "paper_id", ""),
        "title": getattr(paper, "title", ""),
        "authors": getattr(paper, "authors", []) or [],
        "abstract": getattr(paper, "abstract", "") or "",
        "url": getattr(paper, "url", ""),
        "published": getattr(paper, "published", ""),
        "year": (getattr(paper, "published", "") or getattr(paper, "updated", "") or "")[:4],
        "code_url": getattr(paper, "code_url", ""),
        "has_code": bool(getattr(paper, "has_code", False)),
        "code_confidence": round(float(getattr(paper, "code_confidence", 0.0) or 0.0), 3),
    }


def build_gui_app(orchestrator: "ResearchOrchestrator") -> ThreadingHTTPServer:
    workspace_dir = orchestrator.settings.workspace_dir
    heavy_lock = threading.Lock()        # 串行化研究等重任务，保护记忆库/限流
    state: Dict[str, Any] = {"chat": None}

    def _has_llm() -> bool:
        return bool((orchestrator.settings.llm_api_key or "").strip())

    def _get_chat_agent():
        if state["chat"] is None:
            from .agents.conversational_agent import ConversationalAgent
            from .core.llm import LLMClient
            state["chat"] = ConversationalAgent(LLMClient(orchestrator.settings), orchestrator.memory)
        return state["chat"]

    class GuiHandler(BaseHTTPRequestHandler):
        # -------------------- 基础发送工具 -------------------- #
        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

        def log_message(self, format: str, *args) -> None:  # 静音访问日志
            return

        # -------------------- SSE 基础设施 -------------------- #
        def _sse_write(self, event: Dict[str, Any]) -> None:
            payload = json.dumps(event, ensure_ascii=False, default=str)
            self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
            self.wfile.flush()

        def _run_sse(self, job) -> None:
            """以 SSE 跑一个 job(emit)。job 在后台线程运行，本线程负责把事件写给客户端。

            生产者/消费者解耦：job 只往队列丢事件；客户端断开时本线程置 cancel，
            job 下次 emit 时收到 _ClientGone 提前退出。本方法保证不向外抛异常
            （SSE 头已发出，无法再回退到 JSON 错误响应）。
            """
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
            except OSError:
                return

            q: "queue.Queue" = queue.Queue()
            cancel = threading.Event()
            DONE = object()

            def emit(event: Dict[str, Any]) -> None:
                if cancel.is_set():
                    raise _ClientGone()
                q.put(event)

            def worker() -> None:
                try:
                    job(emit)
                except _ClientGone:
                    pass
                except Exception as exc:  # noqa: BLE001 — 把任何业务异常变成 error 事件
                    try:
                        q.put({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
                    except Exception:
                        pass
                finally:
                    q.put(DONE)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            try:
                while True:
                    try:
                        item = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")  # 心跳，保持连接
                        self.wfile.flush()
                        continue
                    if item is DONE:
                        self._sse_write({"type": "done"})
                        break
                    self._sse_write(item)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                cancel.set()

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
                if path == "/api/capabilities":
                    self._send_json({"ok": True, "data": {"llm": _has_llm()}})
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
            except Exception as exc:
                self._send_json({"ok": False, "error": f"invalid body: {exc}"}, status=400)
                return

            # --- SSE 流式接口 --- #
            if path == "/api/discover":
                self._run_sse(lambda emit: self._job_discover(emit, payload))
                return
            if path == "/api/evaluate":
                self._run_sse(lambda emit: self._job_evaluate(emit, payload))
                return
            if path == "/api/analyze":
                self._run_sse(lambda emit: self._job_analyze(emit, payload))
                return
            if path == "/api/research":
                self._run_sse(lambda emit: self._job_research(emit, payload))
                return
            if path == "/api/chat":
                self._run_sse(lambda emit: self._job_chat(emit, payload))
                return

            # --- 同步 JSON 接口 --- #
            try:
                if path == "/api/search":
                    self._handle_search(payload)
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
                if path == "/api/chat-clear":
                    if state["chat"] is not None:
                        state["chat"].clear()
                    self._send_json({"ok": True, "data": {"cleared": True}})
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

        # -------------------- 同步业务处理 -------------------- #
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

        # -------------------- SSE 业务 job -------------------- #
        def _job_discover(self, emit, payload: Dict[str, Any]) -> None:
            topic = str(payload.get("query") or payload.get("topic") or "").strip()
            if not topic:
                emit({"type": "error", "text": "请输入研究主题"})
                return
            try:
                n = int(payload.get("max_papers", 10) or 10)
            except (TypeError, ValueError):
                n = 10
            n = max(3, min(30, n))

            emit({"type": "phase", "label": "开始检索论文", "pct": 4})
            papers = orchestrator.discover_papers(topic, max_results=n, on_event=emit)
            data = [_paper_brief(p) for p in papers]
            with_code = sum(1 for p in data if p["has_code"])
            emit({"type": "phase", "label": "完成", "pct": 100})
            emit({
                "type": "result",
                "payload": {"topic": topic, "count": len(data), "with_code": with_code, "papers": data},
            })

        def _job_evaluate(self, emit, payload: Dict[str, Any]) -> None:
            direction = str(payload.get("direction") or payload.get("topic") or "").strip()
            if not direction:
                emit({"type": "error", "text": "请输入要评估的研究方向"})
                return
            if not _has_llm():
                emit({"type": "error", "text": "方向评估需要 LLM API key，请先在 .env 配置"})
                return

            emit({"type": "phase", "label": "开始评估研究方向", "pct": 4})
            result = orchestrator.evaluate_direction(direction, on_event=emit)
            papers = [{"title": getattr(p, "title", ""), "url": getattr(p, "url", "")}
                      for p in (result.get("papers") or [])[:8]]
            emit({"type": "phase", "label": "完成", "pct": 100})
            emit({
                "type": "result",
                "payload": {
                    "feasibility": result.get("feasibility", 0.0),
                    "novelty": result.get("novelty", 0.0),
                    "impact": result.get("impact", 0.0),
                    "analysis": result.get("analysis", ""),
                    "recommendations": result.get("recommendations", []),
                    "related_topics": result.get("related_topics", []),
                    "benchmarks": result.get("benchmarks", []),
                    "search_queries": result.get("search_queries", []),
                    "papers": papers,
                },
            })

        def _job_analyze(self, emit, payload: Dict[str, Any]) -> None:
            source = str(payload.get("source", "")).strip()
            focus = str(payload.get("focus", "")).strip() or None
            title = str(payload.get("title", "")).strip() or None
            mode = str(payload.get("mode", "read-paper")).strip()

            if not source:
                emit({"type": "error", "text": "请输入论文来源（arXiv ID / 链接 / 本地 PDF 路径）"})
                return
            if not _has_llm():
                emit({"type": "error", "text": "论文精读需要 LLM API key，请先在 .env 配置"})
                return

            possible_file = Path(source).expanduser()
            if possible_file.exists() and possible_file.is_file():
                emit({"type": "phase", "label": "读取本地 PDF", "pct": 3})
                result = orchestrator.analyzer.analyze_local_pdf(
                    possible_file.resolve(), title=title, focus=focus, on_event=emit
                )
                display_title = title or orchestrator.analyzer.fetcher.infer_title_from_pdf(possible_file)
                emit({"type": "result",
                      "payload": {"title": display_title, "result": result, "local": True}})
                return

            paper_id = source.rstrip("/").split("/")[-1].replace(".pdf", "")
            emit({"type": "phase", "label": f"定位 arXiv 论文 {paper_id}", "pct": 3})
            papers = orchestrator.arxiv.search(f"id:{paper_id}", max_results=1)
            if not papers:
                emit({"type": "error", "text": f"未找到 arXiv 论文：{paper_id}"})
                return

            paper = papers[0]
            if mode == "read-paper":
                result = orchestrator.analyzer.analyze_multimodal(paper, focus, on_event=emit)
            else:
                result = orchestrator.analyzer.analyze(paper, focus, on_event=emit)
            emit({"type": "result",
                  "payload": {"title": paper.title, "result": result, "local": False}})

        def _job_research(self, emit, payload: Dict[str, Any]) -> None:
            topic = str(payload.get("topic", "")).strip()
            mode = str(payload.get("mode", "auto")).strip().lower()
            if not topic:
                emit({"type": "error", "text": "请输入研究主题"})
                return
            if not _has_llm():
                emit({"type": "error", "text": "深度研究需要 LLM API key，请先在 .env 配置"})
                return
            if not heavy_lock.acquire(blocking=False):
                emit({"type": "error", "text": "已有一个研究任务在进行中，请等它结束后再试"})
                return
            try:
                if mode == "legacy":
                    self._run_legacy_research(emit, topic)
                else:
                    self._run_auto_research(emit, topic, payload)
            finally:
                heavy_lock.release()

        def _run_legacy_research(self, emit, topic: str) -> None:
            emit({"type": "phase", "label": "启动编排式研究流水线", "pct": 2})
            result = orchestrator.run_deep_research(topic, on_event=emit)
            critic_score = result.critic_reviews[-1].score if result.critic_reviews else None
            emit({"type": "phase", "label": "完成", "pct": 100})
            emit({
                "type": "result",
                "payload": {
                    "mode": "legacy",
                    "finished": True,
                    "report_md": result.final_report_markdown,
                    "report_path": result.report_file,
                    "stats": {
                        "paper_count": len(result.papers),
                        "task_count": len(result.task_results),
                        "critic_score": critic_score,
                        "revision_count": result.revision_count,
                    },
                },
            })

        def _run_auto_research(self, emit, topic: str, payload: Dict[str, Any]) -> None:
            from .agents.manager import ResearchManager

            try:
                max_steps = int(payload.get("max_steps", 30) or 30)
            except (TypeError, ValueError):
                max_steps = 30
            try:
                max_tokens = int(payload.get("max_tokens", 200000) or 200000)
            except (TypeError, ValueError):
                max_tokens = 200000
            max_steps = max(6, min(60, max_steps))

            manager = ResearchManager(
                orchestrator.settings, max_steps=max_steps, max_total_tokens=max_tokens,
                memory=orchestrator.memory, reflection=orchestrator.reflection_engine,
            )
            report_buf = {"raw": "", "emitted": 0}

            def inner(ev: Dict[str, Any]) -> None:
                etype = ev.get("type")
                if etype == "step":
                    emit(ev)
                elif etype == "budget_warning":
                    emit({"type": "log", "level": "warn",
                          "text": f"预算预警：已用 {ev.get('step_pct', 0):.0%} 步数 / "
                                  f"{ev.get('budget_pct', 0):.0%} tokens，开始收尾撰写报告"})
                elif etype == "llm_delta":
                    kind = ev.get("kind")
                    if kind == "tool_name" and ev.get("name") == "finish":
                        emit({"type": "phase", "label": "撰写最终报告", "pct": 92})
                    elif kind == "tool_args" and ev.get("name") == "finish":
                        report_buf["raw"] += ev.get("text", "")
                        out = _extract_json_string_value(report_buf["raw"], "output")
                        if out is not None and len(out) > report_buf["emitted"]:
                            emit({"type": "token", "pane": "report", "text": out[report_buf["emitted"]:]})
                            report_buf["emitted"] = len(out)

            emit({"type": "phase", "label": "自主研究 Agent 启动", "pct": 2})
            result = manager.run(topic, on_event=inner)

            output = result.final_output
            if isinstance(output, dict):
                report = output.get("output", "") or json.dumps(output, ensure_ascii=False)
            else:
                report = str(output) if output is not None else ""

            report_path = ""
            if result.finished and report:
                report_path = self._save_research_report(topic, report)
                if report_buf["emitted"] == 0:
                    emit({"type": "token", "pane": "report", "text": report})

            emit({"type": "phase", "label": "完成", "pct": 100})
            emit({
                "type": "result",
                "payload": {
                    "mode": "auto",
                    "finished": result.finished,
                    "finish_reason": result.finish_reason,
                    "report_md": report,
                    "report_path": report_path,
                    "stats": {
                        "total_steps": len(result.steps),
                        "total_tokens": result.total_tokens,
                        "elapsed_ms": result.total_elapsed_ms,
                    },
                },
            })

        @staticmethod
        def _save_research_report(topic: str, report: str) -> str:
            reports_dir = workspace_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:40].strip()
            path = reports_dir / f"{timestamp}_{safe_topic or 'research'}.md"
            path.write_text(report, encoding="utf-8")
            return str(path)

        # -------------------- 对话 dispatch -------------------- #
        def _job_chat(self, emit, payload: Dict[str, Any]) -> None:
            message = str(payload.get("message", "")).strip()
            if not message:
                emit({"type": "error", "text": "请输入对话内容"})
                return
            if not _has_llm():
                emit({"type": "error", "text": "对话模式需要 LLM API key，请先在 .env 配置"})
                return

            agent = _get_chat_agent()
            emit({"type": "phase", "label": "理解意图", "pct": 8})
            agent.add_user_message(message)
            action = agent.decide()
            emit({
                "type": "intent",
                "action": action.action,
                "queries": action.queries,
                "topic": action.topic,
                "paper_id": action.paper_id,
                "raw_intent": action.raw_intent,
            })

            reply, data = self._chat_dispatch(emit, agent, action)
            agent.add_assistant_message(reply)
            emit({"type": "result", "payload": {"reply": reply, "action": action.action, "data": data}})

        def _chat_dispatch(self, emit, agent, action) -> Tuple[str, Any]:
            act = action.action

            if act in ("ask_user", "chitchat"):
                reply = action.reply or "我在听，请继续说说你的研究想法。"
                self._emit_text_typewriter(emit, reply)
                return reply, None

            if act == "search":
                emit({"type": "phase", "label": "检索论文", "pct": 30})
                queries = action.queries or ([action.topic] if action.topic else [])
                papers = orchestrator.search_papers_multi(queries or ["research"], per_query=4)
                brief = {
                    "count": len(papers),
                    "papers": [{"title": p.title, "url": p.url,
                                "abstract": (p.abstract or "")[:160]} for p in papers[:8]],
                }
                agent.add_tool_result(f"search done. {len(papers)} papers.")
                return self._chat_stream_summary(emit, agent, action, brief), brief

            if act == "evaluate":
                topic = action.topic or (action.queries[0] if action.queries else "")
                result = orchestrator.evaluate_direction(topic, queries=action.queries or None, on_event=emit)
                brief = {
                    "feasibility": result.get("feasibility"),
                    "novelty": result.get("novelty"),
                    "impact": result.get("impact"),
                    "analysis": (result.get("analysis", "") or "")[:1200],
                    "recommendations": result.get("recommendations", []),
                }
                agent.add_tool_result(f"evaluate done. {brief}")
                return self._chat_stream_summary(emit, agent, action, brief), brief

            if act == "analyze":
                paper_id = (action.paper_id or "").strip().split("/")[-1].replace(".pdf", "")
                if not paper_id:
                    reply = "想分析论文的话，给我一个 arXiv ID 或链接就行。"
                    self._emit_text_typewriter(emit, reply)
                    return reply, None
                papers = orchestrator.arxiv.search(f"id:{paper_id}", max_results=1)
                if not papers:
                    reply = f"没找到 arXiv ID 为 {paper_id} 的论文，确认一下？"
                    self._emit_text_typewriter(emit, reply)
                    return reply, None
                result = orchestrator.analyze_paper(papers[0], action.focus or None)
                brief = {
                    "title": papers[0].title,
                    "tldr": result.get("tldr", ""),
                    "problem": result.get("problem", ""),
                    "note_path": result.get("_note_path", ""),
                }
                agent.add_tool_result(f"analyze done for {papers[0].title}")
                return self._chat_stream_summary(emit, agent, action, brief), brief

            if act == "memory_query":
                stats = orchestrator.memory_stats()
                ctx = orchestrator.memory.format_context_for_prompt(action.topic or "research")
                brief = {"stats": stats, "context": ctx[:1800]}
                agent.add_tool_result(f"memory query done. {brief}")
                return self._chat_stream_summary(emit, agent, action, brief), brief

            if act == "paper_note_query":
                notes = orchestrator.memory.find_paper_notes(action.topic or "", top_k=5)
                brief = {"query": action.topic, "note_count": len(notes),
                         "notes": [{"title": n["title"], "preview": n["preview"][:400]} for n in notes]}
                agent.add_tool_result(f"paper note query done. {len(notes)} notes")
                return self._chat_stream_summary(emit, agent, action, brief), brief

            if act == "research":
                topic = action.topic or (action.queries[0] if action.queries else "")
                if not heavy_lock.acquire(blocking=False):
                    reply = "现在已经有一个研究任务在跑了，等它结束我们再深入研究这个，好吗？"
                    self._emit_text_typewriter(emit, reply)
                    return reply, None
                try:
                    emit({"type": "phase", "label": "在对话中发起深度研究", "pct": 5})

                    def phase_only(ev):  # 对话里不把整篇报告灌进气泡，只显示进度
                        if ev.get("type") in ("phase", "log"):
                            emit(ev)

                    result = orchestrator.run_deep_research(topic, on_event=phase_only)
                finally:
                    heavy_lock.release()
                brief = {
                    "topic": result.topic,
                    "report_path": result.report_file,
                    "report_preview": result.final_report_markdown[:1200],
                    "paper_count": len(result.papers),
                }
                agent.add_tool_result(f"research done. report at {result.report_file}")
                return self._chat_stream_summary(emit, agent, action, brief), brief

            reply = action.reply or "我还不太确定怎么帮你，可以说得更具体一点吗？"
            self._emit_text_typewriter(emit, reply)
            return reply, None

        def _chat_stream_summary(self, emit, agent, action, brief) -> str:
            """流式生成对话回复（打字机），并累计返回完整文本。"""
            parts = []
            try:
                for delta in agent.summarize_result_stream(action, brief):
                    parts.append(delta)
                    emit({"type": "token", "pane": "reply", "text": delta})
            except Exception:
                if not parts:
                    fallback = agent.summarize_result(action, brief)
                    self._emit_text_typewriter(emit, fallback)
                    return fallback
            return "".join(parts).strip()

        @staticmethod
        def _emit_text_typewriter(emit, text: str, chunk: int = 18) -> None:
            """把现成文本按小块推成 token 流，制造打字机效果（无需 LLM）。"""
            for i in range(0, len(text), chunk):
                emit({"type": "token", "pane": "reply", "text": text[i:i + chunk]})

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
