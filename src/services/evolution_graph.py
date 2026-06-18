"""算法演进图（Algorithm Evolution Graph）生成。

把一组论文渲染成一张**多泳道、带演进关系**的算法演进图，而不是单条时间线：

- 纵轴 = 发布时间（越靠下越新）。所有节点按 (年, 日期) 全局排序后逐行排列，
  因此"越往下越新"在结构上成立 —— 边只能从早指向晚，**不可能出现时间错误**。
- 横向泳道 = 技术分支。有 LLM key 时由 LLM 把论文归入 2-5 条技术线并命名；
  无 key 时按 arXiv 主类目确定性分组（降级仍是多泳道演进图）。
- 边 = 演进关系。有 LLM key 时由 LLM 在**给定论文集合内部**判定
  builds_on（继承/改进，实线）/ compares_with（对比，虚线），每条带一句理由，
  且边方向恒为早→晚（晚论文继承/对比早论文）；无 key 时退化为同分支时间相邻的浅色连接。

设计原则（与项目一致）：
- 边的"是否存在/类型"尽量基于可核查信息，LLM 只在给定集合内判断、附理由，不臆造集合外关系。
- 无 LLM 也能产出有意义的多泳道演进图，零幻觉、确定性。
- 纯字符串拼接 SVG，零额外依赖；前端 sanitizeSvg 只会移除 script/foreignObject/on*，
  本模块只用 defs/marker/linearGradient/style/title/rect/line/path/circle/text/tspan/g。
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import PaperItem
from ..core.utils import extract_json_object


# 泳道配色：品牌青 / 靛蓝 打头，其余为可区分的高级色
LANE_COLORS = ["#2dd4bf", "#818cf8", "#38bdf8", "#fb7185", "#f59e0b", "#34d399", "#a78bfa", "#f472b6"]
EDGE_BUILDS = "#38bdf8"     # builds_on 实线（与前端图谱 EDGE_COLORS 一致）
EDGE_COMPARES = "#fb7185"   # compares_with 虚线
EDGE_SIMILAR = "#94a3b8"    # similar_to 浅色（无 key 降级时同分支相邻）
CODE_DOT = "#22c55e"        # 有公开代码标记点

GRAPH_MAX_NODES = 22        # 图中最多展示的论文数（保证可读性）
MAX_BRANCHES = 5
MAX_OUT_DEGREE = 2          # 每个节点最多保留的"祖先"边数，避免连线爆炸


class EvolutionGraphBuilder:
    """根据论文集合生成算法演进图 SVG。"""

    def __init__(self, llm: Any = None):
        # llm: 可选的 LLMClient。None 或不可用时走确定性启发式。
        self.llm = llm

    # ------------------------------------------------------------------ #
    # 对外主入口
    # ------------------------------------------------------------------ #
    def build(self, topic: str, papers: List[PaperItem]) -> Dict[str, Any]:
        """返回 {svg, mode, branches, nodes, edges, note}。失败也始终返回可渲染 SVG。"""
        nodes = self._prepare_nodes(papers)
        if not nodes:
            return {
                "svg": self._empty_svg(topic),
                "mode": "empty",
                "branches": [],
                "nodes": [],
                "edges": [],
                "note": "未检索到可用论文，无法生成演进图。",
            }

        used_llm = False
        branches: Optional[List[Dict[str, Any]]] = None
        assign: Optional[Dict[int, int]] = None
        edges: Optional[List[Dict[str, Any]]] = None

        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                branches, assign, edges = self._llm_structure(topic, nodes)
                used_llm = bool(branches)
            except Exception:
                branches = assign = edges = None

        if not branches:
            branches, assign, edges = self._heuristic_structure(nodes)
            used_llm = False

        # 把分支信息回填到节点
        for node in nodes:
            node["branch"] = assign.get(node["id"], 0)
        branch_order = {b["id"]: i for i, b in enumerate(branches)}
        for node in nodes:
            node["lane"] = branch_order.get(node["branch"], 0)

        svg = self._render_svg(topic, nodes, branches, edges, used_llm)
        note = (
            "技术分支与演进关系由 LLM 在论文集合内部判定（实线=继承/改进，虚线=对比），"
            "时间从上到下递增，边方向恒为早→晚。"
            if used_llm
            else "无 LLM key，按 arXiv 主类目分泳道、按发布时间纵向排列（确定性、无幻觉）。"
        )
        return {
            "svg": svg,
            "mode": "llm" if used_llm else "heuristic",
            "branches": branches,
            "nodes": [
                {k: n[k] for k in ("id", "paper_id", "title", "year", "has_code", "lane")}
                for n in nodes
            ],
            "edges": edges,
            "note": note,
        }

    # ------------------------------------------------------------------ #
    # 节点准备
    # ------------------------------------------------------------------ #
    def _prepare_nodes(self, papers: List[PaperItem]) -> List[Dict[str, Any]]:
        items = []
        for p in papers or []:
            title = (getattr(p, "title", "") or "").strip()
            if not title:
                continue
            items.append(p)
        # 输入已按相关度排序：先取最相关的前 N 篇，再按时间排版，兼顾相关性与可读性
        items = items[:GRAPH_MAX_NODES]

        nodes: List[Dict[str, Any]] = []
        for p in items:
            published = getattr(p, "published", "") or ""
            updated = getattr(p, "updated", "") or ""
            nodes.append({
                "paper_id": getattr(p, "paper_id", "") or "",
                "title": (getattr(p, "title", "") or "").strip(),
                "abstract": (getattr(p, "abstract", "") or "").strip(),
                "year": self._year(published, updated),
                "_sortkey": (self._year(published, updated), published or updated or ""),
                "has_code": bool(getattr(p, "has_code", False)),
                "code_url": getattr(p, "code_url", "") or "",
                "url": getattr(p, "url", "") or "",
                "category": self._primary_category(getattr(p, "categories", []) or []),
            })

        nodes.sort(key=lambda n: (n["_sortkey"][0], n["_sortkey"][1], n["title"]))
        for row, node in enumerate(nodes):
            node["id"] = row    # id 即全局时间序行号，保证 id 越大越新
            node["row"] = row
        return nodes

    @staticmethod
    def _year(published: str, updated: str) -> int:
        for value in (published, updated):
            try:
                y = int(str(value)[:4])
                if 1900 <= y <= 2100:
                    return y
            except Exception:
                continue
        return 0

    @staticmethod
    def _primary_category(categories: List[str]) -> str:
        for c in categories:
            c = (c or "").strip()
            if c:
                return c
        return ""

    @staticmethod
    def _as_index(value: Any) -> Optional[int]:
        """把 LLM 返回的索引安全转成 int；拒绝 bool / 非整数浮点 / 非数字串。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        return None

    # ------------------------------------------------------------------ #
    # LLM 结构化：分支 + 边
    # ------------------------------------------------------------------ #
    def _llm_structure(
        self, topic: str, nodes: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]:
        listing = []
        for n in nodes:
            abstract = (n["abstract"] or "").replace("\n", " ")[:260]
            listing.append(
                f'#{n["id"]} ({n["year"] or "?"}{", code" if n["has_code"] else ""}): '
                f'{n["title"]}\n    {abstract}'
            )
        papers_block = "\n".join(listing)

        prompt = f"""You are organizing papers into an ALGORITHM EVOLUTION GRAPH for the topic: "{topic}".

Papers (each line starts with #INDEX, then year, then title, then abstract). They are ALREADY sorted by publication time ascending, so a LARGER index means a NEWER paper:
{papers_block}

Do TWO things, grounded ONLY in the titles/abstracts above (do not invent facts or papers):

1. BRANCHES: group these papers into 2-{MAX_BRANCHES} coherent technical branches (sub-directions / method families). Give each branch a short Chinese name (<= 12 chars). Every paper must be assigned to exactly one branch.

2. EDGES: identify evolution relationships strictly BETWEEN the papers above. For each meaningful relationship output an edge with:
   - "from": index of the EARLIER paper (smaller index)
   - "to": index of the LATER paper (larger index) that builds on / compares with the earlier one
   - "type": "builds_on" (the later paper extends/improves the earlier method) or "compares_with" (the later paper uses the earlier one mainly as a baseline / contrast)
   - "reason": one short Chinese phrase (<= 30 chars) grounded in the abstracts

Rules:
- "from" index MUST be strictly smaller than "to" index (earlier -> later). Never reverse.
- Only connect papers that are plausibly related from their abstracts. Be conservative: skip weak links. Aim for roughly {min(len(nodes) + 2, 18)} edges or fewer.
- Each later paper should build on at most 2 earlier papers.
- Use ONLY the indices shown above.

Return JSON only:
{{
  "branches": [{{"id": 0, "name": "分支名"}}],
  "assignments": [{{"paper": 0, "branch": 0}}],
  "edges": [{{"from": 0, "to": 3, "type": "builds_on", "reason": "..."}}]
}}"""

        raw = self.llm.invoke([{"role": "user", "content": prompt}], 0.2)
        data = extract_json_object(raw) or {}

        raw_branches = data.get("branches") or []
        branches: List[Dict[str, Any]] = []
        seen_bids = set()
        for b in raw_branches:
            if not isinstance(b, dict):
                continue
            try:
                bid = int(b.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(b.get("name", "")).strip()[:16] or f"分支{bid}"
            if bid in seen_bids:
                continue
            seen_bids.add(bid)
            branches.append({"id": bid, "name": name})
            if len(branches) >= MAX_BRANCHES:
                break
        if not branches:
            raise ValueError("LLM returned no branches")

        valid_bids = {b["id"] for b in branches}
        fallback_bid = branches[0]["id"]
        assign: Dict[int, int] = {}
        for a in (data.get("assignments") or []):
            if not isinstance(a, dict):
                continue
            pidx = self._as_index(a.get("paper"))
            bid = self._as_index(a.get("branch"))
            if pidx is None or bid is None:
                continue
            if 0 <= pidx < len(nodes):
                assign[pidx] = bid if bid in valid_bids else fallback_bid
        # 未分配到的论文归入第一条分支，保证完整
        for n in nodes:
            assign.setdefault(n["id"], fallback_bid)

        edges = self._sanitize_edges(data.get("edges") or [], len(nodes))
        # 给分支补色
        for i, b in enumerate(branches):
            b["color"] = LANE_COLORS[i % len(LANE_COLORS)]
        # 按分支内最早论文排序，让演进时间感从左到右递进
        earliest = {b["id"]: 10**9 for b in branches}
        for n in nodes:
            bid = assign.get(n["id"], fallback_bid)
            earliest[bid] = min(earliest[bid], n["row"])
        branches.sort(key=lambda b: earliest.get(b["id"], 10**9))
        return branches, assign, edges

    def _sanitize_edges(self, raw_edges: List[Any], n: int) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        seen = set()
        out_degree: Dict[int, int] = {}
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            src = self._as_index(e.get("from"))
            dst = self._as_index(e.get("to"))
            if src is None or dst is None:
                continue
            if not (0 <= src < n and 0 <= dst < n) or src == dst:
                continue
            # 边方向恒为早→晚（id 即时间序，行号小=更早）。晚论文不可能被早论文继承，
            # 故 from>to 必是 LLM 把 from/to 标反，交换即可恢复正确的演进方向。
            early, late = (src, dst) if src < dst else (dst, src)
            key = (early, late)
            if key in seen:
                continue
            etype = str(e.get("type", "builds_on")).strip().lower()
            if etype not in ("builds_on", "compares_with"):
                etype = "builds_on"
            if out_degree.get(late, 0) >= MAX_OUT_DEGREE:
                continue
            seen.add(key)
            out_degree[late] = out_degree.get(late, 0) + 1
            cleaned.append({
                "src": early,         # 早 → 晚（演进方向）
                "dst": late,
                "type": etype,
                "reason": str(e.get("reason", "")).strip()[:48],
            })
        return cleaned

    # ------------------------------------------------------------------ #
    # 启发式结构（无 LLM）
    # ------------------------------------------------------------------ #
    def _heuristic_structure(
        self, nodes: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]:
        # 按主类目分组
        groups: Dict[str, List[int]] = {}
        for n in nodes:
            cat = n["category"] or "其他"
            groups.setdefault(cat, []).append(n["id"])

        # 取最大的若干类目作为泳道，其余并入"其他"
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
        keep = ordered[:MAX_BRANCHES - 1] if len(ordered) > MAX_BRANCHES else ordered

        branches: List[Dict[str, Any]] = []
        cat_to_bid: Dict[str, int] = {}
        for i, (cat, _ids) in enumerate(keep):
            branches.append({"id": i, "name": self._pretty_category(cat), "color": LANE_COLORS[i % len(LANE_COLORS)]})
            cat_to_bid[cat] = i
        other_bid = None
        assign: Dict[int, int] = {}
        for n in nodes:
            cat = n["category"] or "其他"
            if cat in cat_to_bid:
                assign[n["id"]] = cat_to_bid[cat]
            else:
                if other_bid is None:
                    other_bid = len(branches)
                    branches.append({"id": other_bid, "name": "其他", "color": LANE_COLORS[other_bid % len(LANE_COLORS)]})
                assign[n["id"]] = other_bid

        # 按分支内最早论文排序泳道
        earliest = {b["id"]: 10**9 for b in branches}
        for n in nodes:
            bid = assign[n["id"]]
            earliest[bid] = min(earliest[bid], n["row"])
        branches.sort(key=lambda b: earliest.get(b["id"], 10**9))

        # 同分支内、按时间相邻的论文连一条浅色 similar_to 边（展示演进顺序，不臆造继承）
        edges: List[Dict[str, Any]] = []
        per_branch: Dict[int, List[int]] = {}
        for n in sorted(nodes, key=lambda x: x["row"]):
            per_branch.setdefault(assign[n["id"]], []).append(n["id"])
        for bid, ids in per_branch.items():
            for a, b in zip(ids, ids[1:]):
                edges.append({"src": a, "dst": b, "type": "similar_to", "reason": "同分支时间相邻"})
        return branches, assign, edges

    @staticmethod
    def _pretty_category(cat: str) -> str:
        mapping = {
            "cs.LG": "机器学习", "cs.CV": "计算机视觉", "cs.CL": "自然语言",
            "cs.AI": "人工智能", "cs.RO": "机器人", "cs.NE": "神经计算",
            "stat.ML": "统计学习", "eess.IV": "图像处理", "cs.IR": "信息检索",
            "cs.MA": "多智能体", "cs.SD": "音频", "eess.AS": "语音",
        }
        return mapping.get(cat, cat or "主线")

    # ------------------------------------------------------------------ #
    # SVG 渲染
    # ------------------------------------------------------------------ #
    def _render_svg(
        self,
        topic: str,
        nodes: List[Dict[str, Any]],
        branches: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        used_llm: bool,
    ) -> str:
        n_lanes = max(1, len(branches))
        if n_lanes <= 1:
            lane_w = 560
        elif n_lanes == 2:
            lane_w = 380
        else:
            lane_w = 300
        gap = 28
        left_gutter = 70
        right_pad = 44
        top_band = 168
        row_gap = 74
        node_h = 56
        bottom_pad = 60

        width = left_gutter + n_lanes * lane_w + (n_lanes - 1) * gap + right_pad
        n_rows = len(nodes)
        height = top_band + n_rows * row_gap + bottom_pad

        lane_color = {b["id"]: b.get("color", LANE_COLORS[i % len(LANE_COLORS)]) for i, b in enumerate(branches)}
        lane_index = {b["id"]: i for i, b in enumerate(branches)}

        def lane_cx(branch_id: int) -> float:
            i = lane_index.get(branch_id, 0)
            return left_gutter + i * (lane_w + gap) + lane_w / 2

        def row_cy(row: int) -> float:
            return top_band + row * row_gap + node_h / 2

        node_by_id = {n["id"]: n for n in nodes}
        card_w = lane_w - 28

        parts: List[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Inter, \'Segoe UI\', '
            f'\'PingFang SC\', \'Microsoft YaHei\', sans-serif">'
        )
        parts.append(self._defs())
        # 背景
        parts.append(f'<rect width="{width}" height="{height}" fill="#f8fafc"/>')

        # 年份网格线 + 标签（同一年的第一行处画一条浅线）
        seen_years = set()
        grid_x2 = width - right_pad
        for node in nodes:
            y = top_band + node["row"] * row_gap
            yr = node["year"]
            if yr and yr not in seen_years:
                seen_years.add(yr)
                parts.append(
                    f'<line x1="{left_gutter - 8}" y1="{y - 14}" x2="{grid_x2}" y2="{y - 14}" '
                    f'stroke="#e2e8f0" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{left_gutter - 16}" y="{row_cy(node["row"]) + 4}" text-anchor="end" '
                    f'font-size="15" font-weight="700" fill="#475569">{yr}</text>'
                )
            elif not yr:
                parts.append(
                    f'<text x="{left_gutter - 16}" y="{row_cy(node["row"]) + 4}" text-anchor="end" '
                    f'font-size="13" fill="#94a3b8">?</text>'
                )

        # 头部渐变 band + 标题 + 副标题 + 图例
        parts.append(f'<rect x="0" y="0" width="{width}" height="120" fill="url(#evoHeader)"/>')
        parts.append(
            f'<text x="36" y="52" font-size="27" font-weight="800" fill="#ffffff">'
            f'算法演进图 · {html.escape(self._shorten(topic, 48))}</text>'
        )
        subtitle = (
            "纵轴为发布时间（越靠下越新）· 列为技术分支 · 实线=继承/改进，虚线=对比"
            if used_llm
            else "纵轴为发布时间（越靠下越新）· 列为 arXiv 主类目分支 · 确定性排列"
        )
        parts.append(
            f'<text x="36" y="82" font-size="14" fill="#d1fae5">{html.escape(subtitle)}</text>'
        )
        mode_tag = "LLM 分支 + 演进关系" if used_llm else "无 Key · 确定性分支"
        parts.append(self._legend(width - 36, 44, mode_tag))

        # 泳道表头 pill + 竖直分隔参考线
        header_y = 132
        for b in branches:
            cx = lane_cx(b["id"])
            color = lane_color[b["id"]]
            count = sum(1 for n in nodes if n["branch"] == b["id"])
            label = f'{self._shorten(b["name"], 16)} · {count}'
            pill_w = min(lane_w - 16, max(120, len(label) * 12 + 36))
            parts.append(
                f'<rect x="{cx - pill_w / 2:.1f}" y="{header_y}" width="{pill_w:.1f}" height="30" rx="15" '
                f'fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="1.4"/>'
            )
            parts.append(
                f'<circle cx="{cx - pill_w / 2 + 18:.1f}" cy="{header_y + 15}" r="5" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{cx + 9:.1f}" y="{header_y + 20}" text-anchor="middle" '
                f'font-size="13.5" font-weight="700" fill="#0f172a">{html.escape(label)}</text>'
            )

        # 边（先画，处于节点下方）
        for e in edges:
            src = node_by_id.get(e["src"])
            dst = node_by_id.get(e["dst"])
            if not src or not dst:
                continue
            x1, y1 = lane_cx(src["branch"]), row_cy(src["row"]) + node_h / 2  # 早：底部出
            x2, y2 = lane_cx(dst["branch"]), row_cy(dst["row"]) - node_h / 2  # 晚：顶部入
            etype = e.get("type", "builds_on")
            if etype == "compares_with":
                color, dash, marker = EDGE_COMPARES, ' stroke-dasharray="6 5"', "url(#arrowCompare)"
            elif etype == "similar_to":
                color, dash, marker = EDGE_SIMILAR, ' stroke-dasharray="3 5"', ""
            else:
                color, dash, marker = EDGE_BUILDS, "", "url(#arrowBuild)"
            mid_y = (y1 + y2) / 2
            path = f"M {x1:.1f} {y1:.1f} C {x1:.1f} {mid_y:.1f}, {x2:.1f} {mid_y:.1f}, {x2:.1f} {y2:.1f}"
            marker_attr = f' marker-end="{marker}"' if marker else ""
            title = html.escape(e.get("reason", "") or etype)
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"'
                f'{dash} stroke-opacity="0.75"{marker_attr}><title>{title}</title></path>'
            )

        # 节点卡片
        for node in nodes:
            cx = lane_cx(node["branch"])
            cy = row_cy(node["row"])
            color = lane_color.get(node["branch"], LANE_COLORS[0])
            x = cx - card_w / 2
            y = cy - node_h / 2
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{card_w}" height="{node_h}" rx="11" '
                f'fill="#ffffff" stroke="{color}" stroke-width="1.4" stroke-opacity="0.85"/>'
            )
            # 左侧分支强调条
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="5" height="{node_h}" rx="2.5" fill="{color}"/>'
            )
            # 年份 + 代码点
            parts.append(
                f'<text x="{x + 16:.1f}" y="{y + 19:.1f}" font-size="11.5" font-weight="700" '
                f'fill="{color}">{node["year"] or "—"}</text>'
            )
            if node["has_code"]:
                parts.append(
                    f'<circle cx="{x + card_w - 16:.1f}" cy="{y + 15:.1f}" r="5" fill="{CODE_DOT}"/>'
                )
                parts.append(
                    f'<text x="{x + card_w - 26:.1f}" y="{y + 19:.1f}" text-anchor="end" '
                    f'font-size="10" font-weight="700" fill="{CODE_DOT}">CODE</text>'
                )
            # 标题（最多两行）
            title_lines = self._wrap(node["title"], max(14, int(card_w / 7.2)), 2)
            ty = y + 34
            tspans = ""
            for line in title_lines:
                tspans += f'<tspan x="{x + 16:.1f}" y="{ty:.1f}">{html.escape(line)}</tspan>'
                ty += 15
            parts.append(
                f'<text font-size="12.5" font-weight="600" fill="#0f172a">{tspans}'
                f'<title>{html.escape(node["title"])}</title></text>'
            )

        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def _defs() -> str:
        return (
            "<defs>"
            '<linearGradient id="evoHeader" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0" stop-color="#0f766e"/><stop offset="1" stop-color="#4f46e5"/>'
            "</linearGradient>"
            '<marker id="arrowBuild" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">'
            f'<path d="M0 0 L8 4 L0 8 z" fill="{EDGE_BUILDS}"/></marker>'
            '<marker id="arrowCompare" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">'
            f'<path d="M0 0 L8 4 L0 8 z" fill="{EDGE_COMPARES}"/></marker>'
            "</defs>"
        )

    @staticmethod
    def _legend(x_right: float, y: float, mode_tag: str) -> str:
        # 右上角图例：实线/虚线/代码点 + 模式标签
        parts = [
            f'<g font-size="12" fill="#ecfeff">',
            f'<line x1="{x_right - 250:.1f}" y1="{y}" x2="{x_right - 222:.1f}" y2="{y}" '
            f'stroke="#ffffff" stroke-width="2.4"/>',
            f'<text x="{x_right - 216:.1f}" y="{y + 4}">继承/改进</text>',
            f'<line x1="{x_right - 150:.1f}" y1="{y}" x2="{x_right - 122:.1f}" y2="{y}" '
            f'stroke="#ffffff" stroke-width="2.4" stroke-dasharray="5 4"/>',
            f'<text x="{x_right - 116:.1f}" y="{y + 4}">对比</text>',
            f'<circle cx="{x_right - 70:.1f}" cy="{y}" r="5" fill="{CODE_DOT}"/>',
            f'<text x="{x_right - 60:.1f}" y="{y + 4}">有代码</text>',
            "</g>",
            f'<text x="{x_right:.1f}" y="{y + 26}" text-anchor="end" font-size="11.5" '
            f'fill="#a7f3d0">{html.escape(mode_tag)}</text>',
        ]
        return "".join(parts)

    def _empty_svg(self, topic: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220" viewBox="0 0 720 220" '
            f'font-family="Inter, sans-serif">'
            f'<rect width="720" height="220" fill="#f8fafc"/>'
            f'<rect x="0" y="0" width="720" height="70" fill="url(#evoHeader)"/>'
            f'{self._defs()}'
            f'<text x="32" y="44" font-size="22" font-weight="800" fill="#ffffff">'
            f'算法演进图 · {html.escape(self._shorten(topic, 40))}</text>'
            f'<text x="360" y="150" text-anchor="middle" font-size="15" fill="#64748b">'
            f'暂无足够论文生成演进图</text></svg>'
        )

    # ------------------------------------------------------------------ #
    # 文本工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _wrap(text: str, max_chars: int, max_lines: int) -> List[str]:
        text = " ".join((text or "").split())
        if not text:
            return [""]
        words = text.split(" ")
        lines: List[str] = []
        cur = ""
        for w in words:
            candidate = (cur + " " + w).strip()
            if len(candidate) <= max_chars or not cur:
                cur = candidate
            else:
                lines.append(cur)
                cur = w
                if len(lines) == max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)
        # 超长截断最后一行
        if lines and len(" ".join(words)) > sum(len(l) for l in lines) + len(lines):
            last = lines[-1]
            if len(last) > max_chars:
                lines[-1] = last[: max_chars - 1] + "…"
            # 还有没放下的词
            placed = " ".join(lines)
            if len(placed) < len(text) and not lines[-1].endswith("…"):
                lines[-1] = lines[-1][: max(1, max_chars - 1)] + "…"
        return lines[:max_lines] or [""]
