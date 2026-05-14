"""
论文图片提取服务（参考 dailypaper-skills/paper-reader）

核心能力：
1. 多源 fallback：arXiv HTML → 项目主页 → PDF 提取
2. URL 规范化：去除重复 arxiv_id 路径段
3. 可达性检查：HEAD 请求验证外链有效性
4. 选择性本地化：不可达图片自动下载到 assets/

零遗漏原则：每张论文图片都应被笔记引用，由 caller 校验数量。
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests


@dataclass
class FigureRef:
    """单张图的引用信息。"""
    index: int                       # Figure 编号（1-based，与论文一致）
    caption: str                     # 图说
    url: str = ""                    # 外链或本地路径
    local_path: str = ""             # 本地下载路径（如有）
    source: str = "arxiv_html"       # arxiv_html / project_page / pdf_extract / unknown
    reachable: Optional[bool] = None # None=未检查


class PaperFigureFetcher:
    """
    论文图片提取器 - 多源 fallback。

    使用方式：
        fetcher = PaperFigureFetcher(assets_dir=Path("workspace/paper_notes/assets"))
        figs = fetcher.extract_figures(arxiv_id="2301.00234")
        # → [FigureRef(index=1, caption="...", url="https://arxiv.org/html/.../x1.png"), ...]

        fetcher.localize_unreachable(figs)
        # 把不可达的外链下载到 assets_dir
    """

    HTML_URL = "https://arxiv.org/html/{arxiv_id}"
    HEADERS = {"User-Agent": "deep-research-agent/1.0"}
    REACHABILITY_TIMEOUT = 8

    # 用于识别项目主页的 URL 模式
    PROJECT_PAGE_PATTERNS = [
        r"https?://[^\s)]*\.github\.io/[^\s)]*",
        r"https?://[^\s)]*project[-_]?page[^\s)]*",
        r"https?://[^\s)]*\.io/[^\s)]+",
    ]

    def __init__(self, assets_dir: Path, method_name: Optional[str] = None):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name or "paper"

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def extract_figures(self, arxiv_id: str) -> List[FigureRef]:
        """
        多源 fallback 提取所有图片。

        优先级：arXiv HTML → 项目主页 → PDF 提取
        """
        figures: List[FigureRef] = []

        # Source A: arXiv HTML（最完整）
        html_figs = self._extract_from_arxiv_html(arxiv_id)
        if html_figs:
            figures.extend(html_figs)

        # Source B: 项目主页（补充）
        if html_figs:
            project_url = self._find_project_page(arxiv_id)
            if project_url:
                project_figs = self._extract_from_project_page(project_url, start_index=len(figures) + 1)
                if project_figs:
                    figures.extend(project_figs)

        # Source C: PDF 提取（仅当上述都失败）
        if not figures:
            figures = self._extract_from_pdf(arxiv_id)

        return figures

    def localize_unreachable(self, figures: List[FigureRef]) -> int:
        """
        检查每个外链可达性，不可达的下载到 assets_dir。
        返回本地化数量（用于决定 frontmatter 的 image_source）。
        """
        localized = 0
        for fig in figures:
            if fig.local_path or not fig.url:
                continue
            if self._is_reachable(fig.url):
                fig.reachable = True
                continue
            fig.reachable = False
            # 不可达 → 下载到本地
            local = self._download_image(fig.url, fig.index)
            if local:
                fig.local_path = str(local)
                localized += 1
        return localized

    # ------------------------------------------------------------------ #
    # Source A: arXiv HTML
    # ------------------------------------------------------------------ #

    def _extract_from_arxiv_html(self, arxiv_id: str) -> List[FigureRef]:
        """从 arXiv HTML 提取 figures（尽可能匹配 figure 编号）。"""
        url = self.HTML_URL.format(arxiv_id=arxiv_id)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200:
                return []
            html = resp.text
        except Exception as e:
            print(f"[PaperFigureFetcher] arxiv html fetch failed: {e}")
            return []

        # 解析 <figure> 块：caption + img src
        # arXiv HTML 的 figure 通常带 id="Sx.Fx" 这类标记
        figures: List[FigureRef] = []
        figure_blocks = re.findall(
            r"<figure[^>]*>(.*?)</figure>",
            html, re.DOTALL | re.IGNORECASE,
        )
        for i, block in enumerate(figure_blocks, 1):
            img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', block, re.IGNORECASE)
            if not img_match:
                continue
            raw_src = img_match.group(1)
            # 提取 figcaption 文本
            cap_match = re.search(
                r"<figcaption[^>]*>(.*?)</figcaption>",
                block, re.DOTALL | re.IGNORECASE,
            )
            caption = self._strip_html(cap_match.group(1)) if cap_match else f"Figure {i}"

            normalized_url = self._normalize_arxiv_url(raw_src, arxiv_id, base_url=url)
            figures.append(FigureRef(
                index=i, caption=caption[:200],
                url=normalized_url, source="arxiv_html",
            ))

        return figures

    def _normalize_arxiv_url(self, raw_src: str, arxiv_id: str, base_url: str) -> str:
        """
        URL 规范化（防止 arxiv_id 路径重复 bug）

        例：raw=`2603.05312v1/x1.png` + base=`https://arxiv.org/html/2603.05312v1`
            → 错误拼接：`https://arxiv.org/html/2603.05312v1/2603.05312v1/x1.png`
            → 正确：    `https://arxiv.org/html/2603.05312v1/x1.png`
        """
        # 已是绝对 URL 直接返回
        if raw_src.startswith("http://") or raw_src.startswith("https://"):
            return self._dedupe_path_segment(raw_src, arxiv_id)

        # 相对路径 → 用 urljoin
        joined = urljoin(base_url + "/", raw_src.lstrip("/"))
        return self._dedupe_path_segment(joined, arxiv_id)

    @staticmethod
    def _dedupe_path_segment(url: str, arxiv_id: str) -> str:
        """去除 URL 中重复的 arxiv_id 路径段。"""
        # 匹配 /<id>/<id>/ 形式（包括 v1 等版本号）
        bare_id = arxiv_id.rstrip("v0123456789").rstrip(".")
        for token in (arxiv_id, bare_id):
            if not token:
                continue
            dup = f"/{token}/{token}/"
            if dup in url:
                url = url.replace(dup, f"/{token}/")
        return url

    # ------------------------------------------------------------------ #
    # Source B: 项目主页
    # ------------------------------------------------------------------ #

    def _find_project_page(self, arxiv_id: str) -> Optional[str]:
        """从 arXiv abstract 页面找项目主页 URL。"""
        url = f"https://arxiv.org/abs/{arxiv_id}"
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            if resp.status_code != 200:
                return None
            text = resp.text
        except Exception:
            return None

        for pattern in self.PROJECT_PAGE_PATTERNS:
            m = re.search(pattern, text)
            if m:
                cand = m.group(0).rstrip(".,)\"'")
                # 排除常见噪声
                if "arxiv.org" in cand or "github.com" in cand and "/blob/" in cand:
                    continue
                return cand
        return None

    def _extract_from_project_page(self, page_url: str, start_index: int = 1) -> List[FigureRef]:
        """从项目主页抓 teaser/demo 图。"""
        try:
            resp = requests.get(page_url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200:
                return []
            html = resp.text
        except Exception:
            return []

        # 简单提取所有 <img>，过滤明显的图标/UI 元素
        figures: List[FigureRef] = []
        img_matches = re.findall(
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?',
            html, re.IGNORECASE,
        )
        idx = start_index
        for src, alt in img_matches:
            if any(skip in src.lower() for skip in ["icon", "logo", "favicon", "avatar"]):
                continue
            full_url = urljoin(page_url, src)
            figures.append(FigureRef(
                index=idx, caption=alt or f"Project figure {idx}",
                url=full_url, source="project_page",
            ))
            idx += 1
            if len(figures) >= 5:  # 主页图片不要太多
                break

        return figures

    # ------------------------------------------------------------------ #
    # Source C: PDF 提取
    # ------------------------------------------------------------------ #

    def _extract_from_pdf(self, arxiv_id: str) -> List[FigureRef]:
        """从 PDF 提取图片（最后兜底）。"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return []

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            resp = requests.get(pdf_url, headers=self.HEADERS, timeout=30)
            if resp.status_code != 200:
                return []
            pdf_bytes = resp.content
        except Exception:
            return []

        figures: List[FigureRef] = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num, page in enumerate(doc, 1):
                for img_idx, img in enumerate(page.get_images(full=True), 1):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK 转 RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    # 仅保留 >10KB 的图（小图多是装饰）
                    img_bytes = pix.tobytes("png")
                    if len(img_bytes) < 10_000:
                        continue
                    fig_index = len(figures) + 1
                    local = self.assets_dir / f"{self.method_name}_fig{fig_index}.png"
                    local.write_bytes(img_bytes)
                    figures.append(FigureRef(
                        index=fig_index,
                        caption=f"Figure extracted from PDF page {page_num}",
                        local_path=str(local),
                        source="pdf_extract",
                    ))
                    if len(figures) >= 20:  # 防止超大论文卡死
                        break
            doc.close()
        except Exception as e:
            print(f"[PaperFigureFetcher] pdf extract failed: {e}")

        return figures

    # ------------------------------------------------------------------ #
    # 可达性检查 + 下载
    # ------------------------------------------------------------------ #

    def _is_reachable(self, url: str) -> bool:
        """HEAD 请求检查 URL 可达性。"""
        try:
            resp = requests.head(
                url, headers=self.HEADERS,
                timeout=self.REACHABILITY_TIMEOUT, allow_redirects=True,
            )
            return resp.status_code < 400
        except Exception:
            return False

    def _download_image(self, url: str, index: int) -> Optional[Path]:
        """下载图片到 assets_dir。"""
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 1000:
                return None
            ext = self._guess_ext(url, resp.headers.get("Content-Type", ""))
            path = self.assets_dir / f"{self.method_name}_fig{index}{ext}"
            path.write_bytes(resp.content)
            return path
        except Exception:
            return None

    @staticmethod
    def _guess_ext(url: str, content_type: str) -> str:
        path = urlparse(url).path.lower()
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
            if path.endswith(ext):
                return ext
        if "png" in content_type:
            return ".png"
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        return ".png"

    @staticmethod
    def _strip_html(html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text
