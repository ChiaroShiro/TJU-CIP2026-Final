"""
论文图片提取服务。

支持三类来源：
1. arXiv HTML 页面
2. 项目主页
3. PDF 图片抽取（既支持 arXiv 下载的 PDF，也支持本地 PDF）
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests


@dataclass
class FigureRef:
    """单张图片的引用信息。"""

    index: int
    caption: str
    url: str = ""
    local_path: str = ""
    source: str = "arxiv_html"
    reachable: Optional[bool] = None


class PaperFigureFetcher:
    """论文图片提取器，包含多来源回退逻辑。"""

    HTML_URL = "https://arxiv.org/html/{arxiv_id}"
    PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
    HEADERS = {"User-Agent": "deep-research-agent/1.0"}
    REACHABILITY_TIMEOUT = 8
    MIN_IMAGE_BYTES = 10_000
    MAX_FIGURES = 20

    PROJECT_PAGE_PATTERNS = [
        r"https?://[^\s)]*\.github\.io/[^\s)]*",
        r"https?://[^\s)]*project[-_]?page[^\s)]*",
        r"https?://[^\s)]*\.io/[^\s)]+",
    ]

    def __init__(self, assets_dir: Path, method_name: Optional[str] = None):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name or "paper"

    def extract_figures(self, arxiv_id: str) -> List[FigureRef]:
        """为 arXiv 论文提取图片。"""
        figures: List[FigureRef] = []

        html_figs = self._extract_from_arxiv_html(arxiv_id)
        if html_figs:
            figures.extend(html_figs)

        if html_figs:
            project_url = self._find_project_page(arxiv_id)
            if project_url:
                project_figs = self._extract_from_project_page(
                    project_url,
                    start_index=len(figures) + 1,
                )
                if project_figs:
                    figures.extend(project_figs)

        if not figures:
            figures = self._extract_from_pdf(arxiv_id)

        return figures

    def extract_figures_from_local_pdf(self, pdf_path: Path) -> List[FigureRef]:
        """直接从本地 PDF 中抽取图片。"""
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".pdf":
            return []
        return self._extract_from_pdf_file(path)

    def localize_unreachable(self, figures: List[FigureRef]) -> int:
        """将不可访问的外链图片下载到本地。"""
        localized = 0
        for fig in figures:
            if fig.local_path or not fig.url:
                continue
            if self._is_reachable(fig.url):
                fig.reachable = True
                continue
            fig.reachable = False
            local = self._download_image(fig.url, fig.index)
            if local:
                fig.local_path = str(local)
                localized += 1
        return localized

    def _extract_from_arxiv_html(self, arxiv_id: str) -> List[FigureRef]:
        url = self.HTML_URL.format(arxiv_id=arxiv_id)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200:
                return []
            html = resp.text
        except Exception as exc:
            print(f"[PaperFigureFetcher] arxiv html fetch failed: {exc}")
            return []

        figures: List[FigureRef] = []
        figure_blocks = re.findall(
            r"<figure[^>]*>(.*?)</figure>",
            html,
            re.DOTALL | re.IGNORECASE,
        )
        for i, block in enumerate(figure_blocks, 1):
            img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', block, re.IGNORECASE)
            if not img_match:
                continue
            raw_src = img_match.group(1)
            cap_match = re.search(
                r"<figcaption[^>]*>(.*?)</figcaption>",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            caption = self._strip_html(cap_match.group(1)) if cap_match else f"Figure {i}"
            normalized_url = self._normalize_arxiv_url(raw_src, arxiv_id, base_url=url)
            figures.append(
                FigureRef(
                    index=i,
                    caption=caption[:200],
                    url=normalized_url,
                    source="arxiv_html",
                )
            )
        return figures

    def _normalize_arxiv_url(self, raw_src: str, arxiv_id: str, base_url: str) -> str:
        if raw_src.startswith("http://") or raw_src.startswith("https://"):
            return self._dedupe_path_segment(raw_src, arxiv_id)
        joined = urljoin(base_url + "/", raw_src.lstrip("/"))
        return self._dedupe_path_segment(joined, arxiv_id)

    @staticmethod
    def _dedupe_path_segment(url: str, arxiv_id: str) -> str:
        bare_id = arxiv_id.rstrip("v0123456789").rstrip(".")
        for token in (arxiv_id, bare_id):
            if not token:
                continue
            dup = f"/{token}/{token}/"
            if dup in url:
                url = url.replace(dup, f"/{token}/")
        return url

    def _find_project_page(self, arxiv_id: str) -> Optional[str]:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            if resp.status_code != 200:
                return None
            text = resp.text
        except Exception:
            return None

        for pattern in self.PROJECT_PAGE_PATTERNS:
            match = re.search(pattern, text)
            if not match:
                continue
            candidate = match.group(0).rstrip(".,)\"'")
            if "arxiv.org" in candidate or ("github.com" in candidate and "/blob/" in candidate):
                continue
            return candidate
        return None

    def _extract_from_project_page(self, page_url: str, start_index: int = 1) -> List[FigureRef]:
        try:
            resp = requests.get(page_url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200:
                return []
            html = resp.text
        except Exception:
            return []

        figures: List[FigureRef] = []
        img_matches = re.findall(
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?',
            html,
            re.IGNORECASE,
        )
        idx = start_index
        for src, alt in img_matches:
            if any(skip in src.lower() for skip in ["icon", "logo", "favicon", "avatar"]):
                continue
            figures.append(
                FigureRef(
                    index=idx,
                    caption=alt or f"Project figure {idx}",
                    url=urljoin(page_url, src),
                    source="project_page",
                )
            )
            idx += 1
            if len(figures) >= 5:
                break
        return figures

    def _extract_from_pdf(self, arxiv_id: str) -> List[FigureRef]:
        try:
            resp = requests.get(
                self.PDF_URL.format(arxiv_id=arxiv_id),
                headers=self.HEADERS,
                timeout=30,
            )
            if resp.status_code != 200:
                return []
            return self._extract_from_pdf_bytes(resp.content)
        except Exception:
            return []

    def _extract_from_pdf_file(self, pdf_path: Path) -> List[FigureRef]:
        try:
            return self._extract_from_pdf_bytes(pdf_path.read_bytes())
        except Exception as exc:
            print(f"[PaperFigureFetcher] local pdf extract failed: {exc}")
            return []

    def _extract_from_pdf_bytes(self, pdf_bytes: bytes) -> List[FigureRef]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return []

        figures: List[FigureRef] = []
        seen_hashes = set()

        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc, 1):
                    for img in page.get_images(full=True):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_bytes = pix.tobytes("png")
                        pix = None
                        if len(img_bytes) < self.MIN_IMAGE_BYTES:
                            continue
                        digest = hash(img_bytes)
                        if digest in seen_hashes:
                            continue
                        seen_hashes.add(digest)
                        fig_index = len(figures) + 1
                        local = self.assets_dir / f"{self.method_name}_fig{fig_index}.png"
                        local.write_bytes(img_bytes)
                        figures.append(
                            FigureRef(
                                index=fig_index,
                                caption=f"Figure extracted from PDF page {page_num}",
                                local_path=str(local),
                                source="pdf_extract",
                            )
                        )
                        if len(figures) >= self.MAX_FIGURES:
                            return figures
        except Exception as exc:
            print(f"[PaperFigureFetcher] pdf extract failed: {exc}")

        return figures

    def _is_reachable(self, url: str) -> bool:
        try:
            resp = requests.head(
                url,
                headers=self.HEADERS,
                timeout=self.REACHABILITY_TIMEOUT,
                allow_redirects=True,
            )
            return resp.status_code < 400
        except Exception:
            return False

    def _download_image(self, url: str, index: int) -> Optional[Path]:
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
