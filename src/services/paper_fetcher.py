"""
论文全文获取服务。

支持两种来源：
1. 通过 arXiv ID 下载 PDF 并提取全文
2. 直接读取本地 PDF 并提取全文
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests


@dataclass
class PaperFullText:
    """结构化后的论文全文。"""

    arxiv_id: str
    pdf_path: str
    raw_text: str
    num_pages: int = 0
    sections: Dict[str, str] = field(default_factory=dict)
    num_chars: int = 0


class PaperFetcher:
    """
    arXiv PDF 下载与全文提取。

    也支持从本地 PDF 直接提取文本。
    """

    PDF_URL_TEMPLATES = [
        "https://arxiv.org/pdf/{id}.pdf",
        "https://arxiv.org/pdf/{id}",
    ]

    SECTION_PATTERNS = [
        (r"\babstract\b", "abstract"),
        (r"\bintroduction\b|\b1\s*introduction\b", "introduction"),
        (r"\brelated\s+work\b|\bbackground\b", "related_work"),
        (r"\bmethod(s|ology)?\b|\bapproach\b|\bproposed\b", "method"),
        (r"\bexperiment(s|al)?\b|\bevaluation\b|\bresults?\b", "experiments"),
        (r"\bdiscussion\b|\banalysis\b", "discussion"),
        (r"\bconclusion(s)?\b|\bfuture\s+work\b", "conclusion"),
        (r"\breferences?\b|\bbibliography\b", "references"),
    ]

    def __init__(self, cache_dir: Path, timeout: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_fulltext(self, arxiv_id: str) -> Optional[PaperFullText]:
        """下载 arXiv PDF 并提取全文。"""
        arxiv_id = self._normalize_id(arxiv_id)
        if not arxiv_id:
            return None

        pdf_path = self._download_pdf(arxiv_id)
        if pdf_path is None:
            return None

        return self._build_fulltext_from_pdf(pdf_path, paper_id=arxiv_id)

    def fetch_local_fulltext(
        self,
        pdf_path: Union[str, Path],
        paper_id: Optional[str] = None,
    ) -> Optional[PaperFullText]:
        """从本地 PDF 文件提取全文。"""
        path = Path(pdf_path).expanduser()
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() != ".pdf":
            return None

        local_id = paper_id or path.stem
        return self._build_fulltext_from_pdf(path.resolve(), paper_id=local_id)

    def infer_title_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """优先从 PDF metadata 推断标题，失败时回退到文件名。"""
        path = Path(pdf_path).expanduser()
        fallback = path.stem
        if not path.exists():
            return fallback

        try:
            import fitz  # PyMuPDF

            with fitz.open(str(path)) as doc:
                title = (doc.metadata or {}).get("title", "")
            title = (title or "").strip()
            return title or fallback
        except Exception:
            return fallback

    @staticmethod
    def _normalize_id(arxiv_id: str) -> str:
        arxiv_id = (arxiv_id or "").strip()
        if not arxiv_id:
            return ""
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.rstrip("/").split("/")[-1]
        if arxiv_id.endswith(".pdf"):
            arxiv_id = arxiv_id[:-4]
        return arxiv_id

    def _download_pdf(self, arxiv_id: str) -> Optional[Path]:
        cache_file = self.cache_dir / f"{arxiv_id}.pdf"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file

        last_err = None
        for tpl in self.PDF_URL_TEMPLATES:
            url = tpl.format(id=arxiv_id)
            for attempt in range(2):
                try:
                    resp = requests.get(
                        url,
                        timeout=self.timeout,
                        headers={"User-Agent": "deep-research-agent/1.0"},
                        stream=True,
                    )
                    if resp.status_code == 200:
                        cache_file.write_bytes(resp.content)
                        return cache_file
                    if resp.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    last_err = f"HTTP {resp.status_code}"
                    break
                except Exception as e:
                    last_err = str(e)[:120]
                    time.sleep(1)

        print(f"[PaperFetcher] download failed for {arxiv_id}: {last_err}")
        return None

    def _extract_text(self, pdf_path: Path) -> tuple:
        import fitz  # PyMuPDF

        text_parts: List[str] = []
        with fitz.open(str(pdf_path)) as doc:
            num_pages = doc.page_count
            for page in doc:
                text_parts.append(page.get_text("text"))

        raw_text = "\n".join(text_parts)
        raw_text = re.sub(r"-\n(\w)", r"\1", raw_text)
        return raw_text, num_pages

    def _split_sections(self, text: str) -> Dict[str, str]:
        lines = text.split("\n")
        sections: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_buf: List[str] = []

        for line in lines:
            clean = line.strip().lower()
            matched_key = None
            if 2 <= len(clean) <= 60:
                for pat, key in self.SECTION_PATTERNS:
                    if re.search(pat, clean):
                        matched_key = key
                        break

            if matched_key:
                if current_key and current_buf:
                    sections[current_key] = "\n".join(current_buf).strip()
                current_key = matched_key
                current_buf = []
            elif current_key:
                current_buf.append(line)

        if current_key and current_buf:
            sections[current_key] = "\n".join(current_buf).strip()

        return sections

    def _build_fulltext_from_pdf(self, pdf_path: Path, paper_id: str) -> Optional[PaperFullText]:
        try:
            raw_text, num_pages = self._extract_text(pdf_path)
        except Exception as e:
            print(f"[PaperFetcher] text extraction failed: {e}")
            return None

        sections = self._split_sections(raw_text)
        return PaperFullText(
            arxiv_id=paper_id,
            pdf_path=str(pdf_path),
            raw_text=raw_text,
            num_pages=num_pages,
            sections=sections,
            num_chars=len(raw_text),
        )
