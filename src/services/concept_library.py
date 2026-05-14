"""
概念库管理（参考 dailypaper-skills/paper-reader）

每篇论文笔记里的 [[Concept]] 链接，对应概念库中独立的 Markdown 文件。
当一个新概念首次出现时，自动创建一个 stub 文件等用户后续补充。

放在 workspace/paper_notes/concepts/ 下。
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Set


CONCEPT_LINK_PATTERN = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]+)?\]\]")


class ConceptLibrary:
    """
    概念库维护器。

    使用方式：
        lib = ConceptLibrary(concepts_dir=Path("workspace/paper_notes/concepts"))
        new_concepts = lib.scan_and_create(note_text, paper_method_name="DPO")
        # → ["Direct Preference Optimization", "RLHF", ...]
    """

    STUB_TEMPLATE = """---
type: concept
created: {created}
---

# {concept_name}

> 概念占位文件 — 由 paper-reader 自动创建。

## 定义

（待补充）

## 相关方法

- [[{first_seen_in}]] — 首次出现于该论文笔记

## 参考

（待补充）
"""

    def __init__(self, concepts_dir: Path):
        self.concepts_dir = Path(concepts_dir)
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_create(self, note_text: str, paper_method_name: str = "") -> List[str]:
        """
        扫描笔记中的 [[Concept]] 链接，为不存在的概念创建占位文件。

        返回新创建的概念名列表。
        """
        concepts = self._extract_concepts(note_text)
        created: List[str] = []
        for name in concepts:
            if self._concept_exists(name):
                self._append_reference(name, paper_method_name)
            else:
                self._create_stub(name, paper_method_name)
                created.append(name)
        return created

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _extract_concepts(self, text: str) -> Set[str]:
        """提取所有 [[Concept]] / [[Concept|alias]] 中的 Concept 部分。"""
        return {
            self._sanitize(m.group(1).strip())
            for m in CONCEPT_LINK_PATTERN.finditer(text)
            if m.group(1).strip()
        }

    @staticmethod
    def _sanitize(name: str) -> str:
        """概念名规范化：去除特殊字符、希腊字母转写。"""
        replacements = {"π": "Pi", "σ": "Sigma", "α": "Alpha", "β": "Beta",
                        "γ": "Gamma", "λ": "Lambda", "θ": "Theta", "μ": "Mu"}
        for greek, ascii_name in replacements.items():
            name = name.replace(greek, ascii_name)
        return name.strip()

    def _concept_path(self, name: str) -> Path:
        """规范的概念文件路径。"""
        # 文件名安全化（保留字母数字+空格+连字符）
        safe = "".join(c if c.isalnum() or c in " -_" else "" for c in name)
        safe = safe.strip().replace("  ", " ")
        return self.concepts_dir / f"{safe}.md"

    def _concept_exists(self, name: str) -> bool:
        return self._concept_path(name).exists()

    def _create_stub(self, name: str, first_seen_in: str) -> None:
        path = self._concept_path(name)
        content = self.STUB_TEMPLATE.format(
            concept_name=name,
            created=datetime.utcnow().strftime("%Y-%m-%d"),
            first_seen_in=first_seen_in or "unknown",
        )
        path.write_text(content, encoding="utf-8")

    def _append_reference(self, name: str, paper_method_name: str) -> None:
        """已存在的概念，把当前论文追加到"相关方法"区块（避免重复）。"""
        if not paper_method_name:
            return
        path = self._concept_path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return
        # 已经引用过则跳过
        if f"[[{paper_method_name}]]" in text:
            return
        # 简单追加到末尾
        addition = f"\n- [[{paper_method_name}]] — 相关论文\n"
        if "## 相关方法" in text:
            # 插入到"相关方法"小节末尾
            text = re.sub(
                r"(## 相关方法\n(?:.*\n)*?)(?=\n## |\Z)",
                lambda m: m.group(1) + addition,
                text, count=1,
            )
        else:
            text += "\n## 相关方法\n" + addition
        path.write_text(text, encoding="utf-8")
