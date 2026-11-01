"""标题感知分块（手册第 3 章定案）。

优先按 Markdown 标题层级切分（主题边界清晰）；
超长小节走固定长度兜底，在句末标点处断开并保留 overlap，
避免把一句话硬劈成两半。每个块携带 source/heading 元数据。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import CHUNK_OVERLAP, CHUNK_SIZE

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_ENDINGS = "。！？；!?"


@dataclass
class Chunk:
    """最小检索单元：文本 + 元数据（元数据从第一天就记录——手册第 2 章）。"""

    chunk_id: int = 0
    source: str = ""
    heading: str = ""
    text: str = ""


def split_long_text(text: str) -> list[str]:
    """固定长度兜底切分：尽量在句末标点处断开，相邻块带 overlap。

    保证：每块长度 <= CHUNK_SIZE；除末块外都尽量以句末标点收尾
    （找不到句界才硬切）；相邻块前 CHUNK_OVERLAP 字符重叠。
    """
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            window = text[start:end]
            # 从窗口尾部向前找最后一个句末标点，保证块以完整句子收尾
            cut = max(window.rfind(ch) for ch in _SENTENCE_ENDINGS)
            if cut >= CHUNK_OVERLAP:  # 句界太靠前会切出碎块，不如硬切
                end = start + cut + 1
            # 找不到句界：接受硬切（funNLP 语料的表格类内容常见）
        pieces.append(text[start:end])
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)  # 保证前进，防死循环
    return pieces


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """按标题层级切块；块文本带上标题路径（标题词参与检索，即"标题感知"）。

    第一个标题之前的内容归入"（前言）"块，避免静默丢数据。
    """
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (级别, 标题)
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        body_lines.clear()
        if not body:
            return
        heading_path = " > ".join(title for _, title in heading_stack) or "（前言）"
        for piece in split_long_text(body):
            chunks.append(Chunk(source=source, heading=heading_path,
                                text=f"{heading_path}\n{piece}"))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, m.group(2)))
        else:
            body_lines.append(line)
    flush()

    for i, c in enumerate(chunks):
        c.chunk_id = i
    return chunks
