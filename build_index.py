"""数据处理管道入口（W5）：读语料 → 标题感知分块 → chunks.jsonl。

用法：python build_index.py
W6 的索引构建（BM25 + FAISS）将读取本脚本产物，管道职责单一。
"""
from __future__ import annotations

import json
import sys

from chunking import chunk_markdown
from config import CHUNKS_PATH, CORPUS_DIR, ensure_dirs


def main() -> int:
    ensure_dirs()
    md_files = sorted(CORPUS_DIR.glob("*.md"))
    if not md_files:
        print(f"[错误] {CORPUS_DIR} 下没有 .md 语料，请先放入文档。")
        return 1

    all_chunks = []
    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(text, source=f.name)
        all_chunks.extend(chunks)
        print(f"  {f.name}: {len(chunks)} 块")

    # 全局统一编号（单文件内的局部编号在此被覆盖）
    for i, c in enumerate(all_chunks):
        c.chunk_id = i

    with CHUNKS_PATH.open("w", encoding="utf-8") as w:
        for c in all_chunks:
            w.write(json.dumps(
                {"chunk_id": c.chunk_id, "source": c.source,
                 "heading": c.heading, "text": c.text},
                ensure_ascii=False) + "\n")

    lengths = sorted(len(c.text) for c in all_chunks)
    n = len(lengths)
    print(f"[完成] 共 {n} 块 -> {CHUNKS_PATH.name}")
    print(f"  块长 min/median/max = {lengths[0]}/{lengths[n // 2]}/{lengths[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
