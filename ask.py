"""CLI 问答（W7 演示载体）：检索 -> 生成 -> 答案 + 溯源。

用法：python ask.py 你的问题
前置：已建索引；.env 已配置 DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import sys

from config import EMBEDDING_MODEL
from generation import answer, make_client
from indexing import load
from retrieval import retrieve
from sentence_transformers import SentenceTransformer


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python ask.py 你的问题")
        return 1
    question = sys.argv[1]

    bm25, index, metas = load()
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunks = retrieve(question, bm25, index, metas, model)

    marks = " | ".join(
        f"#{c['chunk_id']}{'B' if c['in_bm25'] else ''}"
        f"{'V' if c['in_vector'] else ''} {c['heading'][:20]}"
        for c in chunks)
    print(f"[检索] {len(chunks)} 块: {marks}")

    client = make_client()
    print("\n[回答]")
    print(answer(client, question, chunks))

    sources = " | ".join(
        f"#{c['chunk_id']} {c['source']}:{c['heading'][:24]}" for c in chunks)
    print(f"\n[来源] {sources}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
