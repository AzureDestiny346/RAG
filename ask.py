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


def ask_pipeline(question: str) -> dict:
    """检索 -> 生成 -> 结构化结果（CLI 与 W8 HTTP 层共用的唯一入口）。"""
    bm25, index, metas = load()
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunks = retrieve(question, bm25, index, metas, model)
    client = make_client()
    return {
        "question": question,
        "answer": answer(client, question, chunks),
        "sources": [
            {"chunk_id": c["chunk_id"], "source": c["source"],
             "heading": c["heading"], "in_bm25": c["in_bm25"],
             "in_vector": c["in_vector"]}
            for c in chunks
        ],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python ask.py 你的问题")
        return 1
    result = ask_pipeline(sys.argv[1])

    print(f"[问题] {result['question']}")
    marks = " | ".join(
        f"#{s['chunk_id']}{'B' if s['in_bm25'] else ''}"
        f"{'V' if s['in_vector'] else ''} {s['heading'][:20]}"
        for s in result["sources"])
    print(f"[检索] {len(result['sources'])} 块: {marks}")
    print(f"\n[回答]\n{result['answer']}")
    sources = " | ".join(
        f"#{s['chunk_id']} {s['source']}:{s['heading'][:24]}"
        for s in result["sources"])
    print(f"\n[来源] {sources}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
