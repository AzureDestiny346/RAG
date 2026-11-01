"""双路索引构建（W6）：BM25 稀疏 + BGE 向量，FAISS 本地持久化。

用法：
  python indexing.py            # 读取 chunks.jsonl，构建并持久化双索引
  python indexing.py 查询词     # 构建后附带一次冒烟查询，打印两路 Top-3

首跑自动下载 bge-small-zh-v1.5（约 100MB）：走 Clash 代理，或设
HF_ENDPOINT=https://hf-mirror.com 加速。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_PATH,
    CHUNKS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_PATH,
    INDEX_DIR,
    META_PATH,
    ensure_dirs,
)
from retrieval import Engine, dual_recall, tokenize


def build_bm25(corpus: list[dict]) -> BM25Okapi:
    """BM25 稀疏索引：块文本已含标题路径前缀，专有名词/标题词可精确命中。"""
    return BM25Okapi([tokenize(c["text"]) for c in corpus])


def encode_passages(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """文档向量：L2 归一化 + IndexFlatIP 内积 = 余弦相似度。"""
    return model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")


def persist(bm25: BM25Okapi, index: faiss.Index, metas: list[dict],
            bm25_path: Path, faiss_path: Path, meta_path: Path) -> None:
    """三件套持久化：BM25 pickle / FAISS write_index / 元数据 jsonl。

    metas 与 FAISS 行序、BM25 行序一一对应（都来自 corpus 顺序）。
    """
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(index, str(faiss_path))
    with meta_path.open("w", encoding="utf-8") as w:
        for m in metas:
            w.write(json.dumps(m, ensure_ascii=False) + "\n")


def load(bm25_path: Path = BM25_PATH, faiss_path: Path = FAISS_PATH,
         meta_path: Path = META_PATH) -> tuple[BM25Okapi, faiss.Index, list[dict]]:
    """热启动：从磁盘加载双索引与元数据（ask._engine 会补上模型组成 Engine）。"""
    for p in (bm25_path, faiss_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}，请先运行 python indexing.py")
    with bm25_path.open("rb") as f:
        bm25 = pickle.load(f)
    index = faiss.read_index(str(faiss_path))
    metas = [json.loads(line) for line in
             meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return bm25, index, metas


def smoke_query(engine: Engine, query: str, top_k: int = 3) -> None:
    """冒烟查询：两路各打 Top-K，人眼验证索引质量（融合逻辑在 retrieval）。"""
    print(f"\n[冒烟查询] {query}")
    bm25_ids, vec_ids = dual_recall(query, engine.bm25, engine.index, engine.model)
    for label, ids in (("BM25", bm25_ids), ("向量", vec_ids)):
        print(f"{label} Top-{top_k}:")
        for rank, i in enumerate(ids[:top_k], 1):
            m = engine.metas[i]
            print(f"  {rank}. [{m['source']}|{m['heading'][:40]}] {m['text'][:60]}...")


def main() -> int:
    if not CHUNKS_PATH.exists():
        print("[错误] 缺少 chunks.jsonl，先运行 python build_index.py")
        return 1
    ensure_dirs()
    corpus = [json.loads(line) for line in
              CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"语料块数: {len(corpus)}")

    t0 = time.perf_counter()
    bm25 = build_bm25(corpus)
    print(f"BM25 构建: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL)  # 首跑在此自动下载
    print(f"模型加载: {time.perf_counter() - t1:.2f}s（首跑含下载，更慢属正常）")

    t2 = time.perf_counter()
    vecs = encode_passages(model, [c["text"] for c in corpus])
    if vecs.shape[1] != EMBEDDING_DIM:
        print(f"[错误] 模型维度 {vecs.shape[1]} 与配置 {EMBEDDING_DIM} 不一致")
        return 1
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    print(f"向量编码+索引: {time.perf_counter() - t2:.2f}s"
          f"（{vecs.shape[0]} 条 × {vecs.shape[1]} 维）")

    t3 = time.perf_counter()
    persist(bm25, index, corpus, BM25_PATH, FAISS_PATH, META_PATH)
    print(f"持久化: {time.perf_counter() - t3:.2f}s -> {INDEX_DIR.name}/")

    # 冷/热启动耗时：简历 bullet 的计量口径由此而来
    t4 = time.perf_counter()
    load()
    print(f"热启动（磁盘加载双索引）: {time.perf_counter() - t4:.2f}s")
    print(f"冷启动合计（构建到持久化）: {time.perf_counter() - t0:.2f}s")

    if len(sys.argv) > 1:
        smoke_query(Engine(bm25, index, corpus, model), sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
