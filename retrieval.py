"""混合检索（W7）：双路召回 + RRF 融合，评测/问答/服务共用的唯一入口。

RRF 公式：score(d) = sum(1 / (RRF_K + rank_i(d)))，RRF_K=60（config）。
两路互补：BM25 兜"点名道姓一定找到"，向量兜"换个说法也能找到"（手册第 6 章）。
"""
from __future__ import annotations

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import BM25_TOP_K, FINAL_TOP_K, RRF_K, VECTOR_TOP_K
from indexing import encode_query, tokenize


def rrf_fuse(bm25_ids: list[int], vec_ids: list[int],
             top_k: int = FINAL_TOP_K) -> list[int]:
    """两路排名 RRF 融合：得分求和，平局按首次出现序稳定排序，截断 top_k。"""
    scores: dict[int, float] = {}
    order: dict[int, int] = {}
    for ids in (bm25_ids, vec_ids):
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            order.setdefault(cid, len(order))
    return sorted(scores, key=lambda c: (-scores[c], order[c]))[:top_k]


def is_hit(qa: dict, top_ids: list[int], metas: list[dict]) -> bool:
    """命中判定（标题路径包含口径）：Top-K 内任一块 heading 含标注标题子串。

    口径依据 spec：不绑 chunk_id（分块一改标注全废），绑定小节语义。
    """
    expected = qa["expected_heading"]
    return any(expected in metas[i]["heading"] for i in top_ids)


def retrieve(query: str, bm25: BM25Okapi, index: faiss.Index, metas: list[dict],
             model: SentenceTransformer, top_k: int = FINAL_TOP_K) -> list[dict]:
    """混合检索唯一入口：双路召回 -> RRF 融合 -> Top-K 块（附渠道信息）。

    返回的每块带 in_bm25 / in_vector 标记，供调试与展示溯源。
    evaluate / ask / server 三类消费者共用，W8 零改写复用。
    """
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ids = sorted(range(len(bm25_scores)),
                      key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]

    q = encode_query(model, query)
    _, vec_idx = index.search(q, VECTOR_TOP_K)
    vec_ids = [int(i) for i in vec_idx[0] if i >= 0]

    fused = rrf_fuse(bm25_ids, vec_ids, top_k=top_k)
    results = []
    for cid in fused:
        m = dict(metas[cid])
        m["in_bm25"] = cid in bm25_ids
        m["in_vector"] = cid in vec_ids
        results.append(m)
    return results
