"""Top-5 命中率评测（W7 核心交付）：三渠道对比 + 逐条明细，零 LLM API 成本。

用法：python evaluate.py
前置：indexing.py 已建索引；eval/qa_30.jsonl 已人工审定。
口径：Hit Rate@5 = Top-5 中存在标题含 expected_heading 的块（retrieval.is_hit）。
"""
from __future__ import annotations

import json
import sys
from datetime import date

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, FINAL_TOP_K, QA_PATH
from indexing import load
from retrieval import Engine, dual_recall, is_hit, rrf_fuse

CHANNELS = (("bm25", "BM25-only"), ("vector", "向量-only"), ("rrf", "RRF 混合"))


def precheck(qa: list[dict], metas: list[dict]) -> None:
    """标注预检：expected_heading 库内 0 匹配 -> 报错拒绝出数字；多匹配 -> 警告。"""
    headings = [m["heading"] for m in metas]
    bad, ambi = [], []
    for r in qa:
        hits = {h for h in headings if r["expected_heading"] in h}
        if not hits:
            bad.append(r["id"])
        elif len(hits) > 1:
            ambi.append(r["id"])
    if bad:
        print(f"[错误] 标注 id={bad} 的 expected_heading 在库内无匹配，拒绝评测。")
        sys.exit(1)
    if ambi:
        print(f"[警告] 标注 id={ambi} 的 expected_heading 残留歧义（判定放宽）。")


def evaluate(qa, engine) -> list[dict]:
    """逐条三渠道评测：BM25 / 向量 / RRF 各自 Top-5 做命中判定。"""
    rows = []
    for r in qa:
        bm25_top, vec_top = dual_recall(r["question"], engine.bm25,
                                        engine.index, engine.model)
        rrf_top = rrf_fuse(bm25_top, vec_top, top_k=FINAL_TOP_K)
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "expected": r["expected_heading"],
            "bm25": is_hit(r, bm25_top[:FINAL_TOP_K], engine.metas),
            "vector": is_hit(r, vec_top[:FINAL_TOP_K], engine.metas),
            "rrf": is_hit(r, rrf_top, engine.metas),
        })
    return rows


def main() -> int:
    if not QA_PATH.exists():
        print("[错误] 缺少 eval/qa_30.jsonl，先完成 30 条人工标注并审定。")
        return 1
    qa = [json.loads(line) for line in
          QA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    bm25, index, metas = load()
    precheck(qa, metas)
    model = SentenceTransformer(EMBEDDING_MODEL)
    rows = evaluate(qa, Engine(bm25, index, metas, model))
    n = len(rows)

    print(f"\n==== Top-5 命中率（Hit Rate@5，{n} 条人工标注，{date.today().isoformat()}）====")
    for key, label in CHANNELS:
        hits = sum(1 for r in rows if r[key])
        print(f"  {label:9s}: {hits}/{n} = {hits / n:.1%}")

    misses = [r for r in rows if not r["rrf"]]
    if misses:
        print(f"\n==== RRF 未命中明细（{len(misses)} 条）====")
        for r in misses:
            failed = [label for key, label in CHANNELS if not r[key]]
            print(f"  id={r['id']} 失败渠道={failed} 期望「{r['expected']}」")

    mark = lambda b: "✅" if b else "❌"
    lines = [
        "# 评测报告：Top-5 命中率（Hit Rate@5）",
        "",
        f"- 日期：{date.today().isoformat()}；标注 {n} 条（人工审定）；语料 funNLP README（320 块）",
        "- 口径：检索 Top-5 中存在标题含 expected_heading 的块（标题路径包含匹配）",
        "- 渠道：BM25-only / 向量-only / RRF 混合（RRF_K=60，两路各取 Top-20）",
        "",
        "| 渠道 | 命中 | 命中率 |",
        "| --- | --- | --- |",
    ]
    for key, label in CHANNELS:
        hits = sum(1 for r in rows if r[key])
        lines.append(f"| {label} | {hits}/{n} | {hits / n:.1%} |")
    lines += ["", "| id | 问题 | 期望小节 | BM25 | 向量 | RRF |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['question'][:30]} | {r['expected'][:20]} "
                     f"| {mark(r['bm25'])} | {mark(r['vector'])} | {mark(r['rrf'])} |")
    report = QA_PATH.parent / "eval_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[完成] 报告已写入 {report.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
