"""W7 检索与评测核心单测（TDD RED 先行）：RRF 融合 / 命中判定。

只测外部行为（纯函数 seam），模型与真实索引不进单测。
运行：.venv\\Scripts\\python.exe -m pytest tests -q
"""
from retrieval import is_hit, rrf_fuse


# ---------- rrf_fuse：RRF 融合 ----------

def test_rrf_known_ranking():
    """融合结果确定（同输入同输出），平局有稳定顺序。"""
    fused = rrf_fuse([10, 20, 30], [20, 10, 40])
    assert fused == rrf_fuse([10, 20, 30], [20, 10, 40])
    assert set(fused) == {10, 20, 30, 40}


def test_rrf_single_agreement_wins():
    """两路都把某块排第一，它必是融合第一。"""
    assert rrf_fuse([5, 1, 2], [5, 3, 4])[0] == 5


def test_rrf_dedup_and_union():
    """结果去重，且覆盖两路并集，不丢块。"""
    fused = rrf_fuse([1, 2, 3], [3, 4])
    assert len(fused) == len(set(fused)) == 4
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_empty_one_channel():
    """单路为空时退化为另一路的顺序。"""
    assert rrf_fuse([], [7, 8]) == [7, 8]
    assert rrf_fuse([7, 8], []) == [7, 8]
    assert rrf_fuse([], []) == []


def test_rrf_respects_top_k():
    """融合后截断到指定 k。"""
    assert len(rrf_fuse([1, 2, 3, 4], [5, 6, 7, 8], top_k=5)) == 5


# ---------- is_hit：命中判定（标题路径包含口径） ----------

METAS = [
    {"chunk_id": 0, "heading": "类ChatGPT的资料"},
    {"chunk_id": 1, "heading": "第 6 章 检索策略 > 混合检索"},
    {"chunk_id": 2, "heading": "第 6 章 检索策略 > 重排序"},
]


def test_hit_by_substring():
    """块的标题路径包含标注标题即命中。"""
    assert is_hit({"id": 1, "expected_heading": "混合检索"}, [1, 2], METAS) is True


def test_hit_by_full_path():
    """标注写完整路径也能命中。"""
    qa = {"id": 2, "expected_heading": "第 6 章 检索策略 > 重排序"}
    assert is_hit(qa, [2], METAS) is True


def test_miss_when_not_in_top5():
    """期望标题不在检索结果中 -> 未命中。"""
    qa = {"id": 3, "expected_heading": "分词工具"}
    assert is_hit(qa, [0, 1, 2], METAS) is False


def test_hit_scans_all_top5():
    """期望块不在第一位但在 Top-5 内 -> 仍命中（Hit@5 语义）。"""
    qa = {"id": 4, "expected_heading": "重排序"}
    assert is_hit(qa, [0, 1, 2], METAS) is True
