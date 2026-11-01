"""索引单测（W6，模型无关）：分词确定性 / BM25 往返 / FAISS 往返。

模型编码路径（需下载 100MB 模型）不进单测，由 integration 冒烟覆盖。
运行：.venv\\Scripts\\python.exe -m pytest tests -q
"""
import pickle

import faiss
import numpy as np

from indexing import build_bm25
from retrieval import tokenize


def test_tokenize_chinese_deterministic():
    """中文分词非空、含关键粒度、结果确定。"""
    toks = tokenize("中文分词工具有哪些？")
    assert toks and any("分词" in t for t in toks)
    assert tokenize("中文分词工具有哪些？") == toks


def test_bm25_build_and_rank():
    """BM25 能把关键词所在块排到第一（3 文档保证稀有词 idf>0）。"""
    corpus = [
        {"chunk_id": 0, "text": "jieba 是中文分词工具"},
        {"chunk_id": 1, "text": "faiss 是向量检索库"},
        {"chunk_id": 2, "text": "bm25 是关键词打分算法"},
    ]
    scores = build_bm25(corpus).get_scores(tokenize("中文分词"))
    assert scores.argmax() == 0


def test_bm25_pickle_roundtrip(tmp_path):
    """BM25 pickle 序列化往返后排序行为一致。

    注意：查询词必须只出现在少数文档——出现在 >=半数文档时
    rank_bm25 的 idf 为 0/负，分数全零，排序无意义（小白经典坑）。
    """
    corpus = [{"chunk_id": i, "text": t}
              for i, t in enumerate(["分词工具 jieba 很流行",
                                     "向量库 faiss 做检索",
                                     "bm25 是打分算法"])]
    p = tmp_path / "bm25.pkl"
    p.write_bytes(pickle.dumps(build_bm25(corpus)))
    bm25b = pickle.loads(p.read_bytes())
    assert bm25b.get_scores(tokenize("faiss")).argmax() == 1


def test_faiss_serialize_roundtrip():
    """FAISS 序列化往返后 Top-1 命中自身。

    用内存序列化而非 write_index：faiss 在 Windows 上打不开
    含中文的路径（pytest 临时目录挂在中文用户名下）；生产的
    write_index 走 ASCII 路径，由集成构建覆盖。
    """
    vecs = np.random.RandomState(0).rand(10, 8).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # 与生产一致：归一化后 IP=余弦
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    loaded = faiss.deserialize_index(faiss.serialize_index(index))
    dist, idx = loaded.search(vecs[:1], 1)
    assert idx[0][0] == 0
    assert dist[0][0] > 0.999
