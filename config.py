"""项目唯一配置源：所有路径、模型名、检索参数集中于此，杜绝魔法数字。

定案依据：RAG搭建手册 §0 + 交接单（2026-11-01）。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根 = 本文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# .env 承载 API Key 等敏感信息，绝不硬编码（手册第 10 章"两个必须养成的习惯"）
load_dotenv(BASE_DIR / ".env")

# ---------- 路径 ----------
CORPUS_DIR = BASE_DIR / "data" / "corpus"          # 语料目录（funNLP README 入库）
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"   # 分块产物（W5 管道输出）
INDEX_DIR = BASE_DIR / "index"                     # 索引产物目录（.gitignore 已排除）
QA_PATH = BASE_DIR / "eval" / "qa_30.jsonl"        # 30 条人工标注（W7）

# 索引产物三件套（W6，.gitignore 已排除，可随时重建）
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "vector.faiss"
META_PATH = INDEX_DIR / "meta.jsonl"

# ---------- 分块（手册第 3 章定案）----------
CHUNK_SIZE = 400       # 兜底分块的目标长度（字符）
CHUNK_OVERLAP = 60     # 相邻块重叠字符数

# ---------- Embedding（手册第 4 章定案）----------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地模型，首跑自动下载约 100MB
EMBEDDING_DIM = 512                          # 该模型输出维度
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # bge-zh 查询侧前缀（模型卡要求，文档侧不加）

# ---------- 检索（手册第 6 章定案：混合检索 + RRF）----------
BM25_TOP_K = 20     # BM25 召回条数
VECTOR_TOP_K = 20   # 向量召回条数
RRF_K = 60          # RRF 常数，论文惯用值
FINAL_TOP_K = 5     # 最终参与命中率判定 / 喂给 LLM 的条数

# ---------- LLM（OpenAI 兼容协议，依赖倒置；支持 DeepSeek / DashScope 通义 / 任意兼容端点）----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

if DEEPSEEK_API_KEY:
    LLM_API_KEY = DEEPSEEK_API_KEY
    _DEFAULT_URL, _DEFAULT_MODEL = "https://api.deepseek.com", "deepseek-chat"
elif DASHSCOPE_API_KEY:
    LLM_API_KEY = DASHSCOPE_API_KEY
    _DEFAULT_URL = os.getenv("DASHSCOPE_BASE_URL",
                             "https://dashscope.aliyuncs.com/compatible-mode/v1")
    _DEFAULT_MODEL = "qwen-plus"
else:
    LLM_API_KEY = ""
    _DEFAULT_URL, _DEFAULT_MODEL = "https://api.deepseek.com", "deepseek-chat"

LLM_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", _DEFAULT_URL)
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODEL)
LLM_TEMPERATURE = 0.2
MAX_CONTEXT_CHARS = 4000   # 上下文预算：按分数塞到预算为止，"给足但不给多"


def ensure_dirs() -> None:
    """确保运行期需要的目录存在（幂等，W5 管道与 W6 索引共用）。"""
    for d in (CORPUS_DIR, INDEX_DIR, CHUNKS_PATH.parent, QA_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
