# RAG 学习项目

以"小而实用"为目标的 RAG（检索增强生成）学习与搭建项目。
W5-W8 建码时间盒 ≤20h：**数字优先于服务化，服务化优先于 Docker**。

## 📖 核心文档

- **[RAG搭建手册.md](./RAG搭建手册.md)** —— 定制版 10 章搭建手册（架构全景 / 解析 / 分块 / Embedding / 向量库 / 检索 / 生成 / 评估 / 进阶 / 部署），先读它的 §0 默认假设。

## ⚙️ 技术栈定案（2026-11-01 建码确认）

| 项 | 定案 | 备注 |
| --- | --- | --- |
| 语言 | Python 3.11+（本机 3.14.4） | 虚拟环境 `.venv` |
| Embedding | BAAI/bge-small-zh-v1.5（本地） | 中文效果好、免费、约 100MB |
| LLM | DeepSeek API（OpenAI 兼容） | Key 放 `.env`，不进仓库 |
| 向量库 | **FAISS（本地持久化）** | 建码定案，替代原"numpy→Chroma"默认 |
| 分块 | 标题感知优先，兜底 400/60 句界切分 | 手册第 3 章 |
| 检索 | BM25 + 向量双路召回 → RRF 融合 → Top-5 | 手册第 6 章 |
| 数据 | Markdown 起步 → PDF 第二阶段 | 元数据从第一天就记录 |

## 📁 项目结构

```text
RAG/
├── config.py            # 唯一配置源：路径/模型/检索参数，杜绝魔法数字
├── chunking.py          # 标题感知分块（W5 已建）
├── build_index.py       # CLI：语料 → chunks.jsonl（W5 已建，管道入口）
├── indexing.py          # BM25 + BGE/FAISS 本地持久化（W6 计划）
├── retrieval.py         # 双路召回 + RRF 融合（W7 计划）
├── generation.py        # DeepSeek 生成，Prompt 三要素（W7 计划）
├── evaluate.py          # Top-5 命中率评测（W7 计划，核心交付）
├── ask.py               # CLI 问答演示（W7 计划）
├── server.py            # FastAPI /health（W5 已建）+ /ask（W8 计划）
├── tests/               # 分块单测（W5 已建）；评测单测（W7）
├── data/corpus/         # 语料（funNLP README）
├── data/chunks.jsonl    # 分块产物（.gitignore，可重建）
├── eval/                # qa_30.jsonl 人工标注（W7）
└── index/               # 索引产物（.gitignore，可重建）
```

## 🚀 快速开始

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      & :: 填入 DEEPSEEK_API_KEY
.venv\Scripts\python.exe build_index.py
.venv\Scripts\python.exe -m uvicorn server:app --reload   & :: GET /health
```

## 📚 语料来源

- [funNLP](https://github.com/fighting41love/funNLP) 的 README（中文 NLP 资源目录），已存入 `data/corpus/`。
- 公开内容仅作本地学习与检索评测用途，版权归原作者；该仓库无明确开源许可证——**若本项目将来公开，语料正文应改为仅保留下载脚本与出处链接**。

## 🗺️ 学习路径（三步走）

1. **开窍（1 晚）**：[HF 博客手写极简 RAG](https://huggingface.co/blog/ngxson/make-your-own-rag)
2. **系统学（1-2 周）**：[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)（只做 Foundational 分类）
3. **深入原理（选做）**：[pguso/rag-from-scratch](https://github.com/pguso/rag-from-scratch)（全本地、Node.js）

## ✅ 里程碑（W5-W8，时间盒 ≤20h）

| 周 | 交付 | 状态 |
| --- | --- | --- |
| W5 | FastAPI 脚手架 + 数据处理管道入口 | ✅ 2026-11-01（funNLP 语料 320 块，median 403 字符；5 单测过；/health 200） |
| W6 | BM25 + BGE 双索引，FAISS 本地持久化 | ⬜ |
| W7 | 混合检索 + RRF + 30 条标注 + **Top-5 命中率数字** | ⬜ |
| W8 | FastAPI `/ask` 服务层收尾（超时砍 Docker 保 CLI） | ⬜ |
