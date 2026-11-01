# Spec: W7 混合检索 + RRF + 30 条标注评测（Top-5 命中率）

> 状态：**ready-for-agent**（无外部 issue tracker，以本文件头标记代替标签）
> 依据：handoff-20261101.md（W7 = 不可让步核心）、RAG搭建手册 第 6/8 章、2026-11-01 会话定案
> 定案补充：命中口径 = 标题路径匹配；评测输出 = BM25/向量/RRF 三路对比

## Problem Statement

索引已经建好（BM25 + BGE/FAISS 双路，W6 落地），但系统只有"两路各自 Top-N"的原始召回，没有融合、没有评测——改任何参数都不知道是变好还是变坏（手册第 8 章：不评估 = 盲飞）。简历 bullet 承诺的"双路召回 + RRF 融合 + 人工评测命中率"至今没有一个真实数字。

## Solution

一条最小评测链路：查询进来 → BM25 与向量两路各召回 Top-20 → RRF 融合去重出 Top-5 → 与 30 条人工标注的"答案所在小节标题"比对 → 输出三路（BM25-only / 向量-only / RRF）Top-5 命中率对比报告。数字可一键重跑、当场复现。

## User Stories

1. As a 求职者（项目作者）, I want 混合检索把 BM25 的字面命中和向量的语义命中融合成一个稳定排名, so that 专有名词和换种说法的问题都能被找到
2. As a 求职者, I want 一键重跑评测脚本就得到 Top-5 命中率数字, so that 简历 bullet 上的数字可当场复现、经得起面试官追问
3. As a 求职者, I want 评测报告同时给出 BM25-only、向量-only、RRF 三组命中率, so that "为什么要混合检索"有对照证据而不是口说
4. As a 求职者, I want 标注格式只要求写"答案所在小节的标题路径", so that 30 条标注的人工成本可控（不抄原文、不记块号）
5. As a 求职者, I want 标注里写错的标题被评测脚本预检拦下并报错, so that 不会带着坏标注测出一个静默错误的数字
6. As a 求职者, I want 分块参数迭代后 chunk_id 变化不影响标注有效性, so that 调分块策略不用重写标注
7. As a 评测脚本, I want RRF 融合是纯函数（排名进、排名出）, so that 不碰 IO 就能验证融合逻辑的确定性与平局稳定性
8. As a 评测脚本, I want 命中判定是纯函数（标注 + 检索结果 + 元数据 → 布尔）, so that 判定口径有唯一实现、可表驱动测试
9. As a 面试官, I want 看到每条标注在三个渠道下的命中明细, so that 我能抽查数字是不是编的
10. As a 面试官, I want 报告里指标命名沿用 RAGAS 口径（如 Hit Rate@5 / Context Recall 语境）, so that 术语与业界对齐、不自造词
11. As a 未来开发者, I want retrieval 的融合入口同时服务 CLI 问答与 HTTP 接口, so that W7 的检索逻辑 W8 直接复用、零改写
12. As a 未来开发者, I want 检索结果附带每块的渠道来源信息, so that 调试时能看出一块到底是 BM25 还是向量把它送进来的
13. As a 项目作者, I want 评测全程零 LLM API 调用, so that 出数字不花钱、不依赖 Key 是否配置
14. As a 项目作者, I want 所有真实数字（命中率/冷热启动耗时）不预填进任何对外材料, so that 诚信红线不破

## Implementation Decisions

- **三个纯函数 seam（已与作者确认）**：RRF 融合（排名列表进 → 融合排名出）、命中判定（标注项 + Top-5 块 + 元数据 → 布尔）、组合检索入口（查询 → 融合后 Top-K 块，内部调用融合函数；evaluate / ask / server 三个消费者共用这一个入口）
- **RRF 公式**：score(d) = Σ 1/(RRF_K + rank_i(d))；RRF_K = 60（论文惯用值，已入 config）；两路各取 Top-20（config 既有），融合去重后取 Top-5（FINAL_TOP_K）
- **命中口径（作者定案）**：检索块的标题路径**包含**标注的 expected_heading 子串即命中；不做 chunk_id 精确匹配（分块一改标注全废）
- **标注预检**：评测前逐条校验 expected_heading 至少匹配一个库内块标题，匹配数为 0 的标注行报错并拒绝出数字（防静默错标注）
- **qa_30.jsonl schema**（决策主体，合成示例值）：`{"id": 1, "question": "想找中文分词工具", "expected_heading": "分词工具", "answer_contains": "jieba（可选，生成层评测备用）"}`
- **评测输出**：三渠道（BM25-only / 向量-only / RRF）各一个 Hit Rate@5 + 每条标注的逐渠道命中明细表；指标名沿用 RESOURCES.md 的 RAGAS 命名
- **标注工作流**：AI 从元数据的标题分布起草 30 条候选（覆盖各主要章节）→ 作者人工审定后入库；标注入库前评测脚本拒绝运行
- **依赖倒置沿用**：检索层只依赖 config 常量与既有双索引加载入口，不感知 W8 的 HTTP 层

## Testing Decisions

- 只测外部行为：融合函数用已知排名断言确定性输出（含并列、单路为空、去重）；命中判定用合成元数据表驱动（命中 / 未命中 / 同标题多块 / 标注标题无匹配）
- 模型相关路径（编码、真实索引）不进单测，沿用 W6 先例由集成冒烟覆盖
- 先例：tests/ 下既有纯函数 + tmp_path 模式（W5/W6 九个测试即此风格）
- 评测脚本本身以"预检能拦坏标注 + 三渠道数字可比"作为验收演示，不为其造 mock

## Out of Scope

- Reranker（bge-reranker-base 精排）——手册第 9 章：评测确认瓶颈后才上
- HTML/URL 行降噪——挂起的 W7 首处方候选，出数字后由数据裁决
- PDF 语料、语义分块、父子分块、多跳查询
- RAGAS 全量指标（Faithfulness 等生成层指标）与 LLM 生成质量评测
- Docker 化

## Further Notes

- 红线：Top-5 命中率只测检索层、零 API 成本；30 条标注真实测出，**任何对外材料不预填数字**；interview-guide 的 68%→89% STAR 与本项目无关，禁止混用叙事
- 本 spec 的标注工作流把作者（用户）作为人工审定瓶颈，AI 起草只是候选，不代签
