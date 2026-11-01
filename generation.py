"""生成层（手册第 7 章定案）：Prompt 三要素 + 上下文预算 + OpenAI 兼容客户端。

三要素：角色（稳定风格）、仅凭资料（把"猜"约束为"读"）、允许说不知道（防幻觉逃生门）。
依赖倒置：业务只依赖 OpenAI 兼容协议，换 Ollama 只改 base_url（.env 可覆盖）。
"""
from __future__ import annotations

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
)

PROMPT_TEMPLATE = """你是严谨的中文资料问答助手。
仅凭下面提供的资料回答问题；资料中没有的内容，直接回答"资料中未提及"，不要编造。

【资料】
{context}

【问题】
{question}
"""


def build_context(chunks: list[dict], budget: int = MAX_CONTEXT_CHARS) -> str:
    """按检索顺序拼接块文本，塞到字符预算为止（给足但不给多，低分块丢弃）。

    至少保留第一块：哪怕超预算也不输出空上下文。
    """
    parts: list[str] = []
    used = 0
    for c in chunks:
        if parts and used + len(c["text"]) > budget:
            break
        parts.append(c["text"])
        used += len(c["text"])
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """组装 Prompt：模板三要素 + 预算内资料 + 问题。"""
    return PROMPT_TEMPLATE.format(
        context=build_context(chunks), question=question)


def make_client(api_key: str | None = None, base_url: str | None = None):
    """OpenAI 兼容客户端工厂：Key 缺失时给出可操作报错（而不是神秘 401）。"""
    from openai import OpenAI

    key = api_key or LLM_API_KEY
    if not key:
        raise RuntimeError(
            "缺少 LLM Key：请在 .env 填入 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY。")
    return OpenAI(api_key=key, base_url=base_url or LLM_BASE_URL)


def answer(client, question: str, chunks: list[dict]) -> str:
    """端到端问答：检索块 -> Prompt -> LLM -> 答案文本。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    return resp.choices[0].message.content
