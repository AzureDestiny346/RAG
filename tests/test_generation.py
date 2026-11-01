"""生成层纯函数单测（W7）：上下文预算切分 / Prompt 三要素完整性。

LLM 调用路径不进单测（answer 需真实客户端），由 e2e 冒烟覆盖。
运行：.venv\\Scripts\\python.exe -m pytest tests -q
"""
from generation import build_context, build_prompt


def _chunk(text: str) -> dict:
    return {"chunk_id": 0, "source": "t.md", "heading": "H", "text": text}


def test_context_respects_budget():
    """两块 300 字符、预算 400：只装第一块（给足但不给多）。"""
    chunks = [_chunk("甲" * 300), _chunk("乙" * 300)]
    ctx = build_context(chunks, budget=400)
    assert "甲" * 10 in ctx
    assert "乙" not in ctx


def test_context_keeps_first_even_if_oversized():
    """第一块就超预算：仍保留，不输出空上下文。"""
    ctx = build_context([_chunk("甲" * 500)], budget=400)
    assert ctx == "甲" * 500


def test_context_empty_input():
    """无块 -> 空上下文（ask 层应在此之前拦截）。"""
    assert build_context([], budget=400) == ""


def test_prompt_contains_three_elements():
    """Prompt 三要素：角色 + 仅凭资料（含逃生门"未提及"）+ 资料与问题齐备。"""
    prompt = build_prompt("怎么找分词工具？", [_chunk("jieba 是分词工具。")])
    assert "问答助手" in prompt            # 角色
    assert "不要编造" in prompt            # 防幻觉约束
    assert "资料中未提及" in prompt        # 逃生门
    assert "jieba 是分词工具。" in prompt  # 资料
    assert "怎么找分词工具？" in prompt    # 问题
