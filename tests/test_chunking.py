"""分块单测（W5）：标题切分 / 兜底断句 / overlap / 元数据完整性。

运行：.venv\\Scripts\\python.exe -m pytest tests/ -q
"""
from chunking import chunk_markdown, split_long_text
from config import CHUNK_OVERLAP, CHUNK_SIZE


def test_heading_split_metadata():
    """标题层级 -> heading 路径正确，chunk_id 连续，块文本带标题前缀。"""
    md = "# A\n正文一。\n## B\n正文二。\n# C\n正文三。\n"
    chunks = chunk_markdown(md, source="t.md")
    assert [c.heading for c in chunks] == ["A", "A > B", "C"]
    assert [c.chunk_id for c in chunks] == [0, 1, 2]
    assert all(c.source == "t.md" for c in chunks)
    assert chunks[0].text.startswith("A\n")


def test_fallback_sentence_boundary():
    """超长文本：每块 <= CHUNK_SIZE，中间块断在句末标点，相邻块重叠。"""
    text = "这是一句测试。" * 200  # 1400 字符
    pieces = split_long_text(text)
    assert len(pieces) >= 2
    assert all(len(p) <= CHUNK_SIZE for p in pieces)
    assert all(p.endswith("。") for p in pieces[:-1])
    assert pieces[1].startswith(pieces[0][-CHUNK_OVERLAP:])


def test_fallback_hard_cut_without_punctuation():
    """整段无句末标点：硬切兜底，末块保留结尾，无空块。"""
    text = "字" * (CHUNK_SIZE * 2 + 100)
    pieces = split_long_text(text)
    assert all(pieces)
    assert all(len(p) <= CHUNK_SIZE for p in pieces)
    assert pieces[0] == text[:CHUNK_SIZE]
    assert pieces[-1].endswith(text[-100:])
    assert pieces[1].startswith(pieces[0][-CHUNK_OVERLAP:])


def test_short_text_single_chunk():
    """短文本不切分，原样成块。"""
    assert split_long_text("短句。") == ["短句。"]


def test_preface_not_lost():
    """第一个标题前的前言不丢（防静默数据丢失）。"""
    chunks = chunk_markdown("前言内容。\n# A\n正文。", source="t.md")
    assert chunks[0].heading == "（前言）"
    assert "前言内容" in chunks[0].text
