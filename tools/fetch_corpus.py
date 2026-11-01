"""语料获取脚本：下载 funNLP README 到 data/corpus/（克隆后先跑这个再建索引）。

用法（项目根目录下）：python tools/fetch_corpus.py
出处：https://github.com/fighting41love/funNLP （版权归原作者，仅本地学习与检索评测用）
说明：语料正文不入 git（该仓库无明确开源许可证），克隆者用本脚本自取。
脚本自包含（纯标准库），作为克隆后的第一步不依赖任何项目模块。
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CORPUS_URL = "https://raw.githubusercontent.com/fighting41love/funNLP/master/README.md"
DEST = Path(__file__).resolve().parent.parent / "data" / "corpus" / "funNLP-README.md"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载语料: {CORPUS_URL}")
    with urllib.request.urlopen(CORPUS_URL, timeout=60) as resp:
        data = resp.read()
    if len(data) < 10000:
        print(f"[错误] 下载体过小（{len(data)} 字节），疑似错误页，请检查网络/代理。")
        return 1
    DEST.write_bytes(data)
    print(f"[完成] {len(data)} 字节 -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
