"""
BPE（Byte Pair Encoding）分词器
================================

算法本身很简单：
    1. 把每个词拆成字符序列
    2. 统计所有相邻字符对的出现频率
    3. 合并频率最高的一对，记一条合并规则
    4. 重复 2~3 直到达到目标词表大小

编码时按**合并规则的顺序**依次应用；解码就是把所有 token 直接拼回去。

面试考点：
  1. BPE / WordPiece / Unigram 的区别
     → BPE 按频率合并；WordPiece 按似然增益（合并后语言模型概率最大）；
       Unigram 从大词表开始按似然删除，方向相反
  2. 为什么大模型几乎都用 BPE 系（SentencePiece 实现的 BPE）
     → 无需预分词、语言无关、能处理任何字节序列（配合 byte-level 可完全避免 OOV）
  3. 词表大小怎么权衡
     → 词表大：序列短、推理快，但 embedding 参数多、低频 token 训练不充分；
       词表小：参数省、泛化好，但序列变长、计算变多

常见追问：
  - "byte-level BPE 解决什么？" → 先转 UTF-8 字节，任何字符（含 emoji、中文）
    都能表示，彻底消除 UNK；代价是一个汉字可能占 2~3 个 token
  - "中文分词有什么特殊问题？" → 汉字信息密度高，中文 tokenizer 词表配比
    影响很大，中英混合场景常有"中文比英文贵"的现象
  - "压缩率怎么算？" → 字节数 / token 数，GPT-4 约 4 字节/token

运行：python code/tokenizer/bpe.py
"""

import re
from collections import Counter


def get_vocab(corpus, counts=None):
    """
    把语料切成词，每个词拆成字符序列（用空格分隔表示）。

    Returns:
        {'l o w </w>': freq, ...}
    """
    counts = counts or Counter(corpus.split())
    vocab = {}
    for word, freq in counts.items():
        vocab[" ".join(word) + " </w>"] = freq
    return vocab


def get_stats(vocab):
    """统计所有相邻符号对的频次。"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab):
    """把 vocab 中所有出现 pair 的地方合并成一个符号。"""
    merged = {}
    bigram = re.escape(" ".join(pair))
    # 用反向断言保证只匹配完整符号，避免把 "l o w" 里的 "o w" 和单词 "ow" 混淆
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word, freq in vocab.items():
        merged[pattern.sub("".join(pair), word)] = freq
    return merged


def train_bpe(corpus, num_merges):
    """训练 BPE，返回合并规则列表。"""
    vocab = get_vocab(corpus)
    merges = []
    for _ in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        best = max(stats, key=stats.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges, vocab


def encode(word, merges):
    """按合并规则顺序对单个词编码。"""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def decode(tokens):
    """解码：直接拼接，去掉词尾标记。"""
    return "".join(tokens).replace("</w>", "")


def _test():
    corpus = (
        "low lower lowest newer newest new wide wider widest "
        "low low newer new new new new wide"
    )

    merges, final_vocab = train_bpe(corpus, num_merges=10)
    print(f"训练得到 {len(merges)} 条合并规则:")
    for i, (a, b) in enumerate(merges, 1):
        print(f"  {i:2d}. {a:>6} + {b:<6} -> {a+b}")

    # --- 常用词的 token 数应当下降 ---
    base_tokens = len(list("low")) + 1          # l o w </w>
    final_tokens = len(encode("low", merges))
    print(f"\n'low' 的 token 数: {base_tokens} -> {final_tokens}")
    assert final_tokens < base_tokens, "BPE 应当减少高频词的 token 数"

    # --- 编解码往返 ---
    for word in ["low", "lowest", "newer", "wide"]:
        toks = encode(word, merges)
        assert decode(toks) == word, f"{word} 往返失败: {toks}"
        print(f"  {word:8s} -> {toks}")

    # --- 训练前后的总 token 数对比 ---
    counts = Counter(corpus.split())
    before = sum((len(w) + 1) * f for w, f in counts.items())
    after = sum(len(encode(w, merges)) * f for w, f in counts.items())
    print(f"\n语料总 token 数: {before} -> {after} "
          f"（压缩率 {after/before:.1%}）")
    assert after < before

    # --- 未登录词也能编码（只要有字符级 fallback）---
    unknown = "lowestt"
    assert decode(encode(unknown, merges)) == unknown
    print(f"未登录词 '{unknown}' 也能编码: {encode(unknown, merges)}")

    print("\nbpe.py 全部通过")


if __name__ == "__main__":
    _test()
