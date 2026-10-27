import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from my_bpe_train import gpt2_bytes_to_unicode
import json
import regex as re
from typing import Iterable, Iterator
import time
import os
import numpy as np

def gpt2_unicode_to_bytes() -> dict[str, bytes]:
    """unicode字符 → bytes对象，GPT2映射反向"""
    b2u = gpt2_bytes_to_unicode()
    u2b = {}
    for byte_int, uni_char in b2u.items():
        u2b[uni_char] = bytes([byte_int])
    return u2b

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # 反向字典，用于encode阶段
        self.bytes_to_id = {b: idx for idx, b in vocab.items()}
        # 优化：merge的优先级字典，加速编码过程
        self.merge_rank = {}
        for idx, (a,b) in enumerate(self.merges):
            self.merge_rank[(a,b)] = idx
        # special tokens如果不在vocab里，追加进vocab
        if special_tokens is not None:
            for token in special_tokens:
                # 注意类型转换
                tok_bytes = token.encode("utf-8")
                if tok_bytes not in self.bytes_to_id:
                    self.vocab[len(self.vocab)] = tok_bytes
                    self.bytes_to_id[tok_bytes] = len(self.bytes_to_id)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """Construct and return a Tokenizer from a serialized vocabulary and
        list of merges and (optionally) a list of special tokens.
        Args:
            vocab_filepath: Path to the serialized vocabulary file.
            merges_filepath: Path to the serialized merges file.
            special_tokens: Optional list of special tokens to register.
        Returns:
            A Tokenizer instance built from the given files.
        """
        u2b = gpt2_unicode_to_bytes()

        def _to_bytes(s: str) -> bytes:
            buf = b""
            for ch in s:
                if ch in u2b:
                    buf += u2b[ch]
                else:
                    buf += ch.encode("utf-8")
            return buf

        # 加载vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            json_vocab = json.load(f)
        vocab = {}
        for uni_subword, idx in json_vocab.items():
            token_id = int(idx)
            token_bytes = _to_bytes(uni_subword)
            vocab[token_id] = token_bytes

        # 加载merges
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s1, s2 = line.split()
                b1 = _to_bytes(s1)
                b2 = _to_bytes(s2)
                merges.append((b1, b2))
        return cls(vocab, merges, special_tokens)

    def pre_tokenize_encode(self, text: str, pattern=None) -> list[tuple[str, bool]]:
        """
        返回列表，每个元素：(文本片段, is_special)
        is_special=True：这一段是特殊token，后续不跑BPE合并
        is_special=False：普通文本，需要GPT2预分词+BPE合并
        """
        if pattern is None:
            pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if not self.special_tokens:
            matches = re.finditer(pattern, text)
            return [(m.group(), False) for m in matches]
        # 注意特殊tokens需要
        sorted_specials = sorted(self.special_tokens, key=lambda s: len(s), reverse=True)
        spl_pat = "|".join(re.escape(tok) for tok in sorted_specials)
        parts = re.split(f"({spl_pat})", text)
        result = []
        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                # 是特殊token
                result.append((part, True))
            else:
                # 普通文本，GPT2预分词
                for m in re.finditer(pattern, part):
                    result.append((m.group(), False))
        return result

    def encode_raw(self, text: str) -> list[int]:
        """
        encode的朴素实现，每次遍历merges，token速度为310 ~ 315 token/s
        """
        token_ids = []
        # 预分词
        word_list = self.pre_tokenize_encode(text)
        # 执行合并
        for word, is_special in word_list:
            if not is_special:
                word_bytes = [bytes([ch]) for ch in word.encode("utf-8")]
                for a, b in self.merges:
                    ab = a + b
                    i = 0
                    n = len(word_bytes)
                    while i < n - 1:
                        if word_bytes[i] == a and word_bytes[i + 1] == b:
                            # 注意右侧必须是列表
                            word_bytes[i: i + 2] = [ab]
                            n -= 1
                        i += 1
                # 根据反向vocab得到idx
                for token in word_bytes:
                    token_ids.append(self.bytes_to_id[token])
            else:
                token_ids.append(self.bytes_to_id[word.encode('utf-8')])
        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        encode的优化实现，token速度约为100000 token/s
        """
        token_ids = []
        # 预分词
        word_list = self.pre_tokenize_encode(text)
        for word, is_special in word_list:
            if not is_special:
                word_bytes = [bytes([ch]) for ch in word.encode("utf-8")]
                while True:
                    # 收集候选pair
                    pairs = []
                    for i in range(len(word_bytes) - 1):
                        p = (word_bytes[i], word_bytes[i + 1])
                        if p in self.merge_rank:
                            pairs.append(p)
                    if not pairs:
                        break
                    # 确定merge
                    best_pair = min(pairs, key=lambda x: self.merge_rank[x])
                    # 执行merge
                    a, b = best_pair
                    ab = a + b
                    new_word_bytes = []
                    i = 0
                    while i < len(word_bytes):
                        if i + 1 < len(word_bytes) and word_bytes[i] == a and word_bytes[i + 1] == b:
                            new_word_bytes.append(ab)
                            i += 2
                        else:
                            new_word_bytes.append(word_bytes[i])
                            i += 1
                    word_bytes = new_word_bytes
                for token in word_bytes:
                    token_ids.append(self.bytes_to_id[token])
            else:
                token_ids.append(self.bytes_to_id[word.encode('utf-8')])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        标准流式编码：按完整行/完整文档迭代，整段编码。
        保证与 encode() 结果100%一致，无截断误差。
        iterable: 逐行/逐段产出文本的迭代器
        """
        cnt = 0
        start_total = time.time()
        for segment in iterable:
            # 注意使用strip，否则不会跳过空行
            if segment.strip() == "":
                continue
            # 一整段完整编码，预分词、BPE都在完整文本上执行
            for tid in self.encode(segment):
                yield tid
                cnt += 1
                if cnt % 2000 == 0:
                    elapsed = time.time() - start_total
                    print(f"[STREAM] processed {cnt:>8d} tokens | speed={cnt/elapsed:.1f} tok/s")
    
    def decode(self, ids: list[int]) -> str:
        # 把id列表解码为token序列
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")

if __name__ == '__main__':
    VOCAB_PATH = '/home/wsd/diy-llm/cs336-notes-and-hw/hw/hw1/cs336_basics/outputs/bpe_train/vocab.json'
    MERGES_PATH = '/home/wsd/diy-llm/cs336-notes-and-hw/hw/hw1/cs336_basics/outputs/bpe_train/merges.json'
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
    test_line_1 = "Once upon a time there was a cat.\n"
    test_line_2 = "\n"
    print("普通行编码：", tokenizer.encode(test_line_1))
    print("纯换行编码：", tokenizer.encode(test_line_2))

    txt_path = "/home/wsd/diy-llm/cs336-notes-and-hw/hw/hw1/cs336_basics/dataset/TinyStories-train.txt"
    out_dir = "dataset/"
    out_bin_path = os.path.join(out_dir, "train.bin")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Phase1] Counting total tokens, reading: {txt_path}")
    def line_iter():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
    
    print("Streaming tokenize ...")
    token_ids = list(tokenizer.encode_iterable(line_iter()))
    total_tokens = len(token_ids)
    print(f"Total tokens: {total_tokens}")

    # 写入 memmap uint32
    arr = np.memmap(out_bin_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))
    arr[:] = np.array(token_ids, dtype=np.uint32)
    arr.flush()
    del arr
    print(f"✅ Binary saved -> {out_bin_path}")