import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from multiprocessing import Pool

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文档长度
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 先按照固定长度划分chunk，后续再根据特殊token移动
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 预读缓冲区长度

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 预读一小段内容寻找特殊token

            # 边界条件：文档阅读到结尾时结束
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 找到特殊token，作为划分chunk的边界
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去重，保证边界不重复
    return sorted(set(chunk_boundaries))

def pre_tokenize(
    text: str,
    pattern: str = None
) -> dict[bytes, int]:
    """
    使用GPT2方法将文本预分词为word-level tokens，并统计每个token的出现频次。
    
    返回格式: {token_bytes: frequency_count}
    例如: {b'Hello': 5, b',': 3, ...}
    """
    if pattern is None:
        # GPT2预分词使用的模式
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_match = re.finditer(pattern, text)
    words = (match.group() for match in word_match)
    word_counter = Counter(words)
    word_dict = {}
    for word, cnt in word_counter.items():
        word_dict[word.encode('utf-8')] = cnt
    return word_dict

def process_chunk(
    args
) -> dict[bytes, int]:
    """
    一个chunk的worker，在子进程运行

    Args:
        (file_path:str, start:int, end:int, special_tokens:list[str])
    
    Returns:
        该chunk内全部pre-token的频次dict {bytes:count}
    """
    file_path, start, end, special_tokens = args
    # 加载text
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    chunk_text = chunk_bytes.decode('utf-8')
    # 处理特殊token，直接切分
    split_pat = "|".join([re.escape(token) for token in special_tokens])
    doc_texts = re.split(split_pat, chunk_text)
    # 每篇文档做预分词
    chunk_counter = Counter()
    for doc_text in doc_texts:
        if not doc_text.strip():  # 跳过空文档
            continue
        doc_dict = pre_tokenize(doc_text)
        chunk_counter.update(doc_dict)
    return dict(chunk_counter)
    
def build_initial_pair_counts(
    word_dict: dict[bytes, int]
) -> defaultdict[tuple[bytes, bytes], int]:
    """
    构建初始Token对频次字典
    Args:
        word_dict: 预分词结果，即词频统计字典

    Returns：
        初始Token对频次字典
    """
    pair_dict = defaultdict(int)
    for b_str, count in word_dict.items():
        for i in range(len(b_str) - 1):
            # 注意要转换为bytes
            pair_dict[(bytes([b_str[i]]), bytes([b_str[i + 1]]))] += count
    return pair_dict

def merge_and_update(
    word_dict: dict[bytes, tuple[list[bytes], int]], 
    best_pair: tuple[bytes, bytes], 
    pair_counts: defaultdict[tuple[bytes, bytes], int]
) -> dict[tuple[bytes, bytes], int]:
    """
    根据词频字典和Token对字典，以及当前的合并Token对，计算新Token对字典

    Args：
        word_dict: 词频字典
        best_pair: 当前合并的Token对
        pair_counts: 当前Token对字典
    
    Returns：
        新Token对字典
    """
    t1, t2 = best_pair
    t12 = t1 + t2
    # 遍历每个word的tokens，只寻找匹配的token对，分别处理前，后的计数
    del pair_counts[(t1, t2)]
    for word, (tokens, count) in word_dict.items():
        n = len(tokens)
        i = 0
        while i < n - 1:
            if t1 == tokens[i] and t2 == tokens[i + 1]:
                if i > 0:
                    pair_counts[(tokens[i - 1], tokens[i])] -= count
                    if pair_counts[(tokens[i - 1], tokens[i])] == 0:
                        del pair_counts[(tokens[i - 1], tokens[i])]
                    pair_counts[(tokens[i - 1], t12)] += count
                if i + 2 < n:
                    pair_counts[(tokens[i + 1], tokens[i + 2])] -= count
                    if pair_counts[(tokens[i + 1], tokens[i + 2])] == 0:
                        del pair_counts[(tokens[i + 1], tokens[i + 2])]
                    pair_counts[(t12, tokens[i + 2])] += count
                tokens[i:i+2] = [t12]
                n -= 1
            i += 1
    return pair_counts

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = None,
    num_processes: int = 4,
    pattern: str = None,
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。
    
    Args:
        input_path: 训练数据路径
        vocab_size: 目标 vocab 大小（包含特殊 token）
        special_tokens: 特殊 token 列表，永不参与合并
        pattern: 预分词正则模式，默认GPT-2预分词模式
        
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            - vocab: 词表 {token_id: token_bytes}
            - merges: 合并规则列表 [(a_bytes, b_bytes), ...]
    """
    # 划分边界，注意参数是bytes，即分隔符，不是所有special_tokens
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    chunks = []
    for i in range(0, len(boundaries) - 1):
        chunks.append((input_path, boundaries[i], boundaries[i + 1], special_tokens))
    # 并行预分词
    with Pool(num_processes) as pool:
        chunk_dicts = pool.map(process_chunk, chunks)
    # 聚合预分词结果
    total_counter = Counter()
    for chunk_dict in chunk_dicts:
        total_counter.update(chunk_dict)
    word_dict = dict(total_counter)
    # 将预分词结果增加tokens序列信息
    global_pre_tokens = {}
    for pre_token_bytes, freq in word_dict.items():
        # 将pre_token按字节拆成单字节bytes列表
        sub_tokens = [bytes([b]) for b in pre_token_bytes]
        global_pre_tokens[pre_token_bytes] = (sub_tokens, freq)
    # 构建初始pair-count计数
    pair_counts = build_initial_pair_counts(word_dict)
    # 初始化词表和合并规则
    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens is not None:
        for tok_str in special_tokens:
            tok_bytes = tok_str.encode("utf‑8")
            vocab[len(vocab)] = tok_bytes
    merges = []
    # 循环合并并更新计数
    while len(vocab) < vocab_size:
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        pair_counts = merge_and_update(global_pre_tokens, best_pair, pair_counts)
    return vocab, merges

if __name__ == '__main__':
    train_bpe