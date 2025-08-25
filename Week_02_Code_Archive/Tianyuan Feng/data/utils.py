import re, random
from typing import List, Tuple
import numpy as np
import torch

# 这行代码定义了一个正则表达式，用于后续的分词。
# re.compile() 会将这个正则表达式模式预编译成一个对象，后续使用时效率更高。
# 模式解析：
#   - `[A-Za-z0-9]+`：匹配一个或多个连续的字母或数字（代表一个单词）。
#   - `|`：逻辑“或”操作。
#   - `[^\sA-Za-z0-9]`：匹配任何不是（^）空白符(\s)、字母、数字的单个字符（代表一个标点符号）。
_tok_re = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

def simple_tokenize(s: str) -> List[str]:
    # 1. `s.lower()`: 将输入字符串全部转换为小写，以确保 "Hello" 和 "hello" 被视为同一个词。
    # 2. `_tok_re.findall(...)`: 使用我们上面定义的正则表达式模式，在小写字符串中查找所有匹配项，并作为一个列表返回。
    return _tok_re.findall(s.lower())
# set the fixed seed in Python, NumPy, Torch
def set_seed(seed: int = 42):
    random.seed(seed); 
    np.random.seed(seed)
    torch.manual_seed(seed); 
    torch.cuda.manual_seed_all(seed)

def encode(text: str, vocab, max_len: int) -> Tuple[list, int]:
    # 1. 调用上面的分词函数，将文本字符串处理成单词列表。
    toks = simple_tokenize(text)
    # 2. 将单词列表转换为ID列表。这是一个列表推导式，具体操作为：
    #    a) `toks[:max_len]`: 首先对单词列表进行截断，确保不超过最大长度。
    #    b) `for t in ...`: 遍历截断后的每个单词 `t`。
    #    c) `vocab.stoi.get(t, vocab.unk)`: 在词汇表的 stoi 字典中查找单词 `t`。
    #       - 如果找到了，就返回它的ID。
    #       - 如果没找到（未登录词），就返回特殊标记 `<unk>` 的ID。
    ids = [vocab.stoi.get(t, vocab.unk) for t in toks[:max_len]]
    # 3. 记录在填充之前的真实句子长度。
    L = len(ids)
    # 4. 进行填充 (Padding)。
    #    - 如果真实长度小于设定的最大长度...
    if L < max_len:
        # ...就在ID列表的末尾，补上 (max_len - L) 个填充符 `<pad>` 的ID。
        ids += [vocab.pad] * (max_len - L)
    # 5. 返回处理完成的、长度统一的ID列表和句子的真实长度。
    return ids, L
