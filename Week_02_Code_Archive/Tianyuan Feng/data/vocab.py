import os, json
from typing import Dict, List
from collections import Counter
from .utils import simple_tokenize

class Vocab:
    def __init__(self, stoi: Dict[str,int], itos: List[str], pad: int, unk: int):
        # String-to-Integer: 一个从单词(string)到ID(integer)的映射字典。
        self.stoi = stoi; 
        # Integer-to-String: 一个从ID(integer)到单词(string)的映射列表。
        self.itos = itos
        # 存储填充符 <pad> 的ID。
        self.pad = pad; 
        # 存储未知词 <unk> 的ID。
        self.unk = unk


# traversal all the texts, split and count the tokens
# <pad>: Used for padding to make sentences of different lengths uniform in a batch.
# <unk>: Represents an unknown token. When a word not in the vocabulary is encountered during training or inference, <unk> is used instead.
    @classmethod
    def build(cls, texts: List[str], max_size=30000, min_freq=2, specials=("<pad>", "<unk>")):
        # 1. 初始化一个计数器(Counter)对象，用于统计词频。
        cnt = Counter()
        # 2. 遍历所有输入的文本(texts)。
        for t in texts:
            #   - 对每段文本使用simple_tokenize进行分词。
            #   - .update()方法将分词后的单词列表喂给计数器，进行词频累加。
            cnt.update(simple_tokenize(t))
        # 3. 初始化 ID-到-单词 的列表(itos)，首先放入特殊符号("<pad>", "<unk>")。
        #    - 这些特殊符号有重要的功能，必须包含在词汇表中。
        itos = list(specials)
        # 4. .most_common()会返回一个按频率从高到低排序的(单词, 词频)列表。
        #    我们遍历这个列表，筛选出符合条件的单词加入词汇表。
        for tok, c in cnt.most_common():
            #   a) 如果单词的词频(c)低于我们设定的最小频率(min_freq)，就停止添加。
            #      因为后面的词频只会更低。
            if c < min_freq: break
            #   b) 如果这个单词本身就是我们已经添加过的特殊符号，就跳过。
            if tok in specials: continue
            #   c) 如果词汇表的当前大小已经达到上限(max_size)，也停止添加。
            if len(itos) >= max_size: break
            #   d) 如果通过了所有筛选，就将这个单词加入itos列表。
            itos.append(tok)
        # 5. 根据itos列表（ID是其索引），创建一个反向的 单词-到-ID 的字典(stoi)。
        stoi = {s:i for i,s in enumerate(itos)}
        # 6. 使用最终生成的itos, stoi以及特殊符号的ID，创建并返回一个Vocab类的实例。
        return cls(stoi, itos, pad=stoi[specials[0]], unk=stoi[specials[1]])

# write into the {itos, pad, unk} for repeatable use
    def to_json(self, path: str):
        # 1. 确保要保存文件的目录存在，如果不存在就创建它。
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 2. 以写入模式("w")和utf-8编码打开文件。
        with open(path, "w", encoding="utf-8") as f:
            # 3. 使用json.dump将关键信息写入文件。
            #    - 我们只需要保存itos列表和特殊符号的ID即可。
            #    - stoi字典可以随时通过itos列表重新构建，无需保存。
            #    - ensure_ascii=False 确保非英文字符能被正确保存。
            json.dump({"itos": self.itos, "pad": self.pad, "unk": self.unk}, f, ensure_ascii=False)
