from typing import Iterable, Dict, List
import re

import torch
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize

try:
	# 确保 punkt 可用
	nltk.data.find('tokenizers/punkt')
except LookupError:
	try:
		nltk.download('punkt', quiet=True)
	except Exception:
		pass


def nltk_tokenize(text: str) -> List[str]:
	"""使用NLTK进行分词"""
	return word_tokenize(str(text))


class SimpleVocab:
	"""简单词汇表类，用于token到index的映射"""
	def __init__(self, token_to_index: Dict[str, int]):
		self.token_to_index = token_to_index
		self._default_index = 0

	def set_default_index(self, idx: int) -> None:
		"""设置默认索引（用于未知token）"""
		self._default_index = int(idx)

	def __getitem__(self, token: str) -> int:
		"""获取token对应的索引"""
		return self.token_to_index.get(token, self._default_index)

	def __len__(self) -> int:
		"""返回词汇表大小"""
		return len(self.token_to_index)


class TextToIds:
	"""可序列化的 tokenizer+vocab 映射器，避免闭包导致的多进程 pickling 问题。"""
	def __init__(self, tokenizer_func, vocab: SimpleVocab, max_len: int):
		self.tokenizer_func = tokenizer_func
		self.vocab = vocab
		self.max_len = int(max_len)

	def __call__(self, text: str) -> torch.Tensor:
		"""将文本转换为ID序列"""
		tokens = [t.lower() for t in self.tokenizer_func(text)]
		ids = [self.vocab[t] for t in tokens][: self.max_len]
		return torch.tensor(ids, dtype=torch.long)


def build_imdb_vocab_from_texts(texts: Iterable[str], min_freq: int = 2):
	"""从文本构建IMDB词汇表"""
	# 统计词频（使用 NLTK 分词）
	freq: Dict[str, int] = {}
	for text in texts:
		for tok in nltk_tokenize(text):
			tok_l = tok.lower()
			freq[tok_l] = freq.get(tok_l, 0) + 1
	# 特殊符号
	specials = ["<unk>", "<pad>"]
	tokens = specials + [t for t, c in freq.items() if c >= int(min_freq) and t not in specials]
	token_to_index = {t: i for i, t in enumerate(tokens)}
	vocab = SimpleVocab(token_to_index)
	vocab.set_default_index(token_to_index["<unk>"])
	return nltk_tokenize, vocab


class CollateText:
	"""可 pickling 的 collate_fn。"""
	def __init__(self, text_to_ids: TextToIds, pad_idx: int, max_len: int):
		self.text_to_ids = text_to_ids
		self.pad_idx = int(pad_idx)
		self.max_len = int(max_len)

	def __call__(self, batch):
		"""整理文本批次数据"""
		texts = []
		labels = []
		for item in batch:
			if isinstance(item, dict):
				label_val = int(item['label'])
				text = item['text']
				labels.append(label_val)
				ids = self.text_to_ids(text)
				texts.append(ids)
			else:
				label, text = item
				ids = self.text_to_ids(text)
				texts.append(ids)
				labels.append(0 if label == 'neg' or label == 0 else 1)
		padded = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)
		if padded.size(1) < self.max_len:
			pad_extra = torch.full((padded.size(0), self.max_len - padded.size(1)), self.pad_idx, dtype=padded.dtype)
			padded = torch.cat([padded, pad_extra], dim=1)
		elif padded.size(1) > self.max_len:
			padded = padded[:, : self.max_len]
		labels = torch.tensor(labels, dtype=torch.long)
		return padded, labels
