import json
import os
from typing import Dict


def write_metrics_json(path: str, metrics: Dict) -> None:
	"""
	将指标数据写入JSON文件

	Args:
		path: 文件保存路径
		metrics: 指标数据字典
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_text_log(path: str, text: str) -> None:
	"""
	将文本日志写入文件（追加模式）

	Args:
		path: 文件保存路径
		text: 要写入的文本内容
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'a', encoding='utf-8') as f:
		f.write(text + "\n")
