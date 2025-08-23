"""
检查检查点文件脚本

该脚本用于检查PyTorch模型检查点文件的内容和结构，
显示模型状态字典中的键值对和张量形状信息。
"""

import argparse
import torch


def main():
	"""
	主函数：检查检查点文件内容

	从指定的检查点文件加载模型状态字典，并显示：
	1. 状态字典中键的数量
	2. 前50个键值对及其张量形状
	"""
	# 设置命令行参数解析器
	parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint files")
	parser.add_argument('--path', type=str, required=True, help="Path to the checkpoint file")
	args = parser.parse_args()

	# 加载检查点文件到CPU内存
	state = torch.load(args.path, map_location='cpu')

	# 显示状态字典的基本信息
	print(f'Keys in checkpoint: {len(state.keys())}')

	# 显示前50个键值对的形状信息
	for k, v in list(state.items())[:50]:
		print(f'{k}: {tuple(v.shape)}')


if __name__ == '__main__':
	main()
