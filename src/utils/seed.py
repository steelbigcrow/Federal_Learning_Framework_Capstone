"""
随机种子设置工具模块
"""
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
	"""
	设置随机种子以确保实验的可重复性

	Args:
		seed: 随机种子值
		deterministic: 是否启用确定性模式（会降低性能但确保完全可重复）
	"""
	random.seed(seed)  # 设置Python内置random模块的种子
	np.random.seed(seed)  # 设置NumPy随机种子
	torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
	torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子

	if deterministic:
		# 启用确定性模式以确保完全可重复的结果
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 限制cuBLAS工作空间
		torch.backends.cudnn.deterministic = True  # 启用cuDNN确定性模式
		torch.backends.cudnn.benchmark = False  # 禁用基准测试模式
	else:
		# 性能优化模式（可能会降低可重复性）
		torch.backends.cudnn.deterministic = False  # 禁用cuDNN确定性模式
		torch.backends.cudnn.benchmark = True  # 启用基准测试模式以获得最佳性能
