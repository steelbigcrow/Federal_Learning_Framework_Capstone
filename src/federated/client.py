"""
联邦学习客户端模块
"""
from copy import deepcopy
from typing import Dict, Tuple

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split

from ..training.train import train_one_epoch
from ..training.evaluate import evaluate


class Client:
	"""
	联邦学习客户端类

	负责本地模型训练、数据分割和本地评估
	"""

	def __init__(self, client_id: int, model_ctor, train_loader, device, optimizer_cfg: Dict, use_amp: bool = False, test_ratio: float = 0.3):
		"""
		初始化联邦学习客户端

		Args:
			client_id: 客户端ID
			model_ctor: 模型构造函数
			train_loader: 训练数据加载器
			device: 计算设备
			optimizer_cfg: 优化器配置
			use_amp: 是否使用自动混合精度
			test_ratio: 测试集比例
		"""
		self.id = client_id
		self.model_ctor = model_ctor
		self.device = device
		self.optimizer_cfg = optimizer_cfg
		self.use_amp = use_amp
		self.test_ratio = test_ratio

		# 将接收到的训练数据分割为训练集和测试集
		original_dataset = train_loader.dataset
		total_size = len(original_dataset)

		# 按照70%训练，30%测试分割
		test_size = int(total_size * test_ratio)
		train_size = total_size - test_size

		# 分割数据集
		train_dataset_new, test_dataset = random_split(
			original_dataset,
			[train_size, test_size],
			generator=torch.Generator().manual_seed(42 + client_id)  # 每个客户端使用不同的随机种子
		)

		# 创建新的数据加载器
		self.train_loader = DataLoader(
			train_dataset_new,
			batch_size=train_loader.batch_size,
			shuffle=True,
			num_workers=getattr(train_loader, 'num_workers', 0),
			pin_memory=getattr(train_loader, 'pin_memory', False),
			collate_fn=getattr(train_loader, 'collate_fn', None)
		)

		self.test_loader = DataLoader(
			test_dataset,
			batch_size=train_loader.batch_size,
			shuffle=False,
			num_workers=getattr(train_loader, 'num_workers', 0),
			pin_memory=getattr(train_loader, 'pin_memory', False),
			collate_fn=getattr(train_loader, 'collate_fn', None)
		)

	def _build_optimizer(self, model):
		"""
		根据配置构建优化器

		Args:
			model: PyTorch模型

		Returns:
			配置好的优化器实例
		"""
		name = self.optimizer_cfg.get('name', 'adam').lower()
		lr = float(self.optimizer_cfg.get('lr', 1e-3))
		wd = float(self.optimizer_cfg.get('weight_decay', 0.0))
		if name == 'adamw':
			return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
		return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

	def local_train_and_eval(self, global_state_dict: Dict[str, torch.Tensor], local_epochs: int = 1) -> Tuple[Dict, Dict, int]:
		"""
		执行本地训练和评估

		Args:
			global_state_dict: 全局模型状态字典
			local_epochs: 本地训练轮数

		Returns:
			训练后的模型状态字典、评估指标字典、训练样本数量
		"""
		model = self.model_ctor()
		model.load_state_dict(global_state_dict, strict=False)
		model.to(self.device)

		# 验证模型已正确移动到设备
		model_device = next(model.parameters()).device

		optimizer = self._build_optimizer(model)
		scaler = torch.cuda.amp.GradScaler() if (self.use_amp and self.device.startswith('cuda')) else None

		# 记录每个epoch的训练指标
		train_metrics_history = []

		for e in range(1, local_epochs + 1):
			train_metrics = train_one_epoch(model, self.train_loader, optimizer, self.device, scaler)

			# 在测试集上评估当前epoch的模型
			test_metrics = evaluate(model, self.test_loader, self.device)

			# 为每个epoch记录完整的训练和测试指标
			epoch_metrics = {
				'epoch': e,
				'train_acc': train_metrics['acc'],
				'train_f1': train_metrics['f1'],
				'train_loss': train_metrics['loss'],
				'test_acc': test_metrics['acc'],
				'test_f1': test_metrics['f1'],
				'test_loss': test_metrics['loss']
			}
			train_metrics_history.append(epoch_metrics)

			# 打印每个epoch的详细指标
			print(f"    Epoch {e}: train loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, f1={train_metrics['f1']:.4f} | test acc={test_metrics['acc']:.4f}, f1={test_metrics['f1']:.4f}")

		# 计算最后一个epoch的指标作为最终结果（用于聚合）
		final_epoch_metrics = train_metrics_history[-1] if train_metrics_history else {}

		# 返回详细的训练历史和最终指标
		combined_metrics = {
			# 最终epoch的指标（用于聚合）
			'train_acc': final_epoch_metrics.get('train_acc', 0),
			'train_f1': final_epoch_metrics.get('train_f1', 0),
			'train_loss': final_epoch_metrics.get('train_loss', 0),
			'test_acc': final_epoch_metrics.get('test_acc', 0),
			'test_f1': final_epoch_metrics.get('test_f1', 0),
			'test_loss': final_epoch_metrics.get('test_loss', 0),
			# 为了兼容性，保留原有的 acc 和 loss 字段
			'acc': final_epoch_metrics.get('test_acc', 0),
			'loss': final_epoch_metrics.get('test_loss', 0),
			# 新增：每个epoch的详细训练历史
			'epoch_history': train_metrics_history
		}

		return deepcopy(model.state_dict()), combined_metrics, len(self.train_loader.dataset)
