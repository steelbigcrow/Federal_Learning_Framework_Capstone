from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def evaluate(model, data_loader, device) -> Dict[str, float]:
	"""
	模型评估函数

	Args:
		model: 要评估的模型
		data_loader: 测试数据加载器
		device: 设备（CPU或GPU）

	Returns:
		包含损失、准确率和F1分数的字典
	"""
	# 处理空数据集情况
	if len(data_loader.dataset) == 0:
		return {"loss": 0.0, "acc": 0.0, "f1": 0.0}

	# 保存原始训练状态
	original_training = model.training
	model.eval()

	total_loss = 0.0
	total_correct = 0
	total = 0
	all_predictions = []
	all_labels = []

	# 验证模型在正确设备上（仅第一个批次）
	first_batch = True

	try:
		with torch.no_grad():
			for batch in data_loader:
				if isinstance(batch, (list, tuple)) and len(batch) == 2:
					x, y = batch
				else:
					raise ValueError("Batch format must be (x, y)")
				x = x.to(device)
				y = y.to(device)

				# 第一个批次时验证设备
				if first_batch:
					model_device = next(model.parameters()).device
					first_batch = False
				logits = model(x)
				loss = F.cross_entropy(logits, y)
				pred = logits.argmax(dim=1)
				total_correct += (pred == y).sum().item()
				total += y.size(0)
				total_loss += loss.item() * y.size(0)

				# 收集预测结果和标签用于F1计算
				all_predictions.extend(pred.cpu().numpy())
				all_labels.extend(y.cpu().numpy())

		# 计算F1分数
		f1 = f1_score(all_labels, all_predictions, average='weighted')

		return {
			"loss": total_loss / max(1, total),
			"acc": total_correct / max(1, total),
			"f1": f1
		}
	finally:
		# 恢复原始训练状态
		if original_training:
			model.train()
