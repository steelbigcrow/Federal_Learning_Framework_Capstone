from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_one_epoch(model, data_loader, optimizer, device, scaler=None) -> Dict[str, float]:
	"""
	训练一个epoch的函数

	Args:
		model: 要训练的模型
		data_loader: 训练数据加载器
		optimizer: 优化器
		device: 训练设备（CPU或GPU）
		scaler: 梯度缩放器（用于混合精度训练）

	Returns:
		包含损失、准确率和F1分数的字典
	"""
	# 处理空数据集情况
	if len(data_loader.dataset) == 0:
		return {"loss": 0.0, "acc": 0.0, "f1": 0.0}

	model.train()
	total_loss = 0.0
	total_correct = 0
	total = 0
	all_predictions = []
	all_labels = []

	# 在第一个批次时显示设备信息
	first_batch = True

	for step, batch in enumerate(data_loader, start=1):
		if isinstance(batch, (list, tuple)) and len(batch) == 2:
			x, y = batch
		else:
			raise ValueError("Batch format must be (x, y)")
		x = x.to(device)
		y = y.to(device)

		# 第一个批次时打印设备验证信息
		if first_batch:
			first_batch = False

		# 清空梯度
		optimizer.zero_grad(set_to_none=True)

		# 混合精度训练
		if scaler is not None:
			with torch.cuda.amp.autocast():
				logits = model(x)
				loss = F.cross_entropy(logits, y)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			# 标准训练
			logits = model(x)
			loss = F.cross_entropy(logits, y)
			loss.backward()
			optimizer.step()

		# 计算准确率并收集预测结果用于F1计算
		pred = logits.argmax(dim=1)
		total_correct += (pred == y).sum().item()
		total_loss += loss.item() * y.size(0)
		total += y.size(0)

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
