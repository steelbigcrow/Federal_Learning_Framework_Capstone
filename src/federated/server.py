"""
联邦学习服务器模块
"""
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import torch

from .aggregator import fedavg, lora_fedavg, adalora_fedavg, get_trainable_keys
from ..training.checkpoints import save_client_round, save_global_round
from ..training.logging_utils import write_metrics_json, write_text_log
from ..training.plotting import plot_all_clients_metrics
from ..training.lora_utils import save_lora_checkpoint
from ..training.adalora_utils import save_adalora_checkpoint, get_adalora_regularization_loss
from ..utils.paths import PathManager


class Server:
	"""
	联邦学习服务器类

	负责协调联邦学习过程、模型聚合、日志记录和检查点管理
	"""

	def __init__(self, model_ctor: Callable[[], torch.nn.Module], clients, path_manager: PathManager, device: str = 'cpu', lora_cfg: Optional[Dict] = None, adalora_cfg: Optional[Dict] = None, save_client_each_round: bool = True, model_info: Optional[Dict] = None):
		"""
		初始化联邦学习服务器

		Args:
			model_ctor: 模型构造函数
			clients: 客户端列表
			path_manager: 路径管理器
			device: 计算设备
			lora_cfg: LoRA配置
			adalora_cfg: AdaLoRA配置
			save_client_each_round: 是否每轮保存客户端模型
			model_info: 模型信息
		"""
		self.model_ctor = model_ctor
		self.clients = clients
		self.paths = path_manager
		self.device = device
		self.global_model = model_ctor().to(device)
		self.lora_cfg = lora_cfg or {}
		self.adalora_cfg = adalora_cfg or {}
		self.save_client_each_round = bool(save_client_each_round)
		self.model_info = model_info or {}  # 存储模型类型和规格信息

		# 验证全局模型设备
		global_model_device = next(self.global_model.parameters()).device

	def run(self, num_rounds: int, local_epochs: int) -> None:
		"""
		执行联邦学习训练过程

		Args:
			num_rounds: 联邦学习轮数
			local_epochs: 客户端本地训练轮数
		"""
		print(f"[Federated Learning] Starting training: {num_rounds} rounds, {len(self.clients)} clients, device={self.device}")

		for r in range(1, num_rounds + 1):
			print(f"\n[Round {r}/{num_rounds}] Starting federated training")

			# 显示客户端数据分割情况
			print(f"Client data distribution:")
			for client in self.clients:
				train_size = len(client.train_loader.dataset)
				test_size = len(client.test_loader.dataset)
				total_size = train_size + test_size
				print(f"  Client {client.id}: total={total_size}, train={train_size}, test={test_size}")
			
			server_log = self.paths.server_round_log(r)
			self.safe_write_single_log(server_log, f"[Federated] >>> Starting Round {r}/{num_rounds}")
			state_dicts: List[Dict] = []
			num_samples: List[int] = []
			for client in self.clients:
				log_file = self.paths.client_round_log(client.id, r)

				msg_dispatch = f"[Federated][Round {r}] Distributing global weights to Client {client.id}"
				self.safe_write_single_log(server_log, msg_dispatch)

				print(f"  Client {client.id} starting training...")
				try:
					client_sd, metrics, n = client.local_train_and_eval(deepcopy(self.global_model.state_dict()), local_epochs)
				except Exception as e:
					print(f"[Error] Client {client.id} training failed: {e}")
					continue
					
				# 构建详细的评估消息，包含5个指标（只写入日志，不打印到终端）
				if 'train_acc' in metrics and 'test_acc' in metrics and 'train_f1' in metrics and 'test_f1' in metrics:
					msg = f"[Federated][Round {r}] Client {client.id} evaluation completed: train(acc={metrics['train_acc']:.4f}, f1={metrics['train_f1']:.4f}, loss={metrics['train_loss']:.4f}), test(acc={metrics['test_acc']:.4f}, f1={metrics['test_f1']:.4f}), samples={n}"
				else:
					# 向后兼容
					msg = f"[Federated][Round {r}] Client {client.id} evaluation completed: acc={metrics['acc']:.4f}, loss={metrics['loss']:.4f}, samples={n}"

				self.safe_write_logs(log_file, server_log, msg)

				print(f"  Client {client.id} training completed")
				
				try:
					# 扩展指标信息，包含模型信息和详细历史
					extended_metrics = {
						**metrics,  # 包含所有原有指标
						'round': r,
						'client': client.id,
						'num_samples': n,
						'model_info': self.model_info,  # 添加模型信息
						'lora': self.lora_cfg
					}
					metrics_path = self.paths.client_round_metrics(client.id, r)
					write_metrics_json(metrics_path, extended_metrics)
				except Exception as e:
					print(f"[Error] Failed to save client metrics: {e}")

				# 根据是否使用LoRA决定保存策略
				if self.save_client_each_round:
					# 构建包含详细评估结果的元数据
					meta = {
						"round": r,
						"client": client.id,
						"num_samples": n,
						"metrics": metrics,  # 包含完整的评估指标
						"lora": self.lora_cfg,
					}
					
					# 统一保存权重到weights文件夹
					try:
						ckpt_path = self.paths.client_round_ckpt(client.id, r)
						meta = self.build_client_meta(r, client.id, n, metrics)

						# LoRA模式：保存LoRA权重
						if self.lora_cfg and self.lora_cfg.get('replaced_modules'):
							# 创建临时模型来提取LoRA权重
							temp_model = self.model_ctor()
							temp_model.load_state_dict(client_sd, strict=False)

							msg_saved = f"[Federated][Round {r}] Client {client.id} LoRA weights saved: {ckpt_path}"
							success = self.safe_save_and_log(
								save_lora_checkpoint, msg_saved, log_file, server_log, ckpt_path,
								temp_model, ckpt_path, meta
							)
							if not success:
								print(f"[Error] Failed to save client {client.id} LoRA weights")
							else:
								print(f"[[Federated][Round {r}] Client {client.id} LoRA weights saved: {ckpt_path}")
						# 基模训练模式：保存完整权重
						else:
							msg_saved = f"[Federated][Round {r}] Client {client.id} complete model saved: {ckpt_path}"
							success = self.safe_save_and_log(
								save_client_round, msg_saved, log_file, server_log, ckpt_path,
								client_sd, ckpt_path, meta=meta
							)
							if not success:
								print(f"[Error] Failed to save client {client.id} model")
							else:
								print(f"[[Federated][Round {r}] Client {client.id} complete model saved: {ckpt_path}")
					except Exception as e:
						print(f"[Error] Failed to save client {client.id} weights: {e}")
				state_dicts.append(client_sd)
				num_samples.append(n)
			# 聚合
			print(f"[Round {r}/{num_rounds}] Starting aggregation of {len(state_dicts)} client models")

			self.safe_write_single_log(server_log, f"[Federated][Round {r}] Starting aggregation (FedAvg)...")

			# 根据是否使用LoRA选择合适的聚合策略
			try:
				if self.lora_cfg and self.lora_cfg.get('replaced_modules'):
					# LoRA模式：只聚合可训练的权重（LoRA权重+分类头）
					trainable_keys = get_trainable_keys(self.global_model)
					print(f"[LoRA Aggregation] Trainable parameters count: {len(trainable_keys)}")
					print(f"[LoRA Aggregation] Trainable parameters: {sorted(list(trainable_keys))[:5]}...")  # 显示前5个

					new_global = lora_fedavg(state_dicts, num_samples, trainable_keys)
					self.global_model.load_state_dict(new_global, strict=False)
					print(f"[Round {r}/{num_rounds}] LoRA aggregation completed (trainable weights only)")
				else:
					# 基模训练模式：聚合所有权重
					new_global = fedavg(state_dicts, num_samples)
					self.global_model.load_state_dict(new_global, strict=False)
					print(f"[Round {r}/{num_rounds}] Standard aggregation completed (all weights)")
			except Exception as e:
				print(f"[Error] Model aggregation failed: {e}")
				continue
			
			# 根据是否使用LoRA决定全局模型保存策略
			try:
				g_meta = {"round": r, "lora": self.lora_cfg}
				
				# 统一保存全局模型权重到weights文件夹
				try:
					g_path = self.paths.global_round_ckpt(r)
					
					# LoRA模式：保存LoRA权重
					if self.lora_cfg and self.lora_cfg.get('replaced_modules'):
						g_meta = self.build_server_meta(r, is_lora=True)
						msg_global = f"[Federated][Round {r}] Global LoRA weights saved: {g_path}"
						success = self.safe_save_and_log(
							save_lora_checkpoint, msg_global, server_log, server_log, g_path,
							self.global_model, g_path, g_meta
						)
						if success:
							print(f"[Round {r}/{num_rounds}] Global LoRA weights saved: {g_path}")
						else:
							print(f"[Error] Failed to save global LoRA weights")
					# 基模训练模式：保存完整权重
					else:
						g_meta = self.build_server_meta(r, is_lora=False)
						msg_global = f"[Federated][Round {r}] Aggregation completed, global complete model saved: {g_path}"
						success = self.safe_save_and_log(
							save_global_round, msg_global, server_log, server_log, g_path,
							self.global_model.state_dict(), g_path, meta=g_meta
						)
						if success:
							print(f"[Round {r}/{num_rounds}] Global complete model saved: {g_path}")
						else:
							print(f"[Error] Failed to save global model")
				except Exception as e:
					print(f"[Error] Failed to save global model: {e}")
			except Exception as e:
				print(f"[Error] Global model save process failed: {e}")
			try:
				round_summary = {
					"round": r,
					"num_clients": len(self.clients),
					"num_samples_total": int(sum(num_samples)) if len(num_samples) > 0 else 0,
					"num_samples_per_client": num_samples,
					"model_info": self.model_info,  # 添加模型信息
					"lora": self.lora_cfg,
				}
				round_metrics_path = self.paths.round_metrics(r)
				write_metrics_json(round_metrics_path, round_summary)
			except Exception as e:
				print(f"[Error] Failed to save round summary: {e}")

			# 为所有客户端生成指标图表
			try:
				client_ids = [client.id for client in self.clients]
				plot_all_clients_metrics(
					client_ids=client_ids,
					metrics_clients_dir=self.paths.metrics_clients_dir,
					plots_dir=self.paths.plots_clients_dir,
					current_round=r
				)
			except Exception as e:
				print(f"[Error] Failed to generate client metrics plots: {e}")

	# ===== 辅助方法 =====

	def safe_write_logs(self, log_file: str, server_log: str, msg: str) -> None:
		"""
		安全地写入日志到多个文件

		Args:
			log_file: 客户端日志文件路径
			server_log: 服务器日志文件路径
			msg: 要写入的消息
		"""
		for log_path in [log_file, server_log]:
			try:
				write_text_log(log_path, msg)
			except Exception as e:
				print(f"[Error] Failed to write log {log_path}: {e}")

	def safe_write_single_log(self, log_path: str, msg: str) -> None:
		"""
		安全地写入日志到单个文件

		Args:
			log_path: 日志文件路径
			msg: 要写入的消息
		"""
		try:
			write_text_log(log_path, msg)
		except Exception as e:
			print(f"[Error] Failed to write log {log_path}: {e}")

	def safe_save_and_log(self, save_func, success_msg: str,
						 log_file: str, server_log: str, print_path: str, *args, **kwargs) -> bool:
		"""
		安全地执行保存操作并记录日志

		Args:
			save_func: 保存函数
			success_msg: 成功消息
			log_file: 客户端日志文件
			server_log: 服务器日志文件
			print_path: 打印路径
			*args, **kwargs: 传递给保存函数的参数

		Returns:
			bool: 保存是否成功
		"""
		try:
			save_func(*args, **kwargs)
			try:
				write_text_log(log_file, success_msg)
				write_text_log(server_log, success_msg)
				print(f"  Saved to: {print_path}")
				return True
			except Exception as e:
				print(f"[Error] Failed to write save success message: {e}")
				return False
		except Exception as e:
			print(f"[Error] Failed to save: {e}")
			return False

	def build_client_meta(self, round_num: int, client_id: int, num_samples: int, metrics: Dict) -> Dict:
		"""
		构建客户端元数据

		Args:
			round_num: 轮数
			client_id: 客户端ID
			num_samples: 样本数量
			metrics: 评估指标

		Returns:
			Dict: 客户端元数据
		"""
		return {
			"round": round_num,
			"client": client_id,
			"num_samples": num_samples,
			"metrics": metrics,
			"lora": self.lora_cfg,
		}

	def build_lora_meta(self, round_num: int, client_id: int, metrics: Dict) -> Dict:
		"""
		构建LoRA元数据

		Args:
			round_num: 轮数
			client_id: 客户端ID
			metrics: 评估指标

		Returns:
			Dict: LoRA元数据
		"""
		return {
			"round": round_num,
			"client": client_id,
			"dataset": self.model_info.get('dataset', 'unknown'),
			"model_type": self.model_info.get('model_type', 'unknown'),
			"base_model_path": self.lora_cfg.get('base_model_path', ''),
			"lora_config": self.lora_cfg,
			"metrics": metrics
		}

	def build_server_meta(self, round_num: int, is_lora: bool = False) -> Dict:
		"""
		构建服务器元数据

		Args:
			round_num: 轮数
			is_lora: 是否为LoRA模式

		Returns:
			Dict: 服务器元数据
		"""
		if is_lora:
			return {
				"round": round_num,
				"dataset": self.model_info.get('dataset', 'unknown'),
				"model_type": self.model_info.get('model_type', 'unknown'),
				"base_model_path": self.lora_cfg.get('base_model_path', ''),
				"lora_config": self.lora_cfg,
				"is_global": True
			}
		else:
			return {"round": round_num, "lora": self.lora_cfg}