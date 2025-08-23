"""
路径管理工具模块
"""
import os
from datetime import datetime
from typing import Dict, Optional


class PathManager:
	"""
	路径管理器类，负责管理训练过程中的各种文件路径

	根据是否使用LoRA自动调整目录结构：
	- LoRA模式：使用loras目录存储权重文件
	- 基模训练模式：使用checkpoints目录存储权重文件
	"""

	def __init__(self, root: str = "./outputs", dataset_name: str = None, model_name: str = None, timestamp: str = None, use_lora: bool = False) -> None:
		"""
		初始化路径管理器

		Args:
			root: 根目录路径，默认为"./outputs"
			dataset_name: 数据集名称
			model_name: 模型名称
			timestamp: 时间戳，如果为None则自动生成
			use_lora: 是否使用LoRA微调模式
		"""
		self.root = root
		self.dataset_name = dataset_name or "unknown"
		self.model_name = model_name or "unknown"
		self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
		self.use_lora = use_lora

		# 根据是否使用LoRA调整命名格式
		if use_lora:
			self.run_name = f"{self.dataset_name}_{self.model_name}_lora_{self.timestamp}"
		else:
			self.run_name = f"{self.dataset_name}_{self.model_name}_{self.timestamp}"
		self._ensure_dirs()

	def _ensure_dirs(self) -> None:
		"""
		确保所有必要的目录都存在
		"""
		os.makedirs(self.root, exist_ok=True)
		
		# 基础目录（总是需要）
		os.makedirs(self.metrics_root, exist_ok=True)
		os.makedirs(self.metrics_server_dir, exist_ok=True)
		os.makedirs(self.metrics_clients_dir, exist_ok=True)
		os.makedirs(self.logs_root, exist_ok=True)
		os.makedirs(self.plots_root, exist_ok=True)

		# 根据是否使用LoRA决定创建哪些目录
		if self.use_lora:
			# LoRA模式：只创建loras目录
			os.makedirs(self.loras_root, exist_ok=True)
			os.makedirs(self.lora_server_dir, exist_ok=True)
			os.makedirs(self.lora_clients_dir, exist_ok=True)
		else:
			# 基模训练模式：只创建checkpoints目录
			os.makedirs(self.fed_root, exist_ok=True)
			os.makedirs(self.checkpoint_server_dir, exist_ok=True)
			os.makedirs(self.checkpoint_clients_dir, exist_ok=True)

	@property
	def fed_root(self) -> str:
		"""
		联邦学习根目录（基模训练模式）

		Returns:
			联邦学习checkpoints根目录路径
		"""
		return os.path.join(self.root, "checkpoints", self.run_name)

	@property
	def checkpoint_server_dir(self) -> str:
		"""
		服务器检查点目录（基模训练模式）

		Returns:
			服务器检查点目录路径
		"""
		return os.path.join(self.fed_root, "server")

	@property
	def checkpoint_clients_dir(self) -> str:
		"""
		客户端检查点目录（基模训练模式）

		Returns:
			客户端检查点目录路径
		"""
		return os.path.join(self.fed_root, "clients")

	def client_round_ckpt(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次检查点文件路径（基模训练模式）

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端检查点文件路径

		Raises:
			ValueError: 在LoRA模式下调用时抛出异常
		"""
		if self.use_lora:
			raise ValueError("LoRA模式下不应该使用checkpoints目录，请使用client_lora_round_path方法")
		client_dir = os.path.join(self.checkpoint_clients_dir, f"client_{client_id}")
		os.makedirs(client_dir, exist_ok=True)
		return os.path.join(client_dir, f"round_{round_id}.pth")

	def global_round_ckpt(self, round_id: int) -> str:
		"""
		获取全局轮次检查点文件路径（基模训练模式）

		Args:
			round_id: 轮次ID

		Returns:
			全局检查点文件路径

		Raises:
			ValueError: 在LoRA模式下调用时抛出异常
		"""
		if self.use_lora:
			raise ValueError("LoRA模式下不应该使用checkpoints目录，请使用server_lora_round_path方法")
		return os.path.join(self.checkpoint_server_dir, f"round_{round_id}.pth")

	@property
	def metrics_root(self) -> str:
		"""
		指标数据根目录

		Returns:
			指标数据根目录路径
		"""
		return os.path.join(self.root, "metrics", self.run_name)

	@property
	def metrics_server_dir(self) -> str:
		"""
		服务器指标数据目录

		Returns:
			服务器指标数据目录路径
		"""
		return os.path.join(self.metrics_root, "server")

	@property
	def metrics_clients_dir(self) -> str:
		"""
		客户端指标数据目录

		Returns:
			客户端指标数据目录路径
		"""
		return os.path.join(self.metrics_root, "clients")

	def client_round_metrics(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次指标文件路径

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端指标文件路径
		"""
		client_dir = os.path.join(self.metrics_clients_dir, f"client_{client_id}")
		os.makedirs(client_dir, exist_ok=True)
		return os.path.join(client_dir, f"round_{round_id}.json")

	def round_metrics(self, round_id: int) -> str:
		"""
		获取轮次指标文件路径

		Args:
			round_id: 轮次ID

		Returns:
			轮次指标文件路径
		"""
		return os.path.join(self.metrics_server_dir, f"round_{round_id}.json")

	@property
	def logs_root(self) -> str:
		"""
		日志文件根目录

		Returns:
			日志文件根目录路径
		"""
		return os.path.join(self.root, "logs", self.run_name)

	def client_round_log(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次日志文件路径

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端日志文件路径
		"""
		client_dir = os.path.join(self.logs_root, "clients", f"client_{client_id}")
		os.makedirs(client_dir, exist_ok=True)
		return os.path.join(client_dir, f"round_{round_id}.log")

	def server_round_log(self, round_id: int) -> str:
		"""
		获取服务器轮次日志文件路径，确保目录存在

		Args:
			round_id: 轮次ID

		Returns:
			服务器日志文件路径
		"""
		server_dir = os.path.join(self.logs_root, "server")
		os.makedirs(server_dir, exist_ok=True)
		return os.path.join(server_dir, f"round_{round_id}.log")

	@property
	def plots_root(self) -> str:
		"""
		图表文件根目录

		Returns:
			图表文件根目录路径
		"""
		return os.path.join(self.root, "plots", self.run_name)

	def client_plots_dir(self, client_id: int) -> str:
		"""
		获取客户端图表目录路径，确保目录存在

		Args:
			client_id: 客户端ID

		Returns:
			客户端图表目录路径
		"""
		client_dir = os.path.join(self.plots_root, f"client_{client_id}")
		os.makedirs(client_dir, exist_ok=True)
		return client_dir

	def client_round_plot(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次图表文件路径

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端图表文件路径
		"""
		client_dir = self.client_plots_dir(client_id)
		return os.path.join(client_dir, f"client_{client_id}_round_{round_id}_metrics.png")

	# LoRA 相关路径方法
	@property
	def loras_root(self) -> str:
		"""
		LoRA权重文件根目录（LoRA模式）

		Returns:
			LoRA权重文件根目录路径
		"""
		return os.path.join(self.root, "loras", self.run_name)

	@property
	def lora_server_dir(self) -> str:
		"""
		服务器LoRA权重目录（LoRA模式）

		Returns:
			服务器LoRA权重目录路径
		"""
		return os.path.join(self.loras_root, "server")

	@property
	def lora_clients_dir(self) -> str:
		"""
		客户端LoRA权重目录（LoRA模式）

		Returns:
			客户端LoRA权重目录路径
		"""
		return os.path.join(self.loras_root, "clients")

	def client_lora_round_path(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次LoRA权重文件路径（LoRA模式）

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端LoRA权重文件路径

		Raises:
			ValueError: 在非LoRA模式下调用时抛出异常
		"""
		if not self.use_lora:
			raise ValueError("非LoRA模式下不应该使用loras目录，请使用client_round_ckpt方法")
		client_dir = os.path.join(self.lora_clients_dir, f"client_{client_id}")
		os.makedirs(client_dir, exist_ok=True)
		return os.path.join(client_dir, f"lora_round_{round_id}.pth")

	def server_lora_round_path(self, round_id: int) -> str:
		"""
		获取服务器轮次LoRA权重文件路径（LoRA模式）

		Args:
			round_id: 轮次ID

		Returns:
			服务器LoRA权重文件路径

		Raises:
			ValueError: 在非LoRA模式下调用时抛出异常
		"""
		if not self.use_lora:
			raise ValueError("非LoRA模式下不应该使用loras目录，请使用global_round_ckpt方法")
		return os.path.join(self.lora_server_dir, f"lora_round_{round_id}.pth")