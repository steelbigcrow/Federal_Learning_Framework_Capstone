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

	def __init__(self, root: str = "./outputs", dataset_name: str = None, model_name: str = None, timestamp: str = None, use_lora: bool = False, use_adalora: bool = False) -> None:
		"""
		初始化路径管理器

		Args:
			root: 根目录路径，默认为"./outputs"
			dataset_name: 数据集名称
			model_name: 模型名称
			timestamp: 时间戳，如果为None则自动生成
			use_lora: 是否使用LoRA微调模式
			use_adalora: 是否使用AdaLoRA微调模式
		"""
		self.root = root
		self.dataset_name = dataset_name or "unknown"
		self.model_name = model_name or "unknown"
		self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
		self.use_lora = use_lora
		self.use_adalora = use_adalora

		# 根据训练模式调整命名格式
		if use_adalora:
			self.run_name = f"{self.dataset_name}_{self.model_name}_adalora_{self.timestamp}"
			self.model_type = "adaloras"
		elif use_lora:
			self.run_name = f"{self.dataset_name}_{self.model_name}_lora_{self.timestamp}"
			self.model_type = "loras"
		else:
			self.run_name = f"{self.dataset_name}_{self.model_name}_{self.timestamp}"
			self.model_type = "models"
		self._ensure_dirs()

	def _ensure_dirs(self) -> None:
		"""
		确保所有必要的目录都存在
		"""
		os.makedirs(self.root, exist_ok=True)
		
		# 基础目录（总是需要）
		os.makedirs(self.run_root, exist_ok=True)
		os.makedirs(self.weights_root, exist_ok=True)
		os.makedirs(self.weights_server_dir, exist_ok=True)
		os.makedirs(self.weights_clients_dir, exist_ok=True)
		os.makedirs(self.logs_root, exist_ok=True)
		os.makedirs(self.logs_server_dir, exist_ok=True)
		os.makedirs(self.logs_clients_dir, exist_ok=True)
		os.makedirs(self.metrics_root, exist_ok=True)
		os.makedirs(self.metrics_server_dir, exist_ok=True)
		os.makedirs(self.metrics_clients_dir, exist_ok=True)
		os.makedirs(self.plots_root, exist_ok=True)
		os.makedirs(self.plots_server_dir, exist_ok=True)
		os.makedirs(self.plots_clients_dir, exist_ok=True)

	@property
	def run_root(self) -> str:
		"""
		运行根目录

		Returns:
			运行根目录路径
		"""
		return os.path.join(self.root, self.model_type, self.run_name)

	@property
	def output_dir(self) -> str:
		"""
		输出目录（与run_root相同，用于保持与旧接口的兼容性）

		Returns:
			输出目录路径
		"""
		return self.run_root

	@property
	def weights_root(self) -> str:
		"""
		权重文件根目录

		Returns:
			权重文件根目录路径
		"""
		return os.path.join(self.run_root, "weights")

	@property
	def weights_server_dir(self) -> str:
		"""
		服务器权重目录

		Returns:
			服务器权重目录路径
		"""
		return os.path.join(self.weights_root, "server")

	@property
	def weights_clients_dir(self) -> str:
		"""
		客户端权重目录

		Returns:
			客户端权重目录路径
		"""
		return os.path.join(self.weights_root, "clients")

	def client_round_ckpt(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次权重文件路径

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端权重文件路径
		"""
		client_dir = os.path.join(self.weights_clients_dir, f"client_{client_id}")
		os.makedirs(client_dir, exist_ok=True)
		if self.use_adalora:
			return os.path.join(client_dir, f"adalora_round_{round_id}.pth")
		elif self.use_lora:
			return os.path.join(client_dir, f"lora_round_{round_id}.pth")
		else:
			return os.path.join(client_dir, f"round_{round_id}.pth")

	def global_round_ckpt(self, round_id: int) -> str:
		"""
		获取全局轮次权重文件路径

		Args:
			round_id: 轮次ID

		Returns:
			全局权重文件路径
		"""
		if self.use_adalora:
			return os.path.join(self.weights_server_dir, f"adalora_round_{round_id}.pth")
		elif self.use_lora:
			return os.path.join(self.weights_server_dir, f"lora_round_{round_id}.pth")
		else:
			return os.path.join(self.weights_server_dir, f"round_{round_id}.pth")

	@property
	def metrics_root(self) -> str:
		"""
		指标数据根目录

		Returns:
			指标数据根目录路径
		"""
		return os.path.join(self.run_root, "metrics")

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

	def server_round_metrics(self, round_id: int) -> str:
		"""
		获取服务器轮次指标文件路径

		Args:
			round_id: 轮次ID

		Returns:
			服务器指标文件路径
		"""
		os.makedirs(self.metrics_server_dir, exist_ok=True)
		return os.path.join(self.metrics_server_dir, f"round_{round_id}.json")
	
	def server_round_log(self, round_id: int) -> str:
		"""
		获取服务器轮次日志文件路径

		Args:
			round_id: 轮次ID

		Returns:
			服务器日志文件路径
		"""
		os.makedirs(self.logs_server_dir, exist_ok=True)
		return os.path.join(self.logs_server_dir, f"round_{round_id}.log")

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
		return os.path.join(self.run_root, "logs")

	@property
	def logs_server_dir(self) -> str:
		"""
		服务器日志目录

		Returns:
			服务器日志目录路径
		"""
		return os.path.join(self.logs_root, "server")

	@property
	def logs_clients_dir(self) -> str:
		"""
		客户端日志目录

		Returns:
			客户端日志目录路径
		"""
		return os.path.join(self.logs_root, "clients")

	def client_round_log(self, client_id: int, round_id: int) -> str:
		"""
		获取客户端轮次日志文件路径

		Args:
			client_id: 客户端ID
			round_id: 轮次ID

		Returns:
			客户端日志文件路径
		"""
		client_dir = os.path.join(self.logs_clients_dir, f"client_{client_id}")
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
		return os.path.join(self.logs_server_dir, f"round_{round_id}.log")

	@property
	def plots_root(self) -> str:
		"""
		图表文件根目录

		Returns:
			图表文件根目录路径
		"""
		return os.path.join(self.run_root, "plots")

	@property
	def plots_server_dir(self) -> str:
		"""
		服务器图表目录

		Returns:
			服务器图表目录路径
		"""
		return os.path.join(self.plots_root, "server")

	@property
	def plots_clients_dir(self) -> str:
		"""
		客户端图表目录

		Returns:
			客户端图表目录路径
		"""
		return os.path.join(self.plots_root, "clients")

	def client_plots_dir(self, client_id: int) -> str:
		"""
		获取客户端图表目录路径，确保目录存在

		Args:
			client_id: 客户端ID

		Returns:
			客户端图表目录路径
		"""
		client_dir = os.path.join(self.plots_clients_dir, f"client_{client_id}")
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

	# 移除旧的LoRA相关方法，现在统一使用上面的方法