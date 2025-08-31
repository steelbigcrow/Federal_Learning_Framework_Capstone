"""
联邦学习服务器抽象基类

定义了联邦学习服务器的标准接口和行为规范。
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
import torch
from datetime import datetime

from .component import FederatedComponent, ComponentStatus
from .client import AbstractClient
from ..exceptions.exceptions import ServerConfigurationError, ServerOperationError


class AbstractServer(FederatedComponent):
    """
    联邦学习服务器抽象基类
    
    定义了服务器在联邦学习中的核心职责：
    1. 管理全局模型
    2. 协调客户端训练
    3. 执行模型聚合
    4. 管理训练轮次
    5. 持久化模型和指标
    
    设计原则：
    - 单一职责：专注于全局协调和聚合
    - 开闭原则：通过策略模式支持不同聚合算法
    - 依赖倒置：依赖抽象的聚合策略和客户端接口
    """
    
    def __init__(self, 
                 model_constructor: Callable[[], torch.nn.Module],
                 clients: List[AbstractClient],
                 config: Dict[str, Any],
                 device: str = "cpu",
                 **kwargs):
        """
        初始化联邦学习服务器
        
        Args:
            model_constructor: 模型构造函数
            clients: 参与联邦学习的客户端列表
            config: 服务器配置
            device: 计算设备
            **kwargs: 其他参数传递给父类
        """
        self._model_constructor = model_constructor
        self._clients = clients
        self._device = device
        
        # 服务器状态
        self._global_model: Optional[torch.nn.Module] = None
        self._current_round = 0
        self._total_rounds = 0
        self._training_start_time: Optional[datetime] = None
        self._training_end_time: Optional[datetime] = None
        
        # 训练历史和指标
        self._round_history: List[Dict[str, Any]] = []
        self._global_metrics: Dict[str, Any] = {}
        
        # 调用父类初始化
        super().__init__(
            component_id="federated_server",
            config=config,
            **kwargs
        )
        
    @property
    def model_constructor(self) -> Callable[[], torch.nn.Module]:
        """获取模型构造函数"""
        return self._model_constructor
        
    @property
    def clients(self) -> List[AbstractClient]:
        """获取客户端列表"""
        return self._clients.copy()
        
    @property
    def device(self) -> str:
        """获取计算设备"""
        return self._device
        
    @property
    def global_model(self) -> Optional[torch.nn.Module]:
        """获取全局模型"""
        return self._global_model
        
    @property
    def current_round(self) -> int:
        """获取当前轮次"""
        return self._current_round
        
    @property
    def total_rounds(self) -> int:
        """获取总轮数"""
        return self._total_rounds
        
    @property
    def num_clients(self) -> int:
        """获取客户端数量"""
        return len(self._clients)
        
    @property
    def is_training(self) -> bool:
        """检查是否正在训练"""
        return self._status == ComponentStatus.RUNNING
        
    @property
    def training_duration(self) -> Optional[float]:
        """获取训练持续时间（秒）"""
        if self._training_start_time and self._training_end_time:
            return (self._training_end_time - self._training_start_time).total_seconds()
        return None
        
    def get_round_history(self) -> List[Dict[str, Any]]:
        """获取轮次历史记录"""
        return self._round_history.copy()
        
    def get_global_metrics(self) -> Dict[str, Any]:
        """获取全局指标"""
        return self._global_metrics.copy()
        
    def add_client(self, client: AbstractClient) -> None:
        """添加客户端"""
        if client not in self._clients:
            self._clients.append(client)
            self._logger.info(f"Added client {client.client_id} to server")
        else:
            self._logger.warning(f"Client {client.client_id} already exists")
            
    def remove_client(self, client_id: int) -> bool:
        """
        移除客户端
        
        Args:
            client_id: 要移除的客户端ID
            
        Returns:
            是否成功移除
        """
        for i, client in enumerate(self._clients):
            if client.client_id == client_id:
                removed_client = self._clients.pop(i)
                self._logger.info(f"Removed client {removed_client.client_id} from server")
                return True
        self._logger.warning(f"Client {client_id} not found")
        return False
        
    @abstractmethod
    def initialize_global_model(self) -> None:
        """
        初始化全局模型
        
        子类必须实现此方法来创建和初始化全局模型
        
        Raises:
            ServerConfigurationError: 当模型初始化失败时
        """
        pass
        
    @abstractmethod
    def select_clients(self, round_number: int) -> List[AbstractClient]:
        """
        选择参与当前轮次训练的客户端
        
        Args:
            round_number: 当前轮次号
            
        Returns:
            被选中的客户端列表
            
        Note:
            默认实现应该选择所有客户端，子类可以实现自定义选择策略
        """
        pass
        
    @abstractmethod
    def aggregate_models(self, 
                        client_models: List[Dict[str, torch.Tensor]],
                        client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            
        Returns:
            聚合后的全局模型状态字典
            
        Raises:
            ServerOperationError: 当聚合过程失败时
        """
        pass
        
    @abstractmethod
    def save_checkpoint(self, round_number: int, metrics: Dict[str, Any]) -> None:
        """
        保存检查点
        
        Args:
            round_number: 轮次号
            metrics: 当前轮次的指标
            
        Raises:
            ServerOperationError: 当保存失败时
        """
        pass
        
    def run_federated_learning(self, num_rounds: int, local_epochs: int = 1) -> Dict[str, Any]:
        """
        执行联邦学习训练
        
        这是主要的训练流程模板方法：
        1. 初始化全局模型
        2. 执行指定轮数的联邦学习
        3. 返回训练结果
        
        Args:
            num_rounds: 联邦学习轮数
            local_epochs: 客户端本地训练轮数
            
        Returns:
            训练结果字典
            
        Raises:
            ServerOperationError: 当训练过程出现错误时
        """
        try:
            self._set_status(ComponentStatus.INITIALIZING)
            self._total_rounds = num_rounds
            self._training_start_time = datetime.now()
            
            # 初始化全局模型
            self.initialize_global_model()
            
            self._set_status(ComponentStatus.RUNNING)
            self._logger.info(f"Starting federated learning: {num_rounds} rounds, {len(self._clients)} clients")
            
            # 执行联邦学习轮次
            for round_num in range(1, num_rounds + 1):
                self._current_round = round_num
                round_result = self._execute_round(round_num, local_epochs)
                self._round_history.append(round_result)
                
                # 记录轮次完成
                self._logger.info(f"Completed round {round_num}/{num_rounds}")
                
            # 训练完成
            self._training_end_time = datetime.now()
            self._set_status(ComponentStatus.READY)
            
            # 计算最终结果
            final_result = self._compile_final_results()
            self._logger.info(f"Federated learning completed in {self.training_duration:.2f} seconds")
            
            return final_result
            
        except Exception as e:
            self._set_status(ComponentStatus.ERROR)
            self._logger.error(f"Federated learning failed: {e}")
            raise ServerOperationError(f"Federated learning execution failed: {e}") from e
            
    def _execute_round(self, round_number: int, local_epochs: int) -> Dict[str, Any]:
        """
        执行单轮联邦学习
        
        Args:
            round_number: 轮次号
            local_epochs: 本地训练轮数
            
        Returns:
            轮次结果字典
        """
        self._logger.info(f"Starting round {round_number}")
        
        # 1. 选择客户端
        selected_clients = self.select_clients(round_number)
        self._logger.info(f"Selected {len(selected_clients)} clients for round {round_number}")
        
        # 2. 分发全局模型给客户端
        global_state = self._global_model.state_dict()
        client_results = []
        client_weights = []
        
        for client in selected_clients:
            try:
                # 客户端训练
                model_state, metrics, num_samples = client.local_train_and_evaluate(
                    global_state, local_epochs
                )
                
                client_results.append(model_state)
                client_weights.append(float(num_samples))
                
                self._logger.info(f"Client {client.client_id} completed training: {num_samples} samples")
                
            except Exception as e:
                self._logger.error(f"Client {client.client_id} training failed: {e}")
                # 可以选择跳过失败的客户端或终止训练
                continue
                
        if not client_results:
            raise ServerOperationError(f"No clients completed training in round {round_number}")
            
        # 3. 聚合模型
        aggregated_state = self.aggregate_models(client_results, client_weights)
        self._global_model.load_state_dict(aggregated_state)
        
        # 4. 计算轮次指标
        round_metrics = self._compute_round_metrics(round_number, selected_clients, client_weights)
        
        # 5. 保存检查点
        self.save_checkpoint(round_number, round_metrics)
        
        return round_metrics
        
    def _compute_round_metrics(self, 
                              round_number: int, 
                              clients: List[AbstractClient], 
                              client_weights: List[float]) -> Dict[str, Any]:
        """计算轮次指标"""
        total_samples = sum(client_weights)
        
        metrics = {
            "round": round_number,
            "num_clients": len(clients),
            "total_samples": int(total_samples),
            "avg_samples_per_client": total_samples / len(clients) if clients else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # 可以在子类中扩展更多指标计算
        return metrics
        
    def _compile_final_results(self) -> Dict[str, Any]:
        """编译最终训练结果"""
        return {
            "total_rounds": self._total_rounds,
            "completed_rounds": len(self._round_history),
            "training_duration": self.training_duration,
            "num_clients": self.num_clients,
            "final_metrics": self._round_history[-1] if self._round_history else {},
            "round_history": self._round_history
        }
        
    def _validate_config(self) -> None:
        """验证服务器配置"""
        required_keys = ['federated']
        
        for key in required_keys:
            if key not in self._config:
                raise ServerConfigurationError(f"Missing required config section: {key}")
                
        federated_config = self._config['federated']
        required_federated_keys = ['num_rounds']
        
        for key in required_federated_keys:
            if key not in federated_config:
                raise ServerConfigurationError(f"Missing required federated config key: {key}")
                
        # 验证轮数配置
        num_rounds = federated_config['num_rounds']
        if not isinstance(num_rounds, int) or num_rounds <= 0:
            raise ServerConfigurationError("num_rounds must be a positive integer")
            
    def _initialize(self) -> None:
        """初始化服务器"""
        self._logger.info("Initializing federated server")
        
        # 验证客户端
        if not self._clients:
            raise ServerConfigurationError("Server must have at least one client")
            
        # 记录服务器信息
        self._logger.info(f"Server initialized with {len(self._clients)} clients")
        for client in self._clients:
            self._logger.info(f"  Client {client.client_id}: {client.num_train_samples} samples")
            
    def __str__(self) -> str:
        """字符串表示"""
        return f"Server(clients={len(self._clients)}, round={self._current_round}/{self._total_rounds})"