"""
联邦学习客户端抽象基类

定义了联邦学习客户端的标准接口和行为规范。
"""

from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.utils.data import DataLoader

from .component import FederatedComponent, ComponentStatus
from ..exceptions.exceptions import ClientConfigurationError, ClientTrainingError


class AbstractClient(FederatedComponent):
    """
    联邦学习客户端抽象基类
    
    定义了客户端在联邦学习中的核心职责：
    1. 接收全局模型参数
    2. 执行本地训练
    3. 返回更新后的模型参数和训练指标
    4. 管理本地数据集
    
    设计原则：
    - 单一职责：专注于本地训练和数据管理
    - 开闭原则：通过继承支持不同训练策略
    - 依赖倒置：依赖抽象的训练策略接口
    """
    
    def __init__(self, 
                 client_id: int,
                 train_data_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = "cpu",
                 **kwargs):
        """
        初始化联邦学习客户端
        
        Args:
            client_id: 客户端唯一标识
            train_data_loader: 训练数据加载器
            config: 客户端配置
            device: 计算设备
            **kwargs: 其他参数传递给父类
        """
        # 存储客户端特定属性
        self._client_id = client_id
        self._device = device
        self._train_data_loader = train_data_loader
        self._test_data_loader: Optional[DataLoader] = None
        
        # 训练相关属性
        self._current_model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._training_metrics: Dict[str, Any] = {}
        self._round_number = 0
        
        # 调用父类初始化
        super().__init__(
            component_id=f"client_{client_id}",
            config=config,
            **kwargs
        )
        
    @property
    def client_id(self) -> int:
        """获取客户端ID"""
        return self._client_id
        
    @property
    def device(self) -> str:
        """获取计算设备"""
        return self._device
        
    @property
    def train_data_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self._train_data_loader
        
    @property
    def test_data_loader(self) -> Optional[DataLoader]:
        """获取测试数据加载器"""
        return self._test_data_loader
        
    @property
    def current_round(self) -> int:
        """获取当前轮次"""
        return self._round_number
        
    @property
    def num_train_samples(self) -> int:
        """获取训练样本数量"""
        return len(self._train_data_loader.dataset) if self._train_data_loader else 0
        
    @property
    def num_test_samples(self) -> int:
        """获取测试样本数量"""
        return len(self._test_data_loader.dataset) if self._test_data_loader else 0
        
    def set_test_data_loader(self, test_data_loader: DataLoader) -> None:
        """设置测试数据加载器"""
        self._test_data_loader = test_data_loader
        
    def get_data_distribution_info(self) -> Dict[str, Any]:
        """
        获取数据分布信息
        
        Returns:
            包含数据分布统计的字典
        """
        info = {
            "train_samples": self.num_train_samples,
            "test_samples": self.num_test_samples,
            "total_samples": self.num_train_samples + self.num_test_samples
        }
        
        # 尝试获取标签分布（如果数据集支持）
        try:
            train_labels = self._extract_labels_from_dataloader(self._train_data_loader)
            if train_labels:
                info["train_label_distribution"] = self._compute_label_distribution(train_labels)
                
            if self._test_data_loader:
                test_labels = self._extract_labels_from_dataloader(self._test_data_loader)
                if test_labels:
                    info["test_label_distribution"] = self._compute_label_distribution(test_labels)
        except Exception as e:
            self._logger.warning(f"Could not compute label distribution: {e}")
            
        return info
        
    @abstractmethod
    def receive_global_model(self, global_model_state: Dict[str, torch.Tensor]) -> None:
        """
        接收全局模型参数
        
        Args:
            global_model_state: 全局模型的状态字典
            
        Raises:
            ClientConfigurationError: 当模型配置不兼容时
        """
        pass
        
    @abstractmethod
    def local_train(self, num_epochs: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
        """
        执行本地训练
        
        Args:
            num_epochs: 本地训练轮数
            
        Returns:
            Tuple[模型状态字典, 训练指标字典, 训练样本数量]
            
        Raises:
            ClientTrainingError: 当训练过程出现错误时
        """
        pass
        
    @abstractmethod
    def local_evaluate(self, model_state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        执行本地评估
        
        Args:
            model_state: 要评估的模型状态，如果为None则评估当前模型
            
        Returns:
            评估指标字典
            
        Raises:
            ClientTrainingError: 当评估过程出现错误时
        """
        pass
        
    def local_train_and_evaluate(self, 
                                 global_model_state: Dict[str, torch.Tensor],
                                 num_epochs: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
        """
        执行完整的本地训练和评估流程
        
        这是一个模板方法，定义了标准的训练评估流程：
        1. 接收全局模型
        2. 执行本地训练
        3. 执行本地评估
        4. 返回结果
        
        Args:
            global_model_state: 全局模型状态
            num_epochs: 训练轮数
            
        Returns:
            Tuple[模型状态字典, 综合指标字典, 训练样本数量]
        """
        try:
            self._set_status(ComponentStatus.RUNNING)
            self._round_number += 1
            
            # 1. 接收全局模型
            self.receive_global_model(global_model_state)
            
            # 2. 执行本地训练
            model_state, train_metrics, num_samples = self.local_train(num_epochs)
            
            # 3. 执行本地评估
            eval_metrics = self.local_evaluate(model_state)
            
            # 4. 合并指标
            combined_metrics = {
                **train_metrics,
                **eval_metrics,
                "round": self._round_number,
                "client_id": self._client_id,
                "num_samples": num_samples,
                "epochs": num_epochs
            }
            
            self._training_metrics = combined_metrics
            self._set_status(ComponentStatus.READY)
            
            return model_state, combined_metrics, num_samples
            
        except Exception as e:
            self._set_status(ComponentStatus.ERROR)
            self._logger.error(f"Training and evaluation failed: {e}")
            raise ClientTrainingError(f"Client {self._client_id} training failed: {e}") from e
            
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        获取训练历史
        
        Returns:
            训练历史记录列表
        """
        return getattr(self, '_training_history', [])
        
    def reset(self) -> None:
        """重置客户端状态"""
        self._current_model = None
        self._optimizer = None
        self._training_metrics = {}
        self._round_number = 0
        self._set_status(ComponentStatus.READY)
        
    def _validate_config(self) -> None:
        """验证客户端配置"""
        required_keys = ['optimizer', 'device']
        
        for key in required_keys:
            if key not in self._config:
                raise ClientConfigurationError(f"Missing required config key: {key}")
                
        # 验证优化器配置
        optimizer_config = self._config.get('optimizer', {})
        if not isinstance(optimizer_config, dict):
            raise ClientConfigurationError("optimizer config must be a dictionary")
            
        if 'name' not in optimizer_config:
            raise ClientConfigurationError("optimizer config must contain 'name' field")
            
        # 验证设备配置
        device = self._config.get('device', self._device)
        if not isinstance(device, str):
            raise ClientConfigurationError("device must be a string")
            
    def _initialize(self) -> None:
        """初始化客户端"""
        self._logger.info(f"Initializing client {self._client_id}")
        self._set_status(ComponentStatus.READY)
        
        # 记录数据分布信息
        data_info = self.get_data_distribution_info()
        self._logger.info(f"Client {self._client_id} data distribution: {data_info}")
        
    def _extract_labels_from_dataloader(self, data_loader: DataLoader) -> Optional[List]:
        """从数据加载器中提取标签"""
        try:
            labels = []
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    _, y = batch[0], batch[1]
                    if isinstance(y, torch.Tensor):
                        labels.extend(y.cpu().tolist())
                    else:
                        labels.extend(y)
            return labels
        except Exception:
            return None
            
    def _compute_label_distribution(self, labels: List) -> Dict[str, int]:
        """计算标签分布"""
        from collections import Counter
        return dict(Counter(labels))
        
    def __str__(self) -> str:
        """字符串表示"""
        return f"Client(id={self._client_id}, round={self._round_number}, samples={self.num_train_samples})"