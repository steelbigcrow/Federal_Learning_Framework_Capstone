"""
策略接口定义

定义了策略模式的标准接口，用于实现可插拔的算法和策略。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic
import torch
from enum import Enum


T = TypeVar('T')  # 泛型类型变量


class StrategyType(Enum):
    """策略类型枚举"""
    AGGREGATION = "aggregation"
    TRAINING = "training"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    DATA_PARTITION = "data_partition"
    CLIENT_SELECTION = "client_selection"


class StrategyInterface(ABC):
    """
    策略接口基类
    
    定义了所有策略类必须实现的基本接口。
    
    设计原则：
    - 策略模式：封装算法并使其可互换
    - 单一职责：每个策略专注于特定算法
    - 开闭原则：通过继承添加新策略
    """
    
    @abstractmethod
    def execute(self, context: Dict[str, Any], **kwargs) -> Any:
        """
        执行策略
        
        Args:
            context: 策略执行上下文
            **kwargs: 额外参数
            
        Returns:
            策略执行结果
            
        Raises:
            StrategyError: 当策略执行失败时
        """
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """获取策略描述"""
        pass
        
    @abstractmethod
    def validate_context(self, context: Dict[str, Any]) -> None:
        """
        验证策略执行上下文
        
        Args:
            context: 策略执行上下文
            
        Raises:
            StrategyError: 当上下文无效时
        """
        pass
        
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取策略配置模式
        
        Returns:
            配置模式字典，用于验证策略配置
        """
        return {}


class AggregationStrategyInterface(StrategyInterface):
    """
    聚合策略接口
    
    专门用于模型参数聚合的策略接口。
    """
    
    @abstractmethod
    def aggregate_models(self, 
                        client_models: List[Dict[str, torch.Tensor]],
                        client_weights: List[float],
                        global_model: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            global_model: 当前全局模型（可选）
            
        Returns:
            聚合后的全局模型状态字典
        """
        pass
        
    @abstractmethod
    def compute_weights(self, 
                       client_metrics: List[Dict[str, Any]]) -> List[float]:
        """
        计算客户端聚合权重
        
        Args:
            client_metrics: 客户端指标列表
            
        Returns:
            标准化的权重列表
        """
        pass
        
    def supports_parameter_filtering(self) -> bool:
        """是否支持参数过滤"""
        return False
        
    def get_required_metrics(self) -> List[str]:
        """获取策略所需的客户端指标"""
        return ["num_samples"]


class TrainingStrategyInterface(StrategyInterface):
    """
    训练策略接口
    
    专门用于客户端本地训练的策略接口。
    """
    
    @abstractmethod
    def train_model(self, 
                    model: torch.nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    config: Dict[str, Any]) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            config: 训练配置
            
        Returns:
            训练指标字典
        """
        pass
        
    @abstractmethod
    def evaluate_model(self, 
                       model: torch.nn.Module,
                       test_loader: torch.utils.data.DataLoader,
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            config: 评估配置
            
        Returns:
            评估指标字典
        """
        pass
        
    def prepare_model(self, 
                      model: torch.nn.Module, 
                      config: Dict[str, Any]) -> torch.nn.Module:
        """
        准备模型（如应用LoRA等）
        
        Args:
            model: 原始模型
            config: 准备配置
            
        Returns:
            准备后的模型
        """
        return model
        
    def get_optimizer(self, 
                     model: torch.nn.Module, 
                     config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        获取优化器
        
        Args:
            model: 模型
            config: 优化器配置
            
        Returns:
            配置好的优化器
        """
        return torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))


class DataPartitionStrategyInterface(StrategyInterface):
    """
    数据分割策略接口
    
    专门用于数据集分割的策略接口。
    """
    
    @abstractmethod
    def partition_data(self, 
                       dataset: Any,
                       num_clients: int,
                       config: Dict[str, Any]) -> List[Any]:
        """
        分割数据集
        
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            config: 分割配置
            
        Returns:
            分割后的数据集列表
        """
        pass
        
    @abstractmethod
    def get_partition_info(self) -> Dict[str, Any]:
        """
        获取分割信息
        
        Returns:
            分割统计信息
        """
        pass
        
    def is_iid(self) -> bool:
        """是否为IID分割"""
        return True
        
    def get_expected_clients(self) -> Optional[int]:
        """获取期望的客户端数量（如果有固定要求）"""
        return None


class ClientSelectionStrategyInterface(StrategyInterface):
    """
    客户端选择策略接口
    
    专门用于联邦学习轮次中选择参与客户端的策略接口。
    """
    
    @abstractmethod
    def select_clients(self, 
                       available_clients: List[Any],
                       round_number: int,
                       config: Dict[str, Any]) -> List[Any]:
        """
        选择客户端
        
        Args:
            available_clients: 可用客户端列表
            round_number: 当前轮次
            config: 选择配置
            
        Returns:
            被选中的客户端列表
        """
        pass
        
    def get_selection_ratio(self) -> float:
        """获取选择比例"""
        return 1.0
        
    def requires_client_metrics(self) -> bool:
        """是否需要客户端历史指标"""
        return False


class OptimizationStrategyInterface(StrategyInterface):
    """
    优化策略接口
    
    专门用于优化算法的策略接口。
    """
    
    @abstractmethod
    def create_optimizer(self, 
                         model_parameters: Any,
                         config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        创建优化器
        
        Args:
            model_parameters: 模型参数
            config: 优化器配置
            
        Returns:
            优化器实例
        """
        pass
        
    def get_supported_optimizers(self) -> List[str]:
        """获取支持的优化器列表"""
        return ["adam", "sgd", "adamw"]


class EvaluationStrategyInterface(StrategyInterface):
    """
    评估策略接口
    
    专门用于模型评估的策略接口。
    """
    
    @abstractmethod
    def evaluate(self, 
                 model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 device: str) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            dataloader: 数据加载器
            device: 计算设备
            
        Returns:
            评估指标字典
        """
        pass
        
    def get_metric_names(self) -> List[str]:
        """获取评估指标名称列表"""
        return ["accuracy", "loss"]
        
    def requires_ground_truth(self) -> bool:
        """是否需要真实标签"""
        return True


class StrategyRegistry:
    """
    策略注册表
    
    用于管理和发现所有可用的策略实现。
    """
    
    _strategies: Dict[StrategyType, Dict[str, StrategyInterface]] = {}
    
    @classmethod
    def register_strategy(cls, 
                         strategy_type: StrategyType, 
                         strategy_name: str,
                         strategy: StrategyInterface) -> None:
        """注册策略"""
        if strategy_type not in cls._strategies:
            cls._strategies[strategy_type] = {}
        cls._strategies[strategy_type][strategy_name] = strategy
        
    @classmethod
    def get_strategy(cls, 
                    strategy_type: StrategyType, 
                    strategy_name: str) -> Optional[StrategyInterface]:
        """获取策略"""
        return cls._strategies.get(strategy_type, {}).get(strategy_name)
        
    @classmethod
    def list_strategies(cls, strategy_type: StrategyType) -> List[str]:
        """列出指定类型的所有策略"""
        return list(cls._strategies.get(strategy_type, {}).keys())
        
    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._strategies.clear()


# 策略注册装饰器
def strategy_register(strategy_type: StrategyType, strategy_name: str):
    """
    策略注册装饰器
    
    用于自动注册策略实现。
    
    Args:
        strategy_type: 策略类型
        strategy_name: 策略名称
    
    Example:
        @strategy_register(StrategyType.AGGREGATION, "fedavg")
        class FedAvgStrategy(AggregationStrategyInterface):
            pass
    """
    def decorator(cls):
        # 注册策略实例
        strategy_instance = cls()
        StrategyRegistry.register_strategy(strategy_type, strategy_name, strategy_instance)
        cls._strategy_type = strategy_type
        cls._strategy_name = strategy_name
        return cls
    return decorator