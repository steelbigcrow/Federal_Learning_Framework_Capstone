"""
Federated Learning Aggregator Implementation

This module provides the concrete implementation of the federated learning aggregator
using OOP design patterns while maintaining backward compatibility with existing code.
"""

from typing import Dict, List, Set, Optional, Any
import torch

from ...core.base.aggregator import AbstractAggregator, AggregationMode
from ...core.exceptions.exceptions import AggregationError


class FederatedAggregator(AbstractAggregator):
    """
    联邦学习聚合器实现
    
    基于OOP设计模式重构的聚合器实现，提供：
    1. 标准化的模型聚合接口
    2. 支持多种聚合算法（FedAvg, LoRA-FedAvg, AdaLoRA-FedAvg）
    3. 完整的参数过滤和验证
    4. 向后兼容的API接口
    
    设计原则：
    - 单一职责：专注于模型聚合逻辑
    - 开闭原则：易于扩展新的聚合算法
    - 依赖倒置：依赖抽象的聚合模式接口
    """
    
    def __init__(self, 
                 aggregation_mode: AggregationMode = AggregationMode.WEIGHTED_AVERAGE,
                 config: Optional[Dict[str, Any]] = None,
                 parameter_filter: Optional[Set[str]] = None,
                 **kwargs):
        """
        初始化联邦学习聚合器
        
        Args:
            aggregation_mode: 聚合模式
            config: 聚合器配置
            parameter_filter: 参数过滤器，指定只聚合哪些参数
            **kwargs: 其他参数
        """
        self._parameter_filter = parameter_filter
        self._trainable_keys_cache: Optional[Set[str]] = None
        
        super().__init__(
            aggregation_mode=aggregation_mode,
            config=config or {},
            **kwargs
        )
        
    @property
    def parameter_filter(self) -> Optional[Set[str]]:
        """获取参数过滤器"""
        return self._parameter_filter
        
    def set_parameter_filter(self, parameter_filter: Set[str]) -> None:
        """
        设置参数过滤器
        
        Args:
            parameter_filter: 要过滤的参数名集合
        """
        self._parameter_filter = parameter_filter
        self._trainable_keys_cache = None  # 清除缓存
        self._logger.info(f"Parameter filter updated with {len(parameter_filter)} parameters")
        
    def aggregate(self, 
                  client_models: List[Dict[str, torch.Tensor]],
                  client_weights: List[float],
                  metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            metadata: 客户端元数据列表
            
        Returns:
            聚合后的全局模型状态字典
            
        Raises:
            AggregationError: 当聚合过程失败时
        """
        try:
            # 根据聚合模式选择聚合算法
            if self._aggregation_mode == AggregationMode.WEIGHTED_AVERAGE:
                return self._weighted_average(client_models, client_weights)
            elif self._aggregation_mode == AggregationMode.LORA_WEIGHTED_AVERAGE:
                return self._lora_weighted_average(client_models, client_weights)
            elif self._aggregation_mode == AggregationMode.ADALORA_WEIGHTED_AVERAGE:
                return self._adalora_weighted_average(client_models, client_weights)
            else:
                raise AggregationError(f"Unsupported aggregation mode: {self._aggregation_mode}")
                
        except Exception as e:
            raise AggregationError(f"Aggregation failed: {e}") from e
            
    def compute_aggregation_weights(self, 
                                   client_weights: List[float],
                                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[float]:
        """
        计算聚合权重
        
        Args:
            client_weights: 客户端原始权重
            metadata: 客户端元数据
            
        Returns:
            标准化的聚合权重列表，和为1.0
        """
        if not client_weights:
            return []
            
        # 转换为float类型
        weights = [float(w) for w in client_weights]
        
        # 计算总和
        total = sum(weights)
        
        if total == 0:
            # 如果总和为0，使用平均权重
            return [1.0 / len(weights)] * len(weights)
            
        # 标准化权重
        normalized_weights = [w / total for w in weights]
        
        return normalized_weights
        
    def get_trainable_keys(self, reference_model: Dict[str, torch.Tensor]) -> Set[str]:
        """
        获取模型中可训练参数的键名
        
        Args:
            reference_model: 参考模型状态字典
            
        Returns:
            可训练参数的键名集合
        """
        if self._trainable_keys_cache is not None:
            return self._trainable_keys_cache
            
        trainable_keys = set()
        
        for key, tensor in reference_model.items():
            # 检查参数是否需要梯度（通过判断是否为LoRA参数或其他可训练参数）
            if self._is_trainable_parameter(key, tensor):
                trainable_keys.add(key)
                
        # 缓存结果
        self._trainable_keys_cache = trainable_keys
        
        return trainable_keys
        
    def _is_trainable_parameter(self, key: str, tensor: torch.Tensor) -> bool:
        """
        判断参数是否为可训练参数
        
        Args:
            key: 参数名
            tensor: 参数张量
            
        Returns:
            是否为可训练参数
        """
        # LoRA参数判断
        lora_patterns = ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']
        
        # 分类头参数判断
        classifier_patterns = ['classifier', 'head', 'output']
        
        # AdaLoRA参数判断
        adalora_patterns = ['adalora', 'svd', 'orthogonal']
        
        key_lower = key.lower()
        
        # 检查是否为LoRA参数
        for pattern in lora_patterns:
            if pattern in key_lower:
                return True
                
        # 检查是否为分类头参数
        for pattern in classifier_patterns:
            if pattern in key_lower:
                return True
                
        # 检查是否为AdaLoRA参数
        for pattern in adalora_patterns:
            if pattern in key_lower:
                return True
                
        # 默认情况下，所有参数都视为可训练（除非被过滤器排除）
        if self._parameter_filter:
            return key in self._parameter_filter
            
        return True
        
    def _weighted_average(self, 
                         client_models: List[Dict[str, torch.Tensor]],
                         client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        标准加权平均聚合
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重列表
            
        Returns:
            聚合后的模型状态字典
        """
        if len(client_models) != len(client_weights) or not client_models:
            raise AggregationError("Invalid input for weighted average aggregation")
            
        total = float(sum(client_weights))
        new_state: Dict[str, torch.Tensor] = {}
        
        # 获取要聚合的参数键
        if self._parameter_filter:
            keys_to_aggregate = self._parameter_filter
        else:
            keys_to_aggregate = set(client_models[0].keys())
            
        # 对每个参数进行加权平均
        for key in keys_to_aggregate:
            if key not in client_models[0]:
                continue
                
            acc = None
            for model_state, weight in zip(client_models, client_weights):
                w = weight / total
                if acc is None:
                    acc = model_state[key].detach() * w
                else:
                    acc += model_state[key].detach() * w
                    
            new_state[key] = acc
            
        # 对于未被聚合的参数，使用第一个客户端的值
        if not self._parameter_filter:
            for key in client_models[0].keys():
                if key not in new_state:
                    new_state[key] = client_models[0][key].detach().clone()
                    
        return new_state
        
    def _lora_weighted_average(self, 
                              client_models: List[Dict[str, torch.Tensor]],
                              client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        LoRA专用加权平均聚合
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重列表
            
        Returns:
            聚合后的模型状态字典
        """
        if not client_models:
            raise AggregationError("No client models provided for LoRA aggregation")
            
        # 获取可训练参数
        trainable_keys = self.get_trainable_keys(client_models[0])
        
        total = float(sum(client_weights))
        new_state: Dict[str, torch.Tensor] = {}
        
        # 只聚合可训练的权重
        for key in client_models[0].keys():
            if key in trainable_keys:
                acc = None
                for model_state, weight in zip(client_models, client_weights):
                    w = weight / total
                    if acc is None:
                        acc = model_state[key].detach() * w
                    else:
                        acc += model_state[key].detach() * w
                new_state[key] = acc
            else:
                # 对于非可训练权重，直接使用第一个客户端的权重
                new_state[key] = client_models[0][key].detach().clone()
                
        return new_state
        
    def _adalora_weighted_average(self, 
                                client_models: List[Dict[str, torch.Tensor]],
                                client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        AdaLoRA专用加权平均聚合
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重列表
            
        Returns:
            聚合后的模型状态字典
        """
        if not client_models:
            raise AggregationError("No client models provided for AdaLoRA aggregation")
            
        # 获取可训练参数
        trainable_keys = self.get_trainable_keys(client_models[0])
        
        total = float(sum(client_weights))
        new_state: Dict[str, torch.Tensor] = {}
        
        # 只聚合可训练的权重
        for key in trainable_keys:
            if key in client_models[0]:
                acc = None
                for model_state, weight in zip(client_models, client_weights):
                    w = weight / total
                    if acc is None:
                        acc = model_state[key].detach() * w
                    else:
                        acc += model_state[key].detach() * w
                new_state[key] = acc
                
        return new_state
        
    def _validate_config(self) -> None:
        """验证聚合器配置"""
        # 可以添加特定的配置验证逻辑
        pass
        
    def _initialize(self) -> None:
        """初始化聚合器"""
        self._logger.info(f"Initializing aggregator with mode: {self._aggregation_mode.value}")
        if self._parameter_filter:
            self._logger.info(f"Parameter filter contains {len(self._parameter_filter)} parameters")
            
    # 向后兼容的静态方法
    @staticmethod
    def fedavg(state_dicts: List[Dict[str, torch.Tensor]], num_samples: List[int]) -> Dict[str, torch.Tensor]:
        """
        标准联邦平均算法（向后兼容）
        
        Args:
            state_dicts: 客户端模型状态字典列表
            num_samples: 各客户端样本数量列表
            
        Returns:
            聚合后的全局模型状态字典
        """
        aggregator = FederatedAggregator()
        return aggregator._weighted_average(state_dicts, num_samples)
        
    @staticmethod
    def lora_fedavg(lora_state_dicts: List[Dict[str, torch.Tensor]], 
                   num_samples: List[int], 
                   trainable_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
        """
        LoRA专用联邦平均算法（向后兼容）
        
        Args:
            lora_state_dicts: LoRA模型状态字典列表
            num_samples: 各客户端样本数量列表
            trainable_keys: 可训练参数的键名集合
            
        Returns:
            聚合后的LoRA模型状态字典
        """
        aggregator = FederatedAggregator(parameter_filter=trainable_keys)
        return aggregator._lora_weighted_average(lora_state_dicts, num_samples)
        
    @staticmethod
    def adalora_fedavg(adalora_state_dicts: List[Dict[str, torch.Tensor]], 
                      num_samples: List[int], 
                      trainable_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
        """
        AdaLoRA专用联邦平均算法（向后兼容）
        
        Args:
            adalora_state_dicts: AdaLoRA模型状态字典列表
            num_samples: 各客户端样本数量列表
            trainable_keys: 可训练参数的键名集合
            
        Returns:
            聚合后的AdaLoRA模型状态字典
        """
        aggregator = FederatedAggregator(parameter_filter=trainable_keys)
        return aggregator._adalora_weighted_average(adalora_state_dicts, num_samples)
        
    @staticmethod
    def get_trainable_keys_from_model(model: torch.nn.Module) -> Set[str]:
        """
        从模型获取可训练参数键名（向后兼容）
        
        Args:
            model: PyTorch模型
            
        Returns:
            可训练参数的键名集合
        """
        trainable_keys = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_keys.add(name)
        return trainable_keys