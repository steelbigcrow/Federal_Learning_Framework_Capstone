"""
联邦学习聚合器抽象基类

定义了联邦学习模型聚合的标准接口和行为规范。
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Set
import torch
from enum import Enum

from .component import FederatedComponent, ComponentStatus
from ..exceptions.exceptions import AggregationError, AggregationConfigurationError


class AggregationMode(Enum):
    """聚合模式枚举"""
    WEIGHTED_AVERAGE = "weighted_average"
    SIMPLE_AVERAGE = "simple_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    FEDPROX = "fedprox"
    CUSTOM = "custom"


class AbstractAggregator(FederatedComponent):
    """
    联邦学习聚合器抽象基类
    
    定义了模型聚合的核心职责：
    1. 接收多个客户端模型
    2. 根据聚合策略计算全局模型
    3. 处理异构模型参数
    4. 支持不同的聚合算法
    5. 提供聚合过程的监控和日志
    
    设计原则：
    - 单一职责：专注于模型聚合逻辑
    - 策略模式：支持多种聚合算法
    - 开闭原则：易于扩展新的聚合策略
    - 依赖倒置：依赖抽象的权重计算接口
    """
    
    def __init__(self, 
                 aggregation_mode: AggregationMode = AggregationMode.WEIGHTED_AVERAGE,
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        初始化聚合器
        
        Args:
            aggregation_mode: 聚合模式
            config: 聚合器配置
            **kwargs: 其他参数传递给父类
        """
        self._aggregation_mode = aggregation_mode
        self._aggregation_count = 0
        self._last_aggregation_time: Optional[float] = None
        self._aggregation_history: List[Dict[str, Any]] = []
        
        # 聚合状态
        self._current_models: List[Dict[str, torch.Tensor]] = []
        self._current_weights: List[float] = []
        self._current_metadata: List[Dict[str, Any]] = []
        
        # 调用父类初始化
        super().__init__(
            component_id=f"aggregator_{aggregation_mode.value}",
            config=config or {},
            **kwargs
        )
        
    @property
    def aggregation_mode(self) -> AggregationMode:
        """获取聚合模式"""
        return self._aggregation_mode
        
    @property
    def aggregation_count(self) -> int:
        """获取聚合次数"""
        return self._aggregation_count
        
    @property
    def last_aggregation_time(self) -> Optional[float]:
        """获取最后一次聚合耗时（秒）"""
        return self._last_aggregation_time
        
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """获取聚合历史记录"""
        return self._aggregation_history.copy()
        
    def set_aggregation_mode(self, mode: AggregationMode) -> None:
        """设置聚合模式"""
        if self._status == ComponentStatus.RUNNING:
            raise AggregationError("Cannot change aggregation mode during aggregation")
            
        self._aggregation_mode = mode
        self._component_id = f"aggregator_{mode.value}"
        self._logger.info(f"Aggregation mode changed to {mode.value}")
        
    @abstractmethod
    def aggregate(self, 
                  client_models: List[Dict[str, torch.Tensor]],
                  client_weights: List[float],
                  metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表（通常是样本数量）
            metadata: 客户端元数据列表（可选）
            
        Returns:
            聚合后的全局模型状态字典
            
        Raises:
            AggregationError: 当聚合过程失败时
        """
        pass
        
    @abstractmethod
    def compute_aggregation_weights(self, 
                                   client_weights: List[float],
                                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[float]:
        """
        计算聚合权重
        
        Args:
            client_weights: 客户端原始权重（如样本数量）
            metadata: 客户端元数据
            
        Returns:
            标准化的聚合权重列表，和为1.0
            
        Raises:
            AggregationError: 当权重计算失败时
        """
        pass
        
    def validate_models(self, client_models: List[Dict[str, torch.Tensor]]) -> None:
        """
        验证客户端模型的兼容性
        
        Args:
            client_models: 客户端模型状态字典列表
            
        Raises:
            AggregationError: 当模型不兼容时
        """
        if not client_models:
            raise AggregationError("No client models provided for aggregation")
            
        if len(client_models) < 2:
            self._logger.warning("Only one client model provided for aggregation")
            return
            
        # 检查所有模型是否有相同的参数结构
        reference_keys = set(client_models[0].keys())
        
        for i, model in enumerate(client_models[1:], 1):
            model_keys = set(model.keys())
            
            if model_keys != reference_keys:
                missing_keys = reference_keys - model_keys
                extra_keys = model_keys - reference_keys
                
                error_msg = f"Model {i} has incompatible structure with reference model (model 0)."
                if missing_keys:
                    error_msg += f" Missing keys: {missing_keys}"
                if extra_keys:
                    error_msg += f" Extra keys: {extra_keys}"
                    
                raise AggregationError(error_msg)
                
            # 检查参数形状
            for key in reference_keys:
                ref_shape = client_models[0][key].shape
                model_shape = model[key].shape
                
                if ref_shape != model_shape:
                    raise AggregationError(
                        f"Parameter '{key}' shape mismatch: "
                        f"reference {ref_shape} vs model {i} {model_shape}"
                    )
                    
    def validate_weights(self, 
                        client_models: List[Dict[str, torch.Tensor]], 
                        client_weights: List[float]) -> None:
        """
        验证权重的有效性
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重列表
            
        Raises:
            AggregationError: 当权重无效时
        """
        if len(client_weights) != len(client_models):
            raise AggregationError(
                f"Number of weights ({len(client_weights)}) does not match "
                f"number of models ({len(client_models)})"
            )
            
        if not all(isinstance(w, (int, float)) and w >= 0 for w in client_weights):
            raise AggregationError("All client weights must be non-negative numbers")
            
        if sum(client_weights) == 0:
            raise AggregationError("Sum of client weights cannot be zero")
            
    def filter_parameters(self, 
                         model_state: Dict[str, torch.Tensor],
                         parameter_filter: Optional[Set[str]] = None) -> Dict[str, torch.Tensor]:
        """
        过滤模型参数
        
        Args:
            model_state: 模型状态字典
            parameter_filter: 要包含的参数名集合，如果为None则包含所有参数
            
        Returns:
            过滤后的模型状态字典
        """
        if parameter_filter is None:
            return model_state
            
        return {k: v for k, v in model_state.items() if k in parameter_filter}
        
    def aggregate_with_validation(self, 
                                 client_models: List[Dict[str, torch.Tensor]],
                                 client_weights: List[float],
                                 metadata: Optional[List[Dict[str, Any]]] = None,
                                 parameter_filter: Optional[Set[str]] = None) -> Dict[str, torch.Tensor]:
        """
        带验证的聚合方法（模板方法）
        
        这是一个模板方法，定义了完整的聚合流程：
        1. 验证输入
        2. 预处理
        3. 执行聚合
        4. 后处理
        5. 记录历史
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            metadata: 客户端元数据列表
            parameter_filter: 参数过滤器
            
        Returns:
            聚合后的全局模型状态字典
        """
        import time
        
        try:
            self._set_status(ComponentStatus.RUNNING)
            start_time = time.time()
            
            # 1. 验证输入
            self.validate_models(client_models)
            self.validate_weights(client_models, client_weights)
            
            # 2. 预处理
            processed_models = client_models
            if parameter_filter:
                processed_models = [
                    self.filter_parameters(model, parameter_filter)
                    for model in client_models
                ]
                
            # 3. 存储当前聚合状态
            self._current_models = processed_models
            self._current_weights = client_weights
            self._current_metadata = metadata or []
            
            # 4. 执行聚合
            aggregated_model = self.aggregate(processed_models, client_weights, metadata)
            
            # 5. 记录聚合历史
            end_time = time.time()
            self._last_aggregation_time = end_time - start_time
            self._aggregation_count += 1
            
            aggregation_record = {
                "round": self._aggregation_count,
                "num_clients": len(client_models),
                "aggregation_time": self._last_aggregation_time,
                "mode": self._aggregation_mode.value,
                "total_weight": sum(client_weights),
                "timestamp": time.time()
            }
            self._aggregation_history.append(aggregation_record)
            
            self._set_status(ComponentStatus.READY)
            
            self._logger.info(
                f"Aggregation {self._aggregation_count} completed: "
                f"{len(client_models)} models, {self._last_aggregation_time:.3f}s"
            )
            
            return aggregated_model
            
        except Exception as e:
            self._set_status(ComponentStatus.ERROR)
            self._logger.error(f"Aggregation failed: {e}")
            raise AggregationError(f"Aggregation process failed: {e}") from e
        finally:
            # 清理当前状态
            self._current_models = []
            self._current_weights = []
            self._current_metadata = []
            
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        if not self._aggregation_history:
            return {"total_aggregations": 0}
            
        times = [record["aggregation_time"] for record in self._aggregation_history]
        
        return {
            "total_aggregations": len(self._aggregation_history),
            "avg_aggregation_time": sum(times) / len(times),
            "min_aggregation_time": min(times),
            "max_aggregation_time": max(times),
            "total_aggregation_time": sum(times)
        }
        
    def reset(self) -> None:
        """重置聚合器状态"""
        self._aggregation_count = 0
        self._last_aggregation_time = None
        self._aggregation_history = []
        self._current_models = []
        self._current_weights = []
        self._current_metadata = []
        self._set_status(ComponentStatus.READY)
        self._logger.info("Aggregator state reset")
        
    def _validate_config(self) -> None:
        """验证聚合器配置"""
        # 可以在子类中添加特定的配置验证
        pass
        
    def _initialize(self) -> None:
        """初始化聚合器"""
        self._logger.info(f"Initializing aggregator with mode: {self._aggregation_mode.value}")
        self._set_status(ComponentStatus.READY)
        
    def __str__(self) -> str:
        """字符串表示"""
        return f"Aggregator(mode={self._aggregation_mode.value}, count={self._aggregation_count})"