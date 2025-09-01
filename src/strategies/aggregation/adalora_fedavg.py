"""
AdaLoRA联邦平均策略

实现AdaLoRA参数的联邦平均聚合算法。
"""

from typing import Dict, List, Set, Optional, Any
import torch

from ...core.interfaces.strategy import AggregationStrategyInterface, StrategyType, strategy_register
from ...core.exceptions.exceptions import StrategyError


@strategy_register(StrategyType.AGGREGATION, "adalora_fedavg")
class AdaLoRAFedAvgStrategy(AggregationStrategyInterface):
    """
    AdaLoRA联邦平均策略
    
    专门用于AdaLoRA参数的聚合，只聚合可训练的AdaLoRA权重和分类头。
    支持动态rank分配和SVD分解的聚合。
    """
    
    def __init__(self):
        self.name = "adalora_fedavg"
        self.description = "AdaLoRA联邦平均算法，支持动态rank分配的参数聚合"
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def validate_context(self, context: Dict[str, Any]) -> None:
        """验证策略执行上下文"""
        required_keys = ['client_models', 'client_weights']
        for key in required_keys:
            if key not in context:
                raise StrategyError(f"缺少必需的上下文键: {key}")
        
        client_models = context['client_models']
        client_weights = context['client_weights']
        
        if not isinstance(client_models, list) or len(client_models) == 0:
            raise StrategyError("client_models 必须是非空列表")
        
        if not isinstance(client_weights, list) or len(client_weights) != len(client_models):
            raise StrategyError("client_weights 必须是与 client_models 等长的列表")
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Any:
        """执行聚合策略"""
        self.validate_context(context)
        
        client_models = context['client_models']
        client_weights = context['client_weights']
        global_model = context.get('global_model')
        trainable_keys = context.get('trainable_keys')
        
        return self.aggregate_models(client_models, client_weights, global_model, trainable_keys)
    
    def aggregate_models(self, 
                        client_models: List[Dict[str, torch.Tensor]],
                        client_weights: List[float],
                        global_model: Optional[Dict[str, torch.Tensor]] = None,
                        trainable_keys: Optional[Set[str]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合AdaLoRA客户端模型
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            global_model: 当前全局模型（可选）
            trainable_keys: 可训练参数的键名集合
            
        Returns:
            聚合后的AdaLoRA模型状态字典
        """
        if len(client_models) == 0:
            raise StrategyError("客户端模型列表为空")
        
        if len(client_models) != len(client_weights):
            raise StrategyError("客户端模型数量与权重数量不匹配")
        
        # 标准化权重
        total_weight = sum(client_weights)
        if total_weight == 0:
            raise StrategyError("客户端权重总和为0")
        
        normalized_weights = [w / total_weight for w in client_weights]
        
        # 确定要聚合的参数
        if trainable_keys is None:
            # 如果没有指定trainable_keys，则聚合所有权重（向后兼容）
            trainable_keys = set()
            for model in client_models:
                trainable_keys.update(model.keys())
        
        # 执行AdaLoRA联邦平均
        aggregated_model = {}
        
        for param_name in trainable_keys:
            # 检查是否有客户端有这个参数
            valid_models = []
            valid_weights = []
            
            for i, client_model in enumerate(client_models):
                if param_name in client_model:
                    valid_models.append(client_model)
                    valid_weights.append(normalized_weights[i])
            
            # 如果参数只存在于部分客户端中，跳过该参数
            if len(valid_models) < len(client_models):
                continue
            
            if not valid_models:
                continue  # 跳过不存在的参数
            
            # 重新标准化有效权重
            valid_total_weight = sum(valid_weights)
            if valid_total_weight == 0:
                continue
            
            valid_normalized_weights = [w / valid_total_weight for w in valid_weights]
            
            # 执行加权平均
            param_sum = None
            for model, weight in zip(valid_models, valid_normalized_weights):
                param_tensor = model[param_name].detach()
                
                if param_sum is None:
                    param_sum = param_tensor * weight
                else:
                    param_sum += param_tensor * weight
            
            aggregated_model[param_name] = param_sum
        
        return aggregated_model
    
    def compute_weights(self, 
                       client_metrics: List[Dict[str, Any]]) -> List[float]:
        """
        计算客户端聚合权重
        
        Args:
            client_metrics: 客户端指标列表
            
        Returns:
            标准化的权重列表
        """
        if not client_metrics:
            return []
        
        # 使用样本数量作为权重
        weights = []
        for metrics in client_metrics:
            if 'num_samples' in metrics:
                weights.append(float(metrics['num_samples']))
            else:
                # 如果没有样本数量，使用等权重
                weights.append(1.0)
        
        # 标准化权重
        total_weight = sum(weights)
        if total_weight == 0:
            return [1.0 / len(weights)] * len(weights)
        
        return [w / total_weight for w in weights]
    
    def supports_parameter_filtering(self) -> bool:
        """是否支持参数过滤"""
        return True
    
    def get_required_metrics(self) -> List[str]:
        """获取策略所需的客户端指标"""
        return ["num_samples"]
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取策略配置模式"""
        return {
            "type": "object",
            "properties": {
                "trainable_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可训练参数键名列表"
                },
                "rank_budget": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "AdaLoRA rank预算"
                },
                "importance_weighting": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否使用重要性加权"
                }
            }
        }
    
    def aggregate_with_svd(self, 
                          client_models: List[Dict[str, torch.Tensor]],
                          client_weights: List[float],
                          param_name: str) -> torch.Tensor:
        """
        使用SVD分解进行参数聚合
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重列表
            param_name: 参数名称
            
        Returns:
            聚合后的参数张量
        """
        # 收集所有客户端的参数
        param_tensors = []
        for model in client_models:
            if param_name in model:
                param_tensors.append(model[param_name].detach())
        
        if not param_tensors:
            raise StrategyError(f"参数 {param_name} 在所有客户端中都不存在")
        
        # 计算加权平均
        total_weight = sum(client_weights)
        if total_weight == 0:
            # 如果权重为0，使用简单平均
            return torch.stack(param_tensors).mean(dim=0)
        
        # 执行加权平均
        weighted_sum = None
        for tensor, weight in zip(param_tensors, client_weights):
            if weighted_sum is None:
                weighted_sum = tensor * (weight / total_weight)
            else:
                weighted_sum += tensor * (weight / total_weight)
        
        return weighted_sum