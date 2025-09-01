"""
LoRA联邦平均策略

实现LoRA参数的联邦平均聚合算法。
"""

from typing import Dict, List, Set, Optional, Any
import torch

from ...core.interfaces.strategy import AggregationStrategyInterface, StrategyType, strategy_register
from ...core.exceptions.exceptions import StrategyError


@strategy_register(StrategyType.AGGREGATION, "lora_fedavg")
class LoRAFedAvgStrategy(AggregationStrategyInterface):
    """
    LoRA联邦平均策略
    
    专门用于LoRA参数的聚合，只聚合可训练的LoRA权重和分类头。
    """
    
    def __init__(self):
        self.name = "lora_fedavg"
        self.description = "LoRA联邦平均算法，只聚合可训练的LoRA权重和分类头"
    
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
        聚合LoRA客户端模型
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            global_model: 当前全局模型（可选）
            trainable_keys: 可训练参数的键名集合
            
        Returns:
            聚合后的LoRA模型状态字典
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
            keys_to_aggregate = set(client_models[0].keys())
        else:
            keys_to_aggregate = trainable_keys
        
        # 执行LoRA联邦平均
        aggregated_model = {}
        first_model = client_models[0]
        
        for param_name in first_model.keys():
            if param_name in keys_to_aggregate:
                # 聚合可训练的权重
                param_sum = None
                
                for i, client_model in enumerate(client_models):
                    weight = normalized_weights[i]
                    
                    if param_name not in client_model:
                        raise StrategyError(f"客户端模型缺少参数: {param_name}")
                    
                    param_tensor = client_model[param_name].detach()
                    
                    if param_sum is None:
                        param_sum = param_tensor * weight
                    else:
                        param_sum += param_tensor * weight
                
                aggregated_model[param_name] = param_sum
            else:
                # 对于非可训练权重，直接使用第一个客户端的权重（应该都相同）
                aggregated_model[param_name] = first_model[param_name].detach().clone()
        
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
                "filter_parameters": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否过滤只聚合可训练参数"
                }
            }
        }
    
    @staticmethod
    def get_trainable_keys(model: torch.nn.Module) -> Set[str]:
        """
        获取模型中所有可训练参数的键名
        
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