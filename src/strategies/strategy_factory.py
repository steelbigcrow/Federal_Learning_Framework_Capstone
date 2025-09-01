"""
策略工厂模块

提供策略的创建、管理和动态选择功能。
"""

import torch
from typing import Dict, Any, Optional, Type, List
from enum import Enum

from ..core.interfaces.strategy import (
    StrategyInterface, 
    StrategyType, 
    StrategyRegistry,
    AggregationStrategyInterface,
    TrainingStrategyInterface
)
from ..core.exceptions.exceptions import StrategyError
from ..strategies import (
    FedAvgStrategy,
    LoRAFedAvgStrategy, 
    AdaLoRAFedAvgStrategy,
    StandardTrainingStrategy,
    LoRATrainingStrategy,
    AdaLoRATrainingStrategy
)


class StrategyFactory:
    """
    策略工厂
    
    负责创建、注册和管理各种策略实例。
    """
    
    def __init__(self):
        """初始化策略工厂"""
        self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """注册默认策略"""
        # 聚合策略
        self.register_strategy(StrategyType.AGGREGATION, "fedavg", FedAvgStrategy())
        self.register_strategy(StrategyType.AGGREGATION, "lora_fedavg", LoRAFedAvgStrategy())
        self.register_strategy(StrategyType.AGGREGATION, "adalora_fedavg", AdaLoRAFedAvgStrategy())
        
        # 训练策略
        self.register_strategy(StrategyType.TRAINING, "standard", StandardTrainingStrategy())
        self.register_strategy(StrategyType.TRAINING, "lora", LoRATrainingStrategy())
        self.register_strategy(StrategyType.TRAINING, "adalora", AdaLoRATrainingStrategy())
    
    def register_strategy(self, 
                         strategy_type: StrategyType, 
                         strategy_name: str, 
                         strategy: StrategyInterface) -> None:
        """
        注册策略
        
        Args:
            strategy_type: 策略类型
            strategy_name: 策略名称
            strategy: 策略实例
        """
        StrategyRegistry.register_strategy(strategy_type, strategy_name, strategy)
    
    def create_strategy(self, 
                        strategy_type: StrategyType, 
                        strategy_name: str, 
                        config: Optional[Dict[str, Any]] = None) -> StrategyInterface:
        """
        创建策略实例
        
        Args:
            strategy_type: 策略类型
            strategy_name: 策略名称
            config: 策略配置
            
        Returns:
            策略实例
            
        Raises:
            StrategyError: 当策略不存在或创建失败时
        """
        strategy = StrategyRegistry.get_strategy(strategy_type, strategy_name)
        
        if strategy is None:
            available_strategies = StrategyRegistry.list_strategies(strategy_type)
            raise StrategyError(
                f"策略 '{strategy_name}' 不存在于类型 '{strategy_type.value}' 中。"
                f"可用策略: {available_strategies}"
            )
        
        # 如果提供了配置，可以在这里进行策略的配置
        if config is not None:
            self._configure_strategy(strategy, config)
        
        return strategy
    
    def create_aggregation_strategy(self, 
                                  strategy_name: str, 
                                  config: Optional[Dict[str, Any]] = None) -> AggregationStrategyInterface:
        """
        创建聚合策略
        
        Args:
            strategy_name: 策略名称
            config: 策略配置
            
        Returns:
            聚合策略实例
        """
        try:
            strategy = self.create_strategy(StrategyType.AGGREGATION, strategy_name, config)
            
            if not isinstance(strategy, AggregationStrategyInterface):
                raise StrategyError(f"策略 '{strategy_name}' 不是聚合策略")
            
            return strategy
        except StrategyError:
            # 重新抛出更具体的错误信息
            available = self.get_available_strategies(StrategyType.AGGREGATION)
            raise StrategyError(
                f"无法创建聚合策略 '{strategy_name}'。可用聚合策略: {available}"
            )
    
    def create_training_strategy(self, 
                               strategy_name: str, 
                               config: Optional[Dict[str, Any]] = None) -> TrainingStrategyInterface:
        """
        创建训练策略
        
        Args:
            strategy_name: 策略名称
            config: 策略配置
            
        Returns:
            训练策略实例
        """
        try:
            strategy = self.create_strategy(StrategyType.TRAINING, strategy_name, config)
            
            if not isinstance(strategy, TrainingStrategyInterface):
                raise StrategyError(f"策略 '{strategy_name}' 不是训练策略")
            
            return strategy
        except StrategyError:
            # 重新抛出更具体的错误信息
            available = self.get_available_strategies(StrategyType.TRAINING)
            raise StrategyError(
                f"无法创建训练策略 '{strategy_name}'。可用训练策略: {available}"
            )
    
    def get_available_strategies(self, strategy_type: StrategyType) -> List[str]:
        """
        获取指定类型的可用策略列表
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            策略名称列表
        """
        return StrategyRegistry.list_strategies(strategy_type)
    
    def get_strategy_info(self, 
                         strategy_type: StrategyType, 
                         strategy_name: str) -> Dict[str, Any]:
        """
        获取策略信息
        
        Args:
            strategy_type: 策略类型
            strategy_name: 策略名称
            
        Returns:
            策略信息字典
        """
        strategy = StrategyRegistry.get_strategy(strategy_type, strategy_name)
        
        if strategy is None:
            raise StrategyError(f"策略 '{strategy_name}' 不存在")
        
        return {
            "name": strategy.get_name(),
            "description": strategy.get_description(),
            "type": strategy_type.value,
            "config_schema": strategy.get_config_schema()
        }
    
    def validate_strategy_config(self, 
                                strategy_type: StrategyType, 
                                strategy_name: str, 
                                config: Dict[str, Any]) -> bool:
        """
        验证策略配置
        
        Args:
            strategy_type: 策略类型
            strategy_name: 策略名称
            config: 配置字典
            
        Returns:
            验证结果
        """
        try:
            # 对于不同类型的策略，需要不同的上下文结构
            if strategy_type == StrategyType.AGGREGATION:
                # 聚合策略需要特定的上下文，但允许覆盖默认值
                context = {
                    'client_models': config.get('client_models', [{'param': torch.tensor([1.0])}]),
                    'client_weights': config.get('client_weights', [1.0]),
                }
                # 添加其他配置项
                for key, value in config.items():
                    if key not in ['client_models', 'client_weights']:
                        context[key] = value
            elif strategy_type == StrategyType.TRAINING:
                # 训练策略应该直接使用提供的配置
                context = config
            else:
                # 其他策略类型直接使用配置
                context = config
            
            strategy = self.create_strategy(strategy_type, strategy_name)
            strategy.validate_context(context)
            return True
        except (StrategyError, KeyError, TypeError, ImportError):
            return False
    
    def _configure_strategy(self, strategy: StrategyInterface, config: Dict[str, Any]) -> None:
        """
        配置策略
        
        Args:
            strategy: 策略实例
            config: 配置字典
        """
        # 这里可以根据具体的策略类型进行配置
        # 目前是简单实现，后续可以扩展
        pass
    
    def list_all_strategies(self) -> Dict[str, List[str]]:
        """
        列出所有策略
        
        Returns:
            按类型分组的策略字典
        """
        result = {}
        for strategy_type in StrategyType:
            strategies = self.get_available_strategies(strategy_type)
            if strategies:
                result[strategy_type.value] = strategies
        return result
    
    def get_default_strategy(self, strategy_type: StrategyType) -> str:
        """
        获取默认策略名称
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            默认策略名称
        """
        defaults = {
            StrategyType.AGGREGATION: "fedavg",
            StrategyType.TRAINING: "standard"
        }
        
        return defaults.get(strategy_type, "default")
    
    def clear_registry(self) -> None:
        """清空策略注册表"""
        StrategyRegistry.clear()
        # 重新注册默认策略以确保基本功能可用
        self._register_default_strategies()
    
    def reload_strategies(self) -> None:
        """重新加载所有策略"""
        self.clear_registry()
        self._register_default_strategies()


# 全局策略工厂实例
strategy_factory = StrategyFactory()