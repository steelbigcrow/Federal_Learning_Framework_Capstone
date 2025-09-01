"""
策略管理器模块

提供策略的高级管理功能，包括策略发现、配置管理、性能监控等。
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import time
from datetime import datetime
import json

from ..core.interfaces.strategy import StrategyType, StrategyInterface
from ..core.exceptions.exceptions import StrategyError
from .strategy_factory import StrategyFactory


@dataclass
class StrategyPerformanceMetrics:
    """策略性能指标"""
    execution_count: int = 0
    total_execution_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_execution_time: Optional[float] = None
    last_error: Optional[str] = None
    
    @property
    def average_execution_time(self) -> float:
        """平均执行时间"""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


@dataclass
class StrategyConfiguration:
    """策略配置"""
    strategy_type: StrategyType
    strategy_name: str
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 0
    description: Optional[str] = None


class StrategyManager:
    """
    策略管理器
    
    提供策略的高级管理功能，包括：
    - 策略发现和注册
    - 配置管理
    - 性能监控
    - 策略推荐
    - 历史记录
    """
    
    def __init__(self):
        """初始化策略管理器"""
        self.factory = StrategyFactory()
        self.performance_metrics: Dict[str, StrategyPerformanceMetrics] = {}
        self.strategy_configs: Dict[str, StrategyConfiguration] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # 初始化所有策略的性能指标
        self._initialize_performance_metrics()
    
    def _initialize_performance_metrics(self) -> None:
        """初始化所有策略的性能指标"""
        for strategy_type in StrategyType:
            strategies = self.factory.get_available_strategies(strategy_type)
            for strategy_name in strategies:
                key = self._get_strategy_key(strategy_type, strategy_name)
                self.performance_metrics[key] = StrategyPerformanceMetrics()
    
    def _get_strategy_key(self, strategy_type: StrategyType, strategy_name: str) -> str:
        """获取策略唯一键"""
        return f"{strategy_type.value}:{strategy_name}"
    
    def register_strategy_config(self, 
                               strategy_type: StrategyType, 
                               strategy_name: str, 
                               config: Dict[str, Any],
                               enabled: bool = True,
                               priority: int = 0,
                               description: Optional[str] = None) -> None:
        """
        注册策略配置
        
        Args:
            strategy_type: 策略类型
            strategy_name: 策略名称
            config: 策略配置
            enabled: 是否启用
            priority: 优先级
            description: 配置描述
        """
        key = self._get_strategy_key(strategy_type, strategy_name)
        strategy_config = StrategyConfiguration(
            strategy_type=strategy_type,
            strategy_name=strategy_name,
            config=config,
            enabled=enabled,
            priority=priority,
            description=description
        )
        self.strategy_configs[key] = strategy_config
    
    def get_strategy_config(self, strategy_type: StrategyType, strategy_name: str) -> Optional[StrategyConfiguration]:
        """获取策略配置"""
        key = self._get_strategy_key(strategy_type, strategy_name)
        return self.strategy_configs.get(key)
    
    def execute_strategy(self, 
                        strategy_type: StrategyType, 
                        strategy_name: str, 
                        context: Dict[str, Any],
                        use_config: bool = True,
                        **kwargs) -> Any:
        """
        执行策略并记录性能指标
        
        Args:
            strategy_type: 策略类型
            strategy_name: 策略名称
            context: 执行上下文
            use_config: 是否使用注册的配置
            **kwargs: 额外参数
            
        Returns:
            策略执行结果
        """
        key = self._get_strategy_key(strategy_type, strategy_name)
        metrics = self.performance_metrics.get(key, StrategyPerformanceMetrics())
        
        # 检查策略是否启用
        if use_config:
            config = self.get_strategy_config(strategy_type, strategy_name)
            if config and not config.enabled:
                raise StrategyError(f"策略 '{strategy_name}' 已被禁用")
        
        start_time = time.time()
        
        try:
            # 创建策略实例
            strategy_config = None
            if use_config:
                config_obj = self.get_strategy_config(strategy_type, strategy_name)
                if config_obj:
                    strategy_config = config_obj.config
            
            strategy = self.factory.create_strategy(strategy_type, strategy_name, strategy_config)
            
            # 执行策略
            result = strategy.execute(context, **kwargs)
            
            # 更新性能指标
            execution_time = time.time() - start_time
            metrics.execution_count += 1
            metrics.total_execution_time += execution_time
            metrics.success_count += 1
            metrics.last_execution_time = execution_time
            
            # 记录执行历史
            self._record_execution_history(
                strategy_type, strategy_name, context, result, 
                execution_time, True, None
            )
            
            return result
            
        except Exception as e:
            # 更新错误指标
            execution_time = time.time() - start_time
            metrics.execution_count += 1
            metrics.total_execution_time += execution_time
            metrics.error_count += 1
            metrics.last_execution_time = execution_time
            metrics.last_error = str(e)
            
            # 记录执行历史
            self._record_execution_history(
                strategy_type, strategy_name, context, None,
                execution_time, False, str(e)
            )
            
            raise
        finally:
            self.performance_metrics[key] = metrics
    
    def _record_execution_history(self, 
                                 strategy_type: StrategyType, 
                                 strategy_name: str, 
                                 context: Dict[str, Any],
                                 result: Any,
                                 execution_time: float,
                                 success: bool,
                                 error: Optional[str]) -> None:
        """记录执行历史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy_type': strategy_type.value,
            'strategy_name': strategy_name,
            'context_keys': list(context.keys()),
            'execution_time': execution_time,
            'success': success,
            'error': error,
            'result_type': type(result).__name__ if result else None
        }
        self.execution_history.append(history_entry)
        
        # 保持历史记录大小
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_performance_metrics(self, strategy_type: StrategyType, strategy_name: str) -> Optional[StrategyPerformanceMetrics]:
        """获取策略性能指标"""
        key = self._get_strategy_key(strategy_type, strategy_name)
        return self.performance_metrics.get(key)
    
    def get_all_performance_metrics(self) -> Dict[str, StrategyPerformanceMetrics]:
        """获取所有策略性能指标"""
        return self.performance_metrics.copy()
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history[-limit:]
    
    def recommend_strategy(self, 
                          strategy_type: StrategyType, 
                          context: Dict[str, Any]) -> str:
        """
        推荐最适合的策略
        
        Args:
            strategy_type: 策略类型
            context: 执行上下文
            
        Returns:
            推荐的策略名称
        """
        available_strategies = self.factory.get_available_strategies(strategy_type)
        
        if not available_strategies:
            raise StrategyError(f"没有可用的 {strategy_type.value} 策略")
        
        # 简单的推荐逻辑：选择性能最好的策略
        best_strategy = None
        best_score = -1.0
        
        for strategy_name in available_strategies:
            metrics = self.get_performance_metrics(strategy_type, strategy_name)
            if metrics and metrics.execution_count > 0:
                # 评分公式：成功率 * (1 / 平均执行时间) * 优先级
                config = self.get_strategy_config(strategy_type, strategy_name)
                priority = config.priority if config else 0
                score = metrics.success_rate * (1 / max(metrics.average_execution_time, 0.001)) * (1 + priority / 10)
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
        
        # 如果没有历史数据，返回默认策略
        if best_strategy is None:
            return self.factory.get_default_strategy(strategy_type)
        
        return best_strategy
    
    def export_configurations(self) -> str:
        """导出所有策略配置"""
        config_data = {}
        for key, config in self.strategy_configs.items():
            config_data[key] = {
                'strategy_type': config.strategy_type.value,
                'strategy_name': config.strategy_name,
                'config': config.config,
                'enabled': config.enabled,
                'priority': config.priority,
                'description': config.description
            }
        return json.dumps(config_data, indent=2, ensure_ascii=False)
    
    def import_configurations(self, config_json: str) -> None:
        """导入策略配置"""
        config_data = json.loads(config_json)
        
        for key, config_info in config_data.items():
            try:
                strategy_type = StrategyType(config_info['strategy_type'])
                self.register_strategy_config(
                    strategy_type=strategy_type,
                    strategy_name=config_info['strategy_name'],
                    config=config_info['config'],
                    enabled=config_info['enabled'],
                    priority=config_info['priority'],
                    description=config_info['description']
                )
            except (KeyError, ValueError) as e:
                print(f"导入配置失败 {key}: {e}")
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self.execution_history.clear()
    
    def reset_performance_metrics(self) -> None:
        """重置性能指标"""
        self.performance_metrics.clear()
        self._initialize_performance_metrics()


# 全局策略管理器实例
strategy_manager = StrategyManager()