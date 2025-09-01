"""
Federated Learning Server Implementation

This module provides the concrete implementation of the federated learning server
using OOP design patterns while maintaining backward compatibility with existing code.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

import torch

from ...core.base.server import AbstractServer
from ...core.exceptions.exceptions import ServerConfigurationError, ServerOperationError
from ...core.interfaces.strategy import AggregationStrategyInterface
from ...utils.paths import PathManager


class FederatedServer(AbstractServer):
    """
    联邦学习服务器实现
    
    基于OOP设计模式重构的服务器实现，提供：
    1. 标准化的联邦学习流程管理
    2. 可配置的聚合策略
    3. 完整的检查点和日志管理
    4. 向后兼容的API接口
    
    设计原则：
    - 单一职责：专注于全局协调和聚合
    - 开闭原则：通过策略模式支持不同聚合算法
    - 依赖倒置：依赖抽象的聚合策略接口
    """
    
    def __init__(self, 
                 model_constructor: Callable[[], torch.nn.Module],
                 clients: List,
                 path_manager: PathManager,
                 config: Dict[str, Any],
                 device: str = "cpu",
                 aggregation_strategy: Optional[AggregationStrategyInterface] = None,
                 **kwargs):
        """
        初始化联邦学习服务器
        
        Args:
            model_constructor: 模型构造函数
            clients: 客户端列表
            path_manager: 路径管理器
            config: 服务器配置字典，包含：
                - lora_cfg: LoRA配置
                - adalora_cfg: AdaLoRA配置
                - save_client_each_round: 是否每轮保存客户端模型
                - model_info: 模型信息
            device: 计算设备
            aggregation_strategy: 聚合策略实例
            **kwargs: 其他参数
        """
        # 提取配置参数
        lora_cfg = config.get('lora_cfg', {})
        adalora_cfg = config.get('adalora_cfg', {})
        save_client_each_round = config.get('save_client_each_round', True)
        model_info = config.get('model_info', {})
        
        # 存储原有参数以保持兼容性
        self._path_manager = path_manager
        self._lora_cfg = lora_cfg
        self._adalora_cfg = adalora_cfg
        self._save_client_each_round = bool(save_client_each_round)
        self._model_info = model_info
        self._aggregation_strategy = aggregation_strategy
        
        # 构建完整配置
        full_config = {
            **config,
            'device': device,
            'lora_cfg': lora_cfg,
            'adalora_cfg': adalora_cfg,
            'save_client_each_round': save_client_each_round,
            'model_info': model_info
        }
        
        # 调用父类初始化
        super().__init__(
            model_constructor=model_constructor,
            clients=clients,
            config=full_config,
            device=device,
            **kwargs
        )
        
    def initialize_global_model(self) -> None:
        """
        初始化全局模型
        
        Raises:
            ServerConfigurationError: 当模型初始化失败时
        """
        try:
            self._global_model = self._model_constructor()
            self._global_model.to(self._device)
            
            # 验证模型设备
            model_device = next(self._global_model.parameters()).device
            if str(model_device) != self._device:
                raise ServerConfigurationError(f"Model device {model_device} doesn't match configured device {self._device}")
                
            self._logger.info("Global model initialized successfully")
            
        except Exception as e:
            raise ServerConfigurationError(f"Failed to initialize global model: {e}") from e
            
    def select_clients(self, round_number: int) -> List:
        """
        选择参与当前轮训练的客户端
        
        Args:
            round_number: 当前轮次号
            
        Returns:
            被选中的客户端列表
        """
        # 默认选择所有客户端
        return self._clients.copy()
        
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
        try:
            if self._aggregation_strategy:
                # 使用策略模式进行聚合
                aggregated_state = self._aggregation_strategy.aggregate(
                    client_models=client_models,
                    client_weights=client_weights,
                    global_model=self._global_model,
                    config=self._config
                )
            else:
                # 使用默认聚合逻辑
                aggregated_state = self._default_aggregate(client_models, client_weights)
                
            return aggregated_state
            
        except Exception as e:
            raise ServerOperationError(f"Model aggregation failed: {e}") from e
            
    def save_checkpoint(self, round_number: int, metrics: Dict[str, Any]) -> None:
        """
        保存检查点
        
        Args:
            round_number: 轮次号
            metrics: 当前轮次的指标
            
        Raises:
            ServerOperationError: 当保存失败时
        """
        try:
            # 保存客户端模型
            if self._save_client_each_round:
                self._save_client_checkpoints(round_number, metrics)
                
            # 保存全局模型
            self._save_global_checkpoint(round_number, metrics)
            
            # 保存轮次指标
            self._save_round_metrics(round_number, metrics)
            
            # 生成指标图表
            self._generate_metrics_plots(round_number)
            
        except Exception as e:
            raise ServerOperationError(f"Checkpoint saving failed: {e}") from e
            
    def _default_aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                          client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        默认聚合逻辑
        
        Args:
            client_models: 客户端模型状态字典列表
            client_weights: 客户端权重列表
            
        Returns:
            聚合后的全局模型状态字典
        """
        from ...federated.aggregator import fedavg, lora_fedavg, get_trainable_keys
        
        # 根据是否使用LoRA选择聚合策略
        if self._lora_cfg and self._lora_cfg.get('replaced_modules'):
            # LoRA模式：只聚合可训练的权重
            trainable_keys = get_trainable_keys(self._global_model)
            print(f"[LoRA Aggregation] Trainable parameters count: {len(trainable_keys)}")
            
            aggregated_state = lora_fedavg(client_models, client_weights, trainable_keys)
            print(f"[LoRA Aggregation] LoRA aggregation completed")
        else:
            # 标准模式：聚合所有权重
            aggregated_state = fedavg(client_models, client_weights)
            print(f"[Standard Aggregation] Standard aggregation completed")
            
        return aggregated_state
        
    def _save_client_checkpoints(self, round_number: int, metrics: Dict[str, Any]) -> None:
        """
        保存客户端检查点
        
        Args:
            round_number: 轮次号
            metrics: 轮次指标
        """
        from ..training.checkpoints import save_client_round
        from ..training.lora_utils import save_lora_checkpoint
        from ..training.logging_utils import write_text_log
        
        for client in self._clients:
            try:
                log_file = self._path_manager.client_round_log(client.client_id, round_number)
                
                # 构建元数据
                meta = self._build_client_meta(round_number, client.client_id, 
                                              getattr(client, 'num_samples', 0), 
                                              getattr(client, '_training_metrics', {}))
                
                ckpt_path = self._path_manager.client_round_ckpt(client.client_id, round_number)
                
                # 根据模式选择保存方式
                if self._lora_cfg and self._lora_cfg.get('replaced_modules'):
                    # LoRA模式：保存LoRA权重
                    success = self._safe_save_checkpoint(
                        save_lora_checkpoint, log_file, ckpt_path,
                        client.global_model if hasattr(client, 'global_model') else None,
                        ckpt_path, meta
                    )
                else:
                    # 标准模式：保存完整模型
                    success = self._safe_save_checkpoint(
                        save_client_round, log_file, ckpt_path,
                        client.global_model if hasattr(client, 'global_model') else None,
                        ckpt_path, meta=meta
                    )
                    
                if success:
                    print(f"Client {client.client_id} checkpoint saved")
                    
            except Exception as e:
                self._logger.error(f"Failed to save client {client.client_id} checkpoint: {e}")
                
    def _save_global_checkpoint(self, round_number: int, metrics: Dict[str, Any]) -> None:
        """
        保存全局模型检查点
        
        Args:
            round_number: 轮次号
            metrics: 轮次指标
        """
        from ..training.checkpoints import save_global_round
        from ..training.lora_utils import save_lora_checkpoint
        from ..training.logging_utils import write_text_log
        
        try:
            server_log = self._path_manager.server_round_log(round_number)
            g_path = self._path_manager.global_round_ckpt(round_number)
            
            # 构建元数据
            g_meta = self._build_server_meta(round_number)
            
            # 根据模式选择保存方式
            if self._lora_cfg and self._lora_cfg.get('replaced_modules'):
                # LoRA模式：保存LoRA权重
                success = self._safe_save_checkpoint(
                    save_lora_checkpoint, server_log, g_path,
                    self._global_model, g_path, g_meta
                )
            else:
                # 标准模式：保存完整模型
                success = self._safe_save_checkpoint(
                    save_global_round, server_log, g_path,
                    self._global_model.state_dict(), g_path, meta=g_meta
                )
                
            if success:
                print(f"Global checkpoint saved for round {round_number}")
                
        except Exception as e:
            self._logger.error(f"Failed to save global checkpoint: {e}")
            
    def _save_round_metrics(self, round_number: int, metrics: Dict[str, Any]) -> None:
        """
        保存轮次指标
        
        Args:
            round_number: 轮次号
            metrics: 轮次指标
        """
        from ..training.logging_utils import write_metrics_json
        
        try:
            round_summary = {
                "round": round_number,
                "num_clients": len(self._clients),
                "model_info": self._model_info,
                "lora": self._lora_cfg,
                **metrics
            }
            
            round_metrics_path = self._path_manager.round_metrics(round_number)
            write_metrics_json(round_metrics_path, round_summary)
            
        except Exception as e:
            self._logger.error(f"Failed to save round metrics: {e}")
            
    def _generate_metrics_plots(self, round_number: int) -> None:
        """
        生成指标图表
        
        Args:
            round_number: 轮次号
        """
        from ..training.plotting import plot_all_clients_metrics
        
        try:
            client_ids = [client.client_id for client in self._clients]
            plot_all_clients_metrics(
                client_ids=client_ids,
                metrics_clients_dir=self._path_manager.metrics_clients_dir,
                plots_dir=self._path_manager.plots_clients_dir,
                current_round=round_number
            )
        except Exception as e:
            self._logger.error(f"Failed to generate metrics plots: {e}")
            
    def _safe_save_checkpoint(self, save_func, log_file: str, ckpt_path: str, *args, **kwargs) -> bool:
        """
        安全地保存检查点
        
        Args:
            save_func: 保存函数
            log_file: 日志文件路径
            ckpt_path: 检查点路径
            *args, **kwargs: 传递给保存函数的参数
            
        Returns:
            是否保存成功
        """
        try:
            save_func(*args, **kwargs)
            self._logger.info(f"Checkpoint saved: {ckpt_path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save checkpoint {ckpt_path}: {e}")
            return False
            
    def _build_client_meta(self, round_num: int, client_id: int, num_samples: int, metrics: Dict) -> Dict:
        """构建客户端元数据"""
        return {
            "round": round_num,
            "client": client_id,
            "num_samples": num_samples,
            "metrics": metrics,
            "lora": self._lora_cfg,
        }
        
    def _build_server_meta(self, round_num: int, is_lora: bool = False) -> Dict:
        """构建服务器元数据"""
        if is_lora:
            return {
                "round": round_num,
                "dataset": self._model_info.get('dataset', 'unknown'),
                "model_type": self._model_info.get('model_type', 'unknown'),
                "base_model_path": self._lora_cfg.get('base_model_path', ''),
                "lora_config": self._lora_cfg,
                "is_global": True
            }
        else:
            return {"round": round_num, "lora": self._lora_cfg}
            
    def _compute_round_metrics(self, round_number: int, clients: List, client_weights: List[float]) -> Dict[str, Any]:
        """计算轮次指标"""
        total_samples = sum(client_weights)
        
        metrics = super()._compute_round_metrics(round_number, clients, client_weights)
        
        # 添加特定指标
        metrics.update({
            "model_info": self._model_info,
            "lora": self._lora_cfg,
            "num_samples_total": int(total_samples),
            "num_samples_per_client": client_weights
        })
        
        return metrics
        
    # 向后兼容的属性和方法
    @property
    def model_ctor(self) -> Callable[[], torch.nn.Module]:
        """兼容性属性：获取模型构造函数"""
        return self._model_constructor
        
    @property
    def clients(self) -> List:
        """兼容性属性：获取客户端列表"""
        return self._clients.copy()
        
    @property
    def paths(self) -> PathManager:
        """兼容性属性：获取路径管理器"""
        return self._path_manager
        
    @property
    def lora_cfg(self) -> Dict[str, Any]:
        """兼容性属性：获取LoRA配置"""
        return self._lora_cfg
        
    @property
    def adalora_cfg(self) -> Dict[str, Any]:
        """兼容性属性：获取AdaLoRA配置"""
        return self._adalora_cfg
        
    @property
    def save_client_each_round(self) -> bool:
        """兼容性属性：获取是否每轮保存客户端模型"""
        return self._save_client_each_round
        
    @property
    def model_info(self) -> Dict[str, Any]:
        """兼容性属性：获取模型信息"""
        return self._model_info
        
    def run(self, num_rounds: int, local_epochs: int) -> None:
        """
        向后兼容的运行方法
        
        Args:
            num_rounds: 联邦学习轮数
            local_epochs: 客户端本地训练轮数
        """
        print(f"[Federated Learning] Starting training: {num_rounds} rounds, {len(self.clients)} clients, device={self.device}")
        
        # 显示客户端数据分布
        print(f"Client data distribution:")
        for client in self.clients:
            train_size = getattr(client, 'num_train_samples', 0)
            test_size = getattr(client, 'num_test_samples', 0)
            total_size = train_size + test_size
            print(f"  Client {client.client_id}: total={total_size}, train={train_size}, test={test_size}")
            
        # 使用基类的联邦学习流程
        result = self.run_federated_learning(num_rounds, local_epochs)
        
        print(f"[Federated Learning] Training completed successfully")
        
    # 辅助方法
    def safe_write_logs(self, log_file: str, server_log: str, msg: str) -> None:
        """安全地写入日志到多个文件"""
        from ..training.logging_utils import write_text_log
        
        for log_path in [log_file, server_log]:
            try:
                write_text_log(log_path, msg)
            except Exception as e:
                print(f"[Error] Failed to write log {log_path}: {e}")
                
    def safe_write_single_log(self, log_path: str, msg: str) -> None:
        """安全地写入日志到单个文件"""
        from ..training.logging_utils import write_text_log
        
        try:
            write_text_log(log_path, msg)
        except Exception as e:
            print(f"[Error] Failed to write log {log_path}: {e}")
            
    def safe_save_and_log(self, save_func, success_msg: str, log_file: str, server_log: str, print_path: str, *args, **kwargs) -> bool:
        """安全地执行保存操作并记录日志"""
        try:
            save_func(*args, **kwargs)
            try:
                write_text_log(log_file, success_msg)
                write_text_log(server_log, success_msg)
                print(f"  Saved to: {print_path}")
                return True
            except Exception as e:
                print(f"[Error] Failed to write save success message: {e}")
                return False
        except Exception as e:
            print(f"[Error] Failed to save: {e}")
            return False
            
    def build_client_meta(self, round_num: int, client_id: int, num_samples: int, metrics: Dict) -> Dict:
        """构建客户端元数据（兼容性方法）"""
        return self._build_client_meta(round_num, client_id, num_samples, metrics)
        
    def build_server_meta(self, round_num: int, is_lora: bool = False) -> Dict:
        """构建服务器元数据（兼容性方法）"""
        return self._build_server_meta(round_num, is_lora)