"""
Federated Learning Client Implementation

This module provides the concrete implementation of the federated learning client
using OOP design patterns while maintaining backward compatibility with existing code.
"""

from copy import deepcopy
from typing import Dict, Tuple, Optional, Any, List

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW

from ...core.base.client import AbstractClient
from ...core.exceptions.exceptions import ClientConfigurationError, ClientTrainingError
from ...core.interfaces.strategy import TrainingStrategyInterface


class FederatedClient(AbstractClient):
    """
    联邦学习客户端实现
    
    基于OOP设计模式重构的客户端实现，提供：
    1. 标准化的本地训练和评估接口
    2. 可配置的训练策略
    3. 完整的数据管理功能
    4. 向后兼容的API接口
    
    设计原则：
    - 单一职责：专注于本地训练和数据管理
    - 开闭原则：通过策略模式支持不同训练算法
    - 依赖倒置：依赖抽象的训练策略接口
    """
    
    def __init__(self, 
                 client_id: int,
                 model_ctor,
                 train_data_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = "cpu",
                 training_strategy: Optional[TrainingStrategyInterface] = None,
                 **kwargs):
        """
        初始化联邦学习客户端
        
        Args:
            client_id: 客户端唯一标识
            model_ctor: 模型构造函数
            train_data_loader: 训练数据加载器
            config: 客户端配置字典，包含：
                - optimizer: 优化器配置
                - use_amp: 是否使用自动混合精度
                - test_ratio: 测试集比例
            device: 计算设备
            training_strategy: 训练策略实例
            **kwargs: 其他参数
        """
        # 提取配置参数
        optimizer_cfg = config.get('optimizer', {})
        use_amp = config.get('use_amp', False)
        test_ratio = config.get('test_ratio', 0.3)
        
        # 存储原有参数以保持兼容性
        self._model_ctor = model_ctor
        self._optimizer_cfg = optimizer_cfg
        self._use_amp = use_amp
        self._test_ratio = test_ratio
        self._training_strategy = training_strategy
        
        # 数据分割（先存储，在父类构造后使用）
        self._original_train_loader = train_data_loader
        self._test_ratio_for_split = test_ratio
        
        # 构建完整配置
        full_config = {
            **config,
            'device': device,
            'optimizer': optimizer_cfg,
            'use_amp': use_amp,
            'test_ratio': test_ratio
        }
        
        super().__init__(
            client_id=client_id,
            train_data_loader=train_data_loader,
            config=full_config,
            device=device,
            **kwargs
        )
        
        # 现在可以安全地访问self._client_id进行数据分割
        self._split_train_test_data(self._original_train_loader, self._test_ratio_for_split)
        
        # 更新数据加载器
        self._train_data_loader = self._train_data_loader
        self.set_test_data_loader(self._test_data_loader)
        
        # 清理临时变量
        delattr(self, '_original_train_loader')
        delattr(self, '_test_ratio_for_split')
        
    def _split_train_test_data(self, train_loader: DataLoader, test_ratio: float) -> None:
        """
        将训练数据分割为训练集和测试集
        
        Args:
            train_loader: 原始训练数据加载器
            test_ratio: 测试集比例
        """
        original_dataset = train_loader.dataset
        total_size = len(original_dataset)
        
        # 按照比例分割
        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size
        
        # 分割数据集
        train_dataset_new, test_dataset = random_split(
            original_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42 + self._client_id)
        )
        
        # 创建新的数据加载器
        self._train_data_loader = DataLoader(
            train_dataset_new,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=getattr(train_loader, 'num_workers', 0),
            pin_memory=getattr(train_loader, 'pin_memory', False),
            collate_fn=getattr(train_loader, 'collate_fn', None)
        )
        
        self._test_data_loader = DataLoader(
            test_dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=getattr(train_loader, 'num_workers', 0),
            pin_memory=getattr(train_loader, 'pin_memory', False),
            collate_fn=getattr(train_loader, 'collate_fn', None)
        )
        
    def receive_global_model(self, global_model_state: Dict[str, torch.Tensor]) -> None:
        """
        接收全局模型参数
        
        Args:
            global_model_state: 全局模型的状态字典
            
        Raises:
            ClientConfigurationError: 当模型配置不兼容时
        """
        try:
            # 创建模型实例
            self._current_model = self._model_ctor()
            self._current_model.load_state_dict(global_model_state, strict=False)
            self._current_model.to(self._device)
            
            # 验证模型设备
            model_device = next(self._current_model.parameters()).device
            # 处理设备字符串格式差异（例如 "cuda:0" vs "cuda"）
            if self._device == 'cuda' and str(model_device).startswith('cuda:'):
                # CUDA设备匹配
                pass
            elif self._device == 'cpu' and str(model_device) == 'cpu':
                # CPU设备匹配
                pass
            elif str(model_device) != self._device:
                raise ClientConfigurationError(f"Model device {model_device} doesn't match configured device {self._device}")
                
            self._logger.info(f"Client {self._client_id} received global model successfully")
            
        except Exception as e:
            raise ClientConfigurationError(f"Failed to load global model: {e}") from e
            
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
        if self._current_model is None:
            raise ClientTrainingError("No model loaded. Call receive_global_model first.")
            
        try:
            # 构建优化器
            optimizer = self._build_optimizer(self._current_model)
            
            # 创建梯度缩放器
            scaler = None
            if self._use_amp and self._device.startswith('cuda'):
                scaler = torch.cuda.amp.GradScaler()
                
            # 使用训练策略或默认训练逻辑
            if self._training_strategy:
                model_state, train_metrics = self._training_strategy.train(
                    model=self._current_model,
                    train_loader=self._train_data_loader,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    device=self._device,
                    scaler=scaler
                )
            else:
                model_state, train_metrics = self._default_train(
                    self._current_model, self._train_data_loader, optimizer, num_epochs, scaler
                )
                
            return model_state, train_metrics, self.num_train_samples
            
        except Exception as e:
            raise ClientTrainingError(f"Local training failed: {e}") from e
            
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
        try:
            # 使用提供的模型状态或当前模型
            if model_state is not None:
                model = self._model_ctor()
                model.load_state_dict(model_state, strict=False)
                model.to(self._device)
            elif self._current_model is not None:
                model = self._current_model
            else:
                raise ClientTrainingError("No model available for evaluation")
                
            # 使用训练策略或默认评估逻辑
            if self._training_strategy:
                eval_metrics = self._training_strategy.evaluate(
                    model=model,
                    test_loader=self._test_data_loader,
                    device=self._device
                )
            else:
                eval_metrics = self._default_evaluate(model, self._test_data_loader, self._device)
                
            return eval_metrics
            
        except Exception as e:
            raise ClientTrainingError(f"Local evaluation failed: {e}") from e
            
    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        根据配置构建优化器
        
        Args:
            model: PyTorch模型
            
        Returns:
            配置好的优化器实例
        """
        name = self._optimizer_cfg.get('name', 'adam').lower()
        lr = float(self._optimizer_cfg.get('lr', 1e-3))
        wd = float(self._optimizer_cfg.get('weight_decay', 0.0))
        
        # 只优化需要梯度的参数
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        if name == 'adamw':
            return AdamW(trainable_params, lr=lr, weight_decay=wd)
        else:
            return Adam(trainable_params, lr=lr, weight_decay=wd)
            
    def _default_train(self, model: torch.nn.Module, train_loader: DataLoader, 
                     optimizer: torch.optim.Optimizer, num_epochs: int, 
                     scaler: Optional[torch.cuda.amp.GradScaler]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        默认训练逻辑
        
        Args:
            model: 训练模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            num_epochs: 训练轮数
            scaler: 梯度缩放器
            
        Returns:
            Tuple[模型状态字典, 训练指标字典]
        """
        train_metrics_history = []
        
        for epoch in range(1, num_epochs + 1):
            # 执行一个epoch的训练
            epoch_metrics = self._train_one_epoch(model, train_loader, optimizer, scaler)
            train_metrics_history.append({
                'epoch': epoch,
                **epoch_metrics
            })
            
            # 打印训练进度
            print(f"    Epoch {epoch}: train loss={epoch_metrics['loss']:.4f}, "
                  f"acc={epoch_metrics['acc']:.4f}, f1={epoch_metrics['f1']:.4f}")
                  
        # 返回最终模型状态和指标历史
        final_metrics = {
            'train_acc': train_metrics_history[-1].get('acc', 0),
            'train_f1': train_metrics_history[-1].get('f1', 0),
            'train_loss': train_metrics_history[-1].get('loss', 0),
            'epoch_history': train_metrics_history
        }
        
        return deepcopy(model.state_dict()), final_metrics
        
    def _default_evaluate(self, model: torch.nn.Module, test_loader: DataLoader, device: str) -> Dict[str, Any]:
        """
        默认评估逻辑
        
        Args:
            model: 评估模型
            test_loader: 测试数据加载器
            device: 计算设备
            
        Returns:
            评估指标字典
        """
        eval_results = self._evaluate_model(model, test_loader)
        
        # 添加test_前缀以确保与plotting系统兼容
        return {
            'test_acc': eval_results.get('acc', 0),
            'test_f1': eval_results.get('f1', 0), 
            'test_loss': eval_results.get('loss', 0)
        }

    def _train_one_epoch(self, model, data_loader, optimizer, scaler=None) -> Dict[str, float]:
        """
        训练一个epoch的函数
        
        Args:
            model: 要训练的模型
            data_loader: 训练数据加载器
            optimizer: 优化器
            scaler: 梯度缩放器（用于混合精度训练）
            
        Returns:
            包含损失、准确率和F1分数的字典
        """
        import torch.nn.functional as F
        from sklearn.metrics import f1_score
        
        # 处理空数据集情况
        if len(data_loader.dataset) == 0:
            return {"loss": 0.0, "acc": 0.0, "f1": 0.0}

        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        # 在第一个批次时显示设备信息
        first_batch = True

        for step, batch in enumerate(data_loader, start=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise ValueError("Batch format must be (x, y)")
            x = x.to(self._device)
            y = y.to(self._device)

            # 第一个批次时打印设备验证信息
            if first_batch:
                first_batch = False

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

            # 混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

            # 计算准确率并收集预测结果用于F1计算
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)

            # 收集预测结果和标签用于F1计算
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        # 计算F1分数
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return {
            "loss": total_loss / max(1, total),
            "acc": total_correct / max(1, total),
            "f1": f1
        }

    def _evaluate_model(self, model, data_loader) -> Dict[str, float]:
        """
        模型评估函数
        
        Args:
            model: 要评估的模型
            data_loader: 测试数据加载器
            
        Returns:
            包含损失、准确率和F1分数的字典
        """
        import torch.nn.functional as F
        from sklearn.metrics import f1_score
        
        # 处理空数据集情况
        if len(data_loader.dataset) == 0:
            return {"loss": 0.0, "acc": 0.0, "f1": 0.0}

        # 保存原始训练状态
        original_training = model.training
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        # 验证模型在正确设备上（仅第一个批次）
        first_batch = True

        try:
            with torch.no_grad():
                for batch in data_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        x, y = batch
                    else:
                        raise ValueError("Batch format must be (x, y)")
                    x = x.to(self._device)
                    y = y.to(self._device)

                    # 第一个批次时验证设备
                    if first_batch:
                        model_device = next(model.parameters()).device
                        first_batch = False
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    pred = logits.argmax(dim=1)
                    total_correct += (pred == y).sum().item()
                    total += y.size(0)
                    total_loss += loss.item() * y.size(0)

                    # 收集预测结果和标签用于F1计算
                    all_predictions.extend(pred.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            # 计算F1分数
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            return {
                "loss": total_loss / max(1, total),
                "acc": total_correct / max(1, total),
                "f1": f1
            }
        finally:
            # 恢复原始训练状态
            if original_training:
                model.train()
        
    # 向后兼容的属性和方法
    @property
    def id(self) -> int:
        """兼容性属性：获取客户端ID"""
        return self._client_id
        
    @property
    def model_ctor(self):
        """兼容性属性：获取模型构造函数"""
        return self._model_ctor
        
    @property
    def optimizer_cfg(self) -> Dict[str, Any]:
        """兼容性属性：获取优化器配置"""
        return self._optimizer_cfg
        
    @property
    def use_amp(self) -> bool:
        """兼容性属性：获取是否使用混合精度"""
        return self._use_amp
        
    @property
    def test_ratio(self) -> float:
        """兼容性属性：获取测试集比例"""
        return self._test_ratio
        
    def local_train_and_eval(self, global_state_dict: Dict[str, torch.Tensor], 
                            local_epochs: int = 1) -> Tuple[Dict, Dict, int]:
        """
        向后兼容的训练和评估方法
        
        Args:
            global_state_dict: 全局模型状态字典
            local_epochs: 本地训练轮数
            
        Returns:
            Tuple[模型状态字典, 综合指标字典, 训练样本数量]
        """
        # 使用基类的标准流程
        model_state, combined_metrics, num_samples = self.local_train_and_evaluate(
            global_state_dict, local_epochs
        )
        
        # 确保返回格式与原版本兼容
        if 'epoch_history' in combined_metrics:
            # 从历史中提取最终测试指标
            final_test_metrics = combined_metrics['epoch_history'][-1] if combined_metrics['epoch_history'] else {}
            
            # 合并指标以确保兼容性
            final_metrics = {
                **combined_metrics,
                'test_acc': final_test_metrics.get('test_acc', combined_metrics.get('test_acc', 0)),
                'test_f1': final_test_metrics.get('test_f1', combined_metrics.get('test_f1', 0)),
                'test_loss': final_test_metrics.get('test_loss', combined_metrics.get('test_loss', 0)),
            }
            
            return model_state, final_metrics, num_samples
            
        return model_state, combined_metrics, num_samples