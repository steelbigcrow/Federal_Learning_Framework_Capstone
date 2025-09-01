"""
标准训练策略

实现联邦学习中的标准客户端训练策略。
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ...core.interfaces.strategy import TrainingStrategyInterface, StrategyType, strategy_register
from ...core.exceptions.exceptions import StrategyError


@strategy_register(StrategyType.TRAINING, "standard")
class StandardTrainingStrategy(TrainingStrategyInterface):
    """
    标准训练策略
    
    实现基础的客户端本地训练逻辑，支持标准联邦学习。
    """
    
    def __init__(self):
        self.name = "standard"
        self.description = "标准联邦学习训练策略，支持基础模型训练"
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def validate_context(self, context: Dict[str, Any]) -> None:
        """验证策略执行上下文"""
        required_keys = ['model', 'train_loader', 'config']
        for key in required_keys:
            if key not in context:
                raise StrategyError(f"缺少必需的上下文键: {key}")
        
        model = context['model']
        train_loader = context['train_loader']
        config = context['config']
        
        if not isinstance(model, torch.nn.Module):
            raise StrategyError("model 必须是 torch.nn.Module 实例")
        
        if not isinstance(train_loader, DataLoader):
            raise StrategyError("train_loader 必须是 DataLoader 实例")
        
        if not isinstance(config, dict):
            raise StrategyError("config 必须是字典")
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Any:
        """执行训练策略"""
        self.validate_context(context)
        
        model = context['model']
        train_loader = context['train_loader']
        config = context['config']
        
        return self.train_model(model, train_loader, config)
    
    def train_model(self, 
                    model: torch.nn.Module,
                    train_loader: DataLoader,
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
        # 准备模型
        model = self.prepare_model(model, config)
        
        # 获取优化器
        optimizer = self.get_optimizer(model, config)
        
        # 获取训练参数
        device = config.get('device', 'cpu')
        epochs = config.get('epochs', 1)
        use_amp = config.get('use_amp', False)
        
        # 设置设备
        model.to(device)
        
        # 初始化混合精度训练
        scaler = None
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # 训练循环
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'num_samples': 0,
            'epochs_completed': 0
        }
        
        for epoch in range(epochs):
            epoch_metrics = self._train_one_epoch(
                model, train_loader, optimizer, device, scaler
            )
            
            # 累积指标
            metrics['loss'] += epoch_metrics['loss']
            metrics['accuracy'] += epoch_metrics['accuracy']
            metrics['f1_score'] += epoch_metrics['f1_score']
            metrics['num_samples'] += epoch_metrics['num_samples']
            metrics['epochs_completed'] += 1
        
        # 计算平均指标
        if metrics['epochs_completed'] > 0:
            metrics['loss'] /= metrics['epochs_completed']
            metrics['accuracy'] /= metrics['epochs_completed']
            metrics['f1_score'] /= metrics['epochs_completed']
        
        return metrics
    
    def _train_one_epoch(self, 
                        model: torch.nn.Module,
                        data_loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        device: str,
                        scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
        """训练一个epoch"""
        # 处理空数据集情况
        if len(data_loader.dataset) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0, "num_samples": 0}
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()
            
            # 收集预测和目标用于F1计算
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / max(1, len(data_loader))
        accuracy = total_correct / max(1, total_samples)
        f1 = f1_score(all_targets, all_predictions, average='weighted') if all_predictions else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_score": f1,
            "num_samples": total_samples
        }
    
    def evaluate_model(self, 
                       model: torch.nn.Module,
                       test_loader: DataLoader,
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
        device = config.get('device', 'cpu')
        model.to(device)
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                total_correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / max(1, len(test_loader))
        accuracy = total_correct / max(1, total_samples)
        f1 = f1_score(all_targets, all_predictions, average='weighted') if all_predictions else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_score": f1,
            "num_samples": total_samples
        }
    
    def prepare_model(self, 
                      model: torch.nn.Module, 
                      config: Dict[str, Any]) -> torch.nn.Module:
        """
        准备模型
        
        Args:
            model: 原始模型
            config: 准备配置
            
        Returns:
            准备后的模型
        """
        # 标准训练策略不需要特殊准备
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
        optimizer_type = config.get('optimizer', 'adam')
        lr = config.get('lr', 0.001)
        weight_decay = config.get('weight_decay', 0.0)
        
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = config.get('momentum', 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise StrategyError(f"不支持的优化器类型: {optimizer_type}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取策略配置模式"""
        return {
            "type": "object",
            "properties": {
                "epochs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "训练轮数"
                },
                "lr": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.001,
                    "description": "学习率"
                },
                "optimizer": {
                    "type": "string",
                    "enum": ["adam", "adamw", "sgd"],
                    "default": "adam",
                    "description": "优化器类型"
                },
                "use_amp": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否使用自动混合精度"
                },
                "device": {
                    "type": "string",
                    "default": "cpu",
                    "description": "训练设备"
                }
            }
        }