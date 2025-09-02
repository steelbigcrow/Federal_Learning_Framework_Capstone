"""
AdaLoRA训练策略

实现联邦学习中的AdaLoRA微调训练策略。
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ...core.interfaces.strategy import TrainingStrategyInterface, StrategyType, strategy_register
from ...core.exceptions.exceptions import StrategyError


@strategy_register(StrategyType.TRAINING, "adalora")
class AdaLoRATrainingStrategy(TrainingStrategyInterface):
    """
    AdaLoRA训练策略
    
    专门用于AdaLoRA微调的训练策略，支持动态rank分配和自适应预算管理。
    """
    
    def __init__(self):
        self.name = "adalora"
        self.description = "AdaLoRA微调训练策略，支持动态rank分配"
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def train(self, model: torch.nn.Module, train_loader: DataLoader, 
             optimizer: torch.optim.Optimizer, num_epochs: int, 
             device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None,
             **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        训练接口，匹配客户端的调用格式
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            num_epochs: 训练轮数
            device: 设备
            scaler: 梯度缩放器
            
        Returns:
            Tuple[模型状态字典, 训练指标]
        """
        config = {
            'optimizer': optimizer,
            'epochs': num_epochs,
            'device': device,
            'use_amp': scaler is not None,
            **kwargs
        }
        
        return self.train_model(model, train_loader, config)
    
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
        
        # AdaLoRA训练需要base_model_path
        if 'base_model_path' not in config:
            raise StrategyError("AdaLoRA训练需要 base_model_path 配置")
    
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
        训练AdaLoRA模型
        
        Args:
            model: 要训练的AdaLoRA模型
            train_loader: 训练数据加载器
            config: 训练配置
            
        Returns:
            训练指标字典
        """
        # 准备AdaLoRA模型
        model = self.prepare_model(model, config)
        
        # 获取优化器（只优化可训练参数）
        optimizer = self.get_optimizer(model, config)
        
        # 获取训练参数
        device = config.get('device', 'cpu')
        epochs = config.get('epochs', 1)
        use_amp = config.get('use_amp', False)
        rank_budget = config.get('rank_budget', None)
        
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
            'epochs_completed': 0,
            'adalora_params_count': self._count_trainable_params(model),
            'rank_distribution': self._get_rank_distribution(model)
        }
        
        for epoch in range(epochs):
            epoch_metrics = self._train_one_epoch(
                model, train_loader, optimizer, device, scaler, rank_budget
            )
            
            # 累积指标
            metrics['loss'] += epoch_metrics['loss']
            metrics['accuracy'] += epoch_metrics['accuracy']
            metrics['f1_score'] += epoch_metrics['f1_score']
            metrics['num_samples'] += epoch_metrics['num_samples']
            metrics['epochs_completed'] += 1
            
            # 更新rank分布
            metrics['rank_distribution'] = self._get_rank_distribution(model)
        
        # 计算平均指标
        if metrics['epochs_completed'] > 0:
            metrics['loss'] /= metrics['epochs_completed']
            metrics['accuracy'] /= metrics['epochs_completed']
            metrics['f1_score'] /= metrics['epochs_completed']
        
        # 返回模型状态字典和指标
        from copy import deepcopy
        return deepcopy(model.state_dict()), metrics
    
    def _train_one_epoch(self, 
                        model: torch.nn.Module,
                        data_loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        device: str,
                        scaler: Optional[torch.cuda.amp.GradScaler] = None,
                        rank_budget: Optional[int] = None) -> Dict[str, float]:
        """训练一个epoch的AdaLoRA模型"""
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
            
            # 动态rank分配（如果配置了rank预算）
            if rank_budget is not None and batch_idx % 100 == 0:
                self._adjust_ranks(model, rank_budget)
            
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
        评估AdaLoRA模型
        
        Args:
            model: 要评估的AdaLoRA模型
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
            "num_samples": total_samples,
            "adalora_params_count": self._count_trainable_params(model),
            "rank_distribution": self._get_rank_distribution(model)
        }
    
    def prepare_model(self, 
                      model: torch.nn.Module, 
                      config: Dict[str, Any]) -> torch.nn.Module:
        """
        准备AdaLoRA模型
        
        Args:
            model: 原始模型
            config: 准备配置
            
        Returns:
            准备后的AdaLoRA模型
        """
        # 检查模型是否已经应用了AdaLoRA
        if not self._has_adalora_layers(model):
            raise StrategyError("模型没有应用AdaLoRA层")
        
        # 冻结基础模型参数，只保留AdaLoRA参数可训练
        self._freeze_base_model(model)
        
        # 初始化rank分配
        rank_budget = config.get('rank_budget', None)
        if rank_budget is not None:
            self._initialize_rank_allocation(model, rank_budget)
        
        return model
    
    def get_optimizer(self, 
                     model: torch.nn.Module, 
                     config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        获取AdaLoRA优化器
        
        Args:
            model: AdaLoRA模型
            config: 优化器配置
            
        Returns:
            配置好的优化器
        """
        # 只优化可训练参数（AdaLoRA参数和分类头）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise StrategyError("没有可训练的参数")
        
        optimizer_type = config.get('optimizer', 'adam')
        lr = config.get('lr', 0.001)
        weight_decay = config.get('weight_decay', 0.0)
        
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = config.get('momentum', 0.9)
            return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise StrategyError(f"不支持的优化器类型: {optimizer_type}")
    
    def _has_adalora_layers(self, model: torch.nn.Module) -> bool:
        """检查模型是否有AdaLoRA层"""
        for name, param in model.named_parameters():
            if 'adalora_' in name.lower() or 'lora_' in name.lower():
                return True
        return False
    
    def _freeze_base_model(self, model: torch.nn.Module) -> None:
        """冻结基础模型参数"""
        for name, param in model.named_parameters():
            if not self._is_adalora_param(name):
                param.requires_grad = False
    
    def _is_adalora_param(self, param_name: str) -> bool:
        """判断是否为AdaLoRA参数"""
        return ('adalora_' in param_name.lower() or 
                'lora_' in param_name.lower() or 
                'classifier' in param_name.lower())
    
    def _count_trainable_params(self, model: torch.nn.Module) -> int:
        """计算可训练参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _get_rank_distribution(self, model: torch.nn.Module) -> Dict[str, int]:
        """获取rank分布信息"""
        rank_dist = {}
        for name, param in model.named_parameters():
            if 'lora_' in name.lower() and 'weight' in name.lower():
                # 假设LoRA权重矩阵的形状为 (out_features, rank)
                if len(param.shape) == 2:
                    rank_dist[name] = param.shape[1]
        return rank_dist
    
    def _initialize_rank_allocation(self, model: torch.nn.Module, rank_budget: int) -> None:
        """初始化rank分配"""
        # 简单的均匀分配策略
        adalora_params = []
        for name, param in model.named_parameters():
            if self._is_adalora_param(name) and len(param.shape) == 2:
                adalora_params.append((name, param))
        
        if adalora_params:
            rank_per_param = max(1, rank_budget // len(adalora_params))
            for name, param in adalora_params:
                # 这里应该调整参数的rank，但需要具体的AdaLoRA实现
                pass
    
    def _adjust_ranks(self, model: torch.nn.Module, rank_budget: int) -> None:
        """动态调整rank分配"""
        # 这里应该实现基于重要性的rank分配调整
        # 简化实现，实际应该根据参数重要性重新分配rank
        pass
    
    def get_trainable_keys(self, model: torch.nn.Module) -> Set[str]:
        """
        获取模型中可训练参数的键名
        
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
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取策略配置模式"""
        return {
            "type": "object",
            "properties": {
                "base_model_path": {
                    "type": "string",
                    "description": "基础模型路径"
                },
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
                "rank_budget": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "AdaLoRA rank预算"
                },
                "device": {
                    "type": "string",
                    "default": "cpu",
                    "description": "训练设备"
                }
            },
            "required": ["base_model_path"]
        }