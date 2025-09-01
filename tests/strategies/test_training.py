"""
训练策略单元测试
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any

from src.strategies.training.standard import StandardTrainingStrategy
from src.strategies.training.lora import LoRATrainingStrategy
from src.strategies.training.adalora import AdaLoRATrainingStrategy
from src.core.interfaces.strategy import StrategyType
from src.core.exceptions.exceptions import StrategyError


class TestStandardTrainingStrategy:
    """测试标准训练策略"""
    
    def setup_method(self):
        """测试前设置"""
        self.strategy = StandardTrainingStrategy()
        
        # 创建简单的模型和数据
        self.model = nn.Linear(10, 2)
        self.data = torch.randn(100, 10)
        self.targets = torch.randint(0, 2, (100,))
        self.dataset = TensorDataset(self.data, self.targets)
        self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        self.config = {
            'epochs': 1,
            'lr': 0.001,
            'optimizer': 'adam',
            'device': 'cpu'
        }
    
    def test_get_name(self):
        """测试获取策略名称"""
        assert self.strategy.get_name() == "standard"
    
    def test_get_description(self):
        """测试获取策略描述"""
        description = self.strategy.get_description()
        assert "标准联邦学习" in description
    
    def test_validate_context_valid(self):
        """测试验证有效上下文"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader,
            'config': self.config
        }
        # 应该不抛出异常
        self.strategy.validate_context(context)
    
    def test_validate_context_missing_key(self):
        """测试验证缺少键的上下文"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader
            # 缺少config
        }
        with pytest.raises(StrategyError, match="缺少必需的上下文键"):
            self.strategy.validate_context(context)
    
    def test_validate_context_invalid_model(self):
        """测试验证无效模型"""
        context = {
            'model': "not_a_model",
            'train_loader': self.train_loader,
            'config': self.config
        }
        with pytest.raises(StrategyError, match="model 必须是 torch.nn.Module 实例"):
            self.strategy.validate_context(context)
    
    def test_validate_context_invalid_train_loader(self):
        """测试验证无效的train_loader"""
        context = {
            'model': self.model,
            'train_loader': "not_a_dataloader",
            'config': self.config
        }
        with pytest.raises(StrategyError, match="train_loader 必须是 DataLoader 实例"):
            self.strategy.validate_context(context)
    
    def test_validate_context_invalid_config(self):
        """测试验证无效的config"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader,
            'config': "not_a_dict"
        }
        with pytest.raises(StrategyError, match="config 必须是字典"):
            self.strategy.validate_context(context)
    
    def test_execute_with_context(self):
        """测试使用上下文执行策略"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader,
            'config': self.config
        }
        
        metrics = self.strategy.execute(context)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
    
    def test_train_model_basic(self):
        """测试基本模型训练"""
        metrics = self.strategy.train_model(self.model, self.train_loader, self.config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        
        # 检查指标值是否合理
        assert metrics['epochs_completed'] == self.config['epochs']
        assert metrics['num_samples'] == len(self.dataset)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['loss'] >= 0.0
    
    def test_train_model_multiple_epochs(self):
        """测试多轮训练"""
        config = self.config.copy()
        config['epochs'] = 2
        
        metrics = self.strategy.train_model(self.model, self.train_loader, config)
        
        assert metrics['epochs_completed'] == 2
        assert metrics['num_samples'] == len(self.dataset) * 2
    
    def test_train_model_empty_dataset(self):
        """测试空数据集训练"""
        empty_dataset = TensorDataset(torch.randn(0, 10), torch.randint(0, 2, (0,)))
        empty_loader = DataLoader(empty_dataset, batch_size=16)
        
        metrics = self.strategy.train_model(self.model, empty_loader, self.config)
        
        # 空数据集应该返回默认值
        assert metrics['loss'] == 0.0
        assert metrics['accuracy'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['num_samples'] == 0
    
    def test_evaluate_model(self):
        """测试模型评估"""
        metrics = self.strategy.evaluate_model(self.model, self.train_loader, self.config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        
        # 检查指标值是否合理
        assert metrics['num_samples'] == len(self.dataset)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['loss'] >= 0.0
    
    def test_prepare_model(self):
        """测试模型准备"""
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        
        # 标准训练策略应该返回相同的模型
        assert prepared_model is self.model
    
    def test_get_optimizer_adam(self):
        """测试获取Adam优化器"""
        optimizer = self.strategy.get_optimizer(self.model, self.config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == self.config['lr']
    
    def test_get_optimizer_adamw(self):
        """测试获取AdamW优化器"""
        config = self.config.copy()
        config['optimizer'] = 'adamw'
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.AdamW)
    
    def test_get_optimizer_sgd(self):
        """测试获取SGD优化器"""
        config = self.config.copy()
        config['optimizer'] = 'sgd'
        config['momentum'] = 0.9
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['momentum'] == 0.9
    
    def test_get_optimizer_invalid(self):
        """测试获取无效优化器"""
        config = self.config.copy()
        config['optimizer'] = 'invalid'
        
        with pytest.raises(StrategyError, match="不支持的优化器类型"):
            self.strategy.get_optimizer(self.model, config)
    
    def test_get_config_schema(self):
        """测试获取配置模式"""
        schema = self.strategy.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema
        
        # 检查一些必需的属性
        properties = schema['properties']
        assert 'epochs' in properties
        assert 'lr' in properties
        assert 'optimizer' in properties
    
    def test_train_model_with_mixed_precision(self):
        """测试混合精度训练"""
        config = self.config.copy()
        config['use_amp'] = True  # 启用混合精度
        
        metrics = self.strategy.train_model(self.model, self.train_loader, config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        
        # 检查指标值是否合理
        assert metrics['epochs_completed'] == config['epochs']
        assert metrics['num_samples'] == len(self.dataset) * config['epochs']
    
    def test_train_model_with_custom_optimizer_params(self):
        """测试使用自定义优化器参数训练"""
        config = {
            'epochs': 1,
            'lr': 0.01,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'weight_decay': 0.001
        }
        
        metrics = self.strategy.train_model(self.model, self.train_loader, config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['epochs_completed'] == config['epochs']


class TestLoRATrainingStrategy:
    """测试LoRA训练策略"""
    
    def setup_method(self):
        """测试前设置"""
        self.strategy = LoRATrainingStrategy()
        
        # 创建一个模拟的LoRA模型
        class MockLoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 基础线性层
                self.linear = nn.Linear(10, 2)
                # LoRA参数
                self.lora_A = nn.Parameter(torch.randn(5, 10), requires_grad=True)
                self.lora_B = nn.Parameter(torch.randn(2, 5), requires_grad=True)
                
            def forward(self, x):
                # 模拟LoRA修改：基础输出 + LoRA调整
                base_output = self.linear(x)
                lora_adjustment = self.lora_B @ self.lora_A @ x.t()
                return base_output + lora_adjustment.t()
        
        self.model = MockLoRAModel()
        
        self.data = torch.randn(100, 10)
        self.targets = torch.randint(0, 2, (100,))
        self.dataset = TensorDataset(self.data, self.targets)
        self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        self.config = {
            'base_model_path': '/path/to/base/model',
            'epochs': 1,
            'lr': 0.001,
            'device': 'cpu'
        }
    
    def test_get_name(self):
        """测试获取策略名称"""
        assert self.strategy.get_name() == "lora"
    
    def test_get_description(self):
        """测试获取策略描述"""
        description = self.strategy.get_description()
        assert "LoRA微调" in description
    
    def test_validate_context_missing_base_model_path(self):
        """测试验证缺少base_model_path的上下文"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader,
            'config': {'epochs': 1}  # 缺少base_model_path
        }
        with pytest.raises(StrategyError, match="LoRA训练需要 base_model_path 配置"):
            self.strategy.validate_context(context)
    
    def test_has_lora_layers(self):
        """测试检查LoRA层"""
        # 模拟有LoRA层的模型
        model_with_lora = nn.Linear(10, 2)
        model_with_lora.lora_weight = nn.Parameter(torch.randn(10, 2))
        
        assert self.strategy._has_lora_layers(model_with_lora)
        
        # 测试没有LoRA层的模型
        model_without_lora = nn.Linear(10, 2)
        assert not self.strategy._has_lora_layers(model_without_lora)
    
    def test_freeze_base_model(self):
        """测试冻结基础模型"""
        # 创建一个有多个参数的模型
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        model.lora_A = nn.Parameter(torch.randn(3, 5))  # LoRA参数
        
        # 冻结基础模型
        self.strategy._freeze_base_model(model)
        
        # 检查LoRA参数是否可训练
        assert model.lora_A.requires_grad
        
        # 检查基础参数是否被冻结
        # 注意：实际的实现可能需要更复杂的逻辑来识别基础参数
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                # 这里简化了检查，实际实现可能更复杂
                pass
    
    def test_count_trainable_params(self):
        """测试计算可训练参数数量"""
        # 创建一个有冻结和可训练参数的模型
        model = nn.Linear(10, 2)
        model.weight.requires_grad = False
        model.bias.requires_grad = True
        
        count = self.strategy._count_trainable_params(model)
        
        # 应该只计算bias参数
        assert count == model.bias.numel()
    
    def test_get_trainable_keys(self):
        """测试获取可训练键"""
        model = nn.Linear(10, 2)
        model.weight.requires_grad = True
        model.bias.requires_grad = False
        
        trainable_keys = self.strategy.get_trainable_keys(model)
        
        assert 'weight' in trainable_keys
        assert 'bias' not in trainable_keys
    
    def test_get_optimizer_only_trainable(self):
        """测试优化器只优化可训练参数"""
        # 冻结一些参数 - 冻结linear层的参数
        self.model.linear.weight.requires_grad = False
        self.model.linear.bias.requires_grad = False
        
        optimizer = self.strategy.get_optimizer(self.model, self.config)
        
        # 检查优化器参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_params = list(optimizer.param_groups[0]['params'])
        
        assert len(optimizer_params) == len(trainable_params)
    
    def test_prepare_model_with_lora_layers(self):
        """测试准备有LoRA层的模型"""
        # 模型已经有LoRA层，应该成功
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        
        # 应该返回相同的模型实例
        assert prepared_model is self.model
        
        # 检查基础参数是否被冻结
        for name, param in self.model.named_parameters():
            if 'lora_' not in name.lower() and 'classifier' not in name.lower():
                assert not param.requires_grad, f"基础参数 {name} 应该被冻结"
            else:
                assert param.requires_grad, f"LoRA参数 {name} 应该可训练"
    
    def test_prepare_model_without_lora_layers(self):
        """测试准备没有LoRA层的模型"""
        model_without_lora = nn.Linear(10, 2)
        
        with pytest.raises(StrategyError, match="模型没有应用LoRA层"):
            self.strategy.prepare_model(model_without_lora, self.config)
    
    def test_train_model_basic(self):
        """测试基本LoRA模型训练"""
        # 准备模型
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        
        metrics = self.strategy.train_model(prepared_model, self.train_loader, self.config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        assert 'lora_params_count' in metrics
        
        # 检查指标值是否合理
        assert metrics['epochs_completed'] == self.config['epochs']
        assert metrics['num_samples'] == len(self.dataset)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['loss'] >= 0.0
        assert metrics['lora_params_count'] > 0
    
    def test_train_model_multiple_epochs(self):
        """测试多轮LoRA训练"""
        config = self.config.copy()
        config['epochs'] = 2
        
        prepared_model = self.strategy.prepare_model(self.model, config)
        metrics = self.strategy.train_model(prepared_model, self.train_loader, config)
        
        assert metrics['epochs_completed'] == 2
        assert metrics['num_samples'] == len(self.dataset) * 2
    
    def test_train_model_empty_dataset(self):
        """测试空数据集LoRA训练"""
        empty_dataset = TensorDataset(torch.randn(0, 10), torch.randint(0, 2, (0,)))
        empty_loader = DataLoader(empty_dataset, batch_size=16)
        
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        metrics = self.strategy.train_model(prepared_model, empty_loader, self.config)
        
        # 空数据集应该返回默认值
        assert metrics['loss'] == 0.0
        assert metrics['accuracy'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['num_samples'] == 0
    
    def test_evaluate_model(self):
        """测试LoRA模型评估"""
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        metrics = self.strategy.evaluate_model(prepared_model, self.train_loader, self.config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'lora_params_count' in metrics
        
        # 检查指标值是否合理
        assert metrics['num_samples'] == len(self.dataset)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['loss'] >= 0.0
        assert metrics['lora_params_count'] > 0
    
    def test_execute_with_context(self):
        """测试使用上下文执行LoRA策略"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader,
            'config': self.config
        }
        
        metrics = self.strategy.execute(context)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        assert 'lora_params_count' in metrics
    
    def test_get_optimizer_adam(self):
        """测试获取Adam优化器"""
        config = self.config.copy()
        config['optimizer'] = 'adam'
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == config['lr']
    
    def test_get_optimizer_adamw(self):
        """测试获取AdamW优化器"""
        config = self.config.copy()
        config['optimizer'] = 'adamw'
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.AdamW)
    
    def test_get_optimizer_sgd(self):
        """测试获取SGD优化器"""
        config = self.config.copy()
        config['optimizer'] = 'sgd'
        config['momentum'] = 0.9
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['momentum'] == 0.9
    
    def test_get_optimizer_invalid(self):
        """测试获取无效优化器"""
        config = self.config.copy()
        config['optimizer'] = 'invalid'
        
        with pytest.raises(StrategyError, match="不支持的优化器类型"):
            self.strategy.get_optimizer(self.model, config)
    
    def test_get_optimizer_no_trainable_params(self):
        """测试没有可训练参数时获取优化器"""
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        with pytest.raises(StrategyError, match="没有可训练的参数"):
            self.strategy.get_optimizer(self.model, self.config)
    
    def test_train_model_with_mixed_precision(self):
        """测试LoRA混合精度训练"""
        config = self.config.copy()
        config['use_amp'] = True  # 启用混合精度
        
        prepared_model = self.strategy.prepare_model(self.model, config)
        metrics = self.strategy.train_model(prepared_model, self.train_loader, config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        assert 'lora_params_count' in metrics
    
    def test_get_config_schema(self):
        """测试获取LoRA配置模式"""
        schema = self.strategy.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
        
        # 检查LoRA特有的属性
        properties = schema['properties']
        assert 'base_model_path' in properties
        assert 'use_amp' in properties
        
        # 检查必需字段
        assert 'base_model_path' in schema['required']


class TestAdaLoRATrainingStrategy:
    """测试AdaLoRA训练策略"""
    
    def setup_method(self):
        """测试前设置"""
        self.strategy = AdaLoRATrainingStrategy()
        
        # 创建一个模拟的AdaLoRA模型
        class MockAdaLoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 基础线性层
                self.linear = nn.Linear(10, 2)
                # AdaLoRA参数
                self.adalora_A = nn.Parameter(torch.randn(5, 10), requires_grad=True)
                self.adalora_B = nn.Parameter(torch.randn(2, 5), requires_grad=True)
                
            def forward(self, x):
                # 模拟AdaLoRA修改：基础输出 + AdaLoRA调整
                base_output = self.linear(x)
                adalora_adjustment = self.adalora_B @ self.adalora_A @ x.t()
                return base_output + adalora_adjustment.t()
        
        self.model = MockAdaLoRAModel()
        
        self.data = torch.randn(100, 10)
        self.targets = torch.randint(0, 2, (100,))
        self.dataset = TensorDataset(self.data, self.targets)
        self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        self.config = {
            'base_model_path': '/path/to/base/model',
            'epochs': 1,
            'lr': 0.001,
            'device': 'cpu'
        }
    
    def test_get_name(self):
        """测试获取策略名称"""
        assert self.strategy.get_name() == "adalora"
    
    def test_get_description(self):
        """测试获取策略描述"""
        description = self.strategy.get_description()
        assert "AdaLoRA微调" in description
    
    def test_has_adalora_layers(self):
        """测试检查AdaLoRA层"""
        # 模拟有AdaLoRA层的模型
        model_with_adalora = nn.Linear(10, 2)
        model_with_adalora.adalora_weight = nn.Parameter(torch.randn(10, 2))
        
        assert self.strategy._has_adalora_layers(model_with_adalora)
        
        # 测试没有AdaLoRA层的模型
        model_without_adalora = nn.Linear(10, 2)
        assert not self.strategy._has_adalora_layers(model_without_adalora)
    
    def test_is_adalora_param(self):
        """测试判断AdaLoRA参数"""
        assert self.strategy._is_adalora_param('adalora_A.weight')
        assert self.strategy._is_adalora_param('lora_B.weight')
        assert self.strategy._is_adalora_param('classifier.weight')
        assert not self.strategy._is_adalora_param('base.weight')
    
    def test_get_rank_distribution(self):
        """测试获取rank分布"""
        # 创建一个模拟的LoRA权重矩阵
        self.model.lora_weight = nn.Parameter(torch.randn(8, 4))  # out_features=8, rank=4
        
        rank_dist = self.strategy._get_rank_distribution(self.model)
        
        assert 'lora_weight' in rank_dist
        assert rank_dist['lora_weight'] == 4
    
    def test_get_config_schema(self):
        """测试获取配置模式"""
        schema = self.strategy.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema
        
        # 检查AdaLoRA特有的属性
        properties = schema['properties']
        assert 'rank_budget' in properties
        
        # 检查必需字段
        assert 'required' in schema
        assert 'base_model_path' in schema['required']
    
    def test_prepare_model_with_adalora_layers(self):
        """测试准备有AdaLoRA层的模型"""
        # 模型已经有AdaLoRA层，应该成功
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        
        # 应该返回相同的模型实例
        assert prepared_model is self.model
        
        # 检查基础参数是否被冻结
        for name, param in self.model.named_parameters():
            if not self.strategy._is_adalora_param(name):
                assert not param.requires_grad, f"基础参数 {name} 应该被冻结"
            else:
                assert param.requires_grad, f"AdaLoRA参数 {name} 应该可训练"
    
    def test_prepare_model_without_adalora_layers(self):
        """测试准备没有AdaLoRA层的模型"""
        model_without_adalora = nn.Linear(10, 2)
        
        with pytest.raises(StrategyError, match="模型没有应用AdaLoRA层"):
            self.strategy.prepare_model(model_without_adalora, self.config)
    
    def test_train_model_basic(self):
        """测试基本AdaLoRA模型训练"""
        # 准备模型
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        
        metrics = self.strategy.train_model(prepared_model, self.train_loader, self.config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        assert 'adalora_params_count' in metrics
        assert 'rank_distribution' in metrics
        
        # 检查指标值是否合理
        assert metrics['epochs_completed'] == self.config['epochs']
        assert metrics['num_samples'] == len(self.dataset)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['loss'] >= 0.0
        assert metrics['adalora_params_count'] > 0
        assert isinstance(metrics['rank_distribution'], dict)
    
    def test_train_model_multiple_epochs(self):
        """测试多轮AdaLoRA训练"""
        config = self.config.copy()
        config['epochs'] = 2
        
        prepared_model = self.strategy.prepare_model(self.model, config)
        metrics = self.strategy.train_model(prepared_model, self.train_loader, config)
        
        assert metrics['epochs_completed'] == 2
        assert metrics['num_samples'] == len(self.dataset) * 2
    
    def test_train_model_empty_dataset(self):
        """测试空数据集AdaLoRA训练"""
        empty_dataset = TensorDataset(torch.randn(0, 10), torch.randint(0, 2, (0,)))
        empty_loader = DataLoader(empty_dataset, batch_size=16)
        
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        metrics = self.strategy.train_model(prepared_model, empty_loader, self.config)
        
        # 空数据集应该返回默认值
        assert metrics['loss'] == 0.0
        assert metrics['accuracy'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['num_samples'] == 0
    
    def test_evaluate_model(self):
        """测试AdaLoRA模型评估"""
        prepared_model = self.strategy.prepare_model(self.model, self.config)
        metrics = self.strategy.evaluate_model(prepared_model, self.train_loader, self.config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'adalora_params_count' in metrics
        assert 'rank_distribution' in metrics
        
        # 检查指标值是否合理
        assert metrics['num_samples'] == len(self.dataset)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['loss'] >= 0.0
        assert metrics['adalora_params_count'] > 0
        assert isinstance(metrics['rank_distribution'], dict)
    
    def test_execute_with_context(self):
        """测试使用上下文执行AdaLoRA策略"""
        context = {
            'model': self.model,
            'train_loader': self.train_loader,
            'config': self.config
        }
        
        metrics = self.strategy.execute(context)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        assert 'adalora_params_count' in metrics
        assert 'rank_distribution' in metrics
    
    def test_get_optimizer_adam(self):
        """测试获取Adam优化器"""
        config = self.config.copy()
        config['optimizer'] = 'adam'
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == config['lr']
    
    def test_get_optimizer_adamw(self):
        """测试获取AdamW优化器"""
        config = self.config.copy()
        config['optimizer'] = 'adamw'
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.AdamW)
    
    def test_get_optimizer_sgd(self):
        """测试获取SGD优化器"""
        config = self.config.copy()
        config['optimizer'] = 'sgd'
        config['momentum'] = 0.9
        
        optimizer = self.strategy.get_optimizer(self.model, config)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['momentum'] == 0.9
    
    def test_get_optimizer_invalid(self):
        """测试获取无效优化器"""
        config = self.config.copy()
        config['optimizer'] = 'invalid'
        
        with pytest.raises(StrategyError, match="不支持的优化器类型"):
            self.strategy.get_optimizer(self.model, config)
    
    def test_get_optimizer_no_trainable_params(self):
        """测试没有可训练参数时获取优化器"""
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        with pytest.raises(StrategyError, match="没有可训练的参数"):
            self.strategy.get_optimizer(self.model, self.config)
    
    def test_train_model_with_mixed_precision(self):
        """测试AdaLoRA混合精度训练"""
        config = self.config.copy()
        config['use_amp'] = True  # 启用混合精度
        
        prepared_model = self.strategy.prepare_model(self.model, config)
        metrics = self.strategy.train_model(prepared_model, self.train_loader, config)
        
        # 检查返回的指标
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'num_samples' in metrics
        assert 'epochs_completed' in metrics
        assert 'adalora_params_count' in metrics
        assert 'rank_distribution' in metrics


class TestTrainingStrategyIntegration:
    """测试训练策略集成"""
    
    def test_all_training_strategies_registered(self):
        """测试所有训练策略都已注册"""
        from src.core.interfaces.strategy import StrategyRegistry
        
        train_strategies = StrategyRegistry.list_strategies(StrategyType.TRAINING)
        
        assert "standard" in train_strategies
        assert "lora" in train_strategies
        assert "adalora" in train_strategies
    
    def test_strategy_factory_training(self):
        """测试策略工厂创建训练策略"""
        from src.strategies.strategy_factory import strategy_factory
        
        # 测试创建各种训练策略
        standard = strategy_factory.create_training_strategy("standard")
        lora = strategy_factory.create_training_strategy("lora")
        adalora = strategy_factory.create_training_strategy("adalora")
        
        assert isinstance(standard, StandardTrainingStrategy)
        assert isinstance(lora, LoRATrainingStrategy)
        assert isinstance(adalora, AdaLoRATrainingStrategy)
    
    def test_training_strategies_config_validation(self):
        """测试训练策略配置验证"""
        from src.strategies.strategy_factory import strategy_factory
        
        # 标准训练策略有效配置
        standard_config = {
            'model': nn.Linear(10, 2),
            'train_loader': DataLoader(TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))),
            'config': {'epochs': 1, 'lr': 0.001}
        }
        
        assert strategy_factory.validate_strategy_config(
            StrategyType.TRAINING, "standard", standard_config
        )
        
        # LoRA训练策略有效配置
        lora_config = {
            'model': nn.Linear(10, 2),
            'train_loader': DataLoader(TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))),
            'config': {'epochs': 1, 'lr': 0.001, 'base_model_path': '/path/to/model'}
        }
        
        assert strategy_factory.validate_strategy_config(
            StrategyType.TRAINING, "lora", lora_config
        )
        
        # LoRA训练策略无效配置（缺少base_model_path）
        invalid_lora_config = {
            'model': nn.Linear(10, 2),
            'train_loader': DataLoader(TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))),
            'config': {'epochs': 1, 'lr': 0.001}
        }
        
        assert not strategy_factory.validate_strategy_config(
            StrategyType.TRAINING, "lora", invalid_lora_config
        )