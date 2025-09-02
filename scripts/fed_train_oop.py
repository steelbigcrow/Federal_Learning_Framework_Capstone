#!/usr/bin/env python3
"""
联邦学习训练脚本 - OOP架构版本

该脚本使用新的OOP架构执行联邦学习训练过程，
提供更好的可扩展性和模块化设计。

使用示例：
    python scripts/fed_train_oop.py --arch-config configs/arch/mnist_mlp.yaml --train-config configs/federated.yaml
"""

import os
import sys
from copy import deepcopy

# 允许直接以 `python scripts/fed_train_oop.py` 运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_two_configs, build_argparser, validate_training_config, set_seed, PathManager, get_device
from src.datasets import get_mnist_datasets, get_imdb_splits, partition_mnist_label_shift, partition_imdb_label_shift
from src.models import create_model
from src.training import inject_lora_modules, mark_only_lora_as_trainable, load_base_model_checkpoint
from src.training.adalora_utils import inject_adalora_modules, mark_only_adalora_as_trainable, create_rank_allocator
from src.factories.component_factory import ComponentFactory
from src.strategies.training import StandardTrainingStrategy, LoRATrainingStrategy, AdaLoRATrainingStrategy
from src.strategies.aggregation import FedAvgStrategy, LoRAFedAvgStrategy, AdaLoRAFedAvgStrategy


def main():
    """主函数：使用OOP架构执行联邦学习训练"""
    # 设置命令行参数解析器
    parser = build_argparser()
    parser.add_argument('--use-lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--use-adalora', action='store_true', help='Enable AdaLoRA fine-tuning')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching')
    parser.add_argument('--data-cache-dir', type=str, default='./data_cache', help='Data cache directory')
    parser.add_argument('--auto-eval', action='store_true', help='Automatically evaluate model after training')
    parser.add_argument('--use-strategy-pattern', action='store_true', help='Use strategy pattern (default: factory pattern)')
    args = parser.parse_args()

    # 加载配置文件
    cfg = load_two_configs(args.arch_config, args.train_config, args.override)
    
    # 验证配置
    validate_training_config(cfg)
    
    # 设置随机种子
    set_seed(cfg.get('seed', 42))
    
    # 获取设备
    device = get_device()
    
    # 创建路径管理器
    pm = PathManager(
        root=cfg.get('logging', {}).get('root', './outputs'),
        dataset_name=cfg['dataset']['name'],
        model_name=cfg['model']['name'],
        use_lora=args.use_lora,
        use_adalora=args.use_adalora
    )
    
    # 创建全局模型
    global_model = create_model(
        dataset=cfg['dataset']['name'],
        model=cfg['model']['name'],
        cfg=cfg['model'],
        extra={'device': device}
    )
    
    # 处理LoRA/AdaLoRA
    lora_cfg = {}
    adalora_cfg = {}
    
    if args.use_adalora:
        # AdaLoRA处理
        adalora_cfg = cfg.get('adalora', {})
        base_model_path = adalora_cfg.get('base_model_path')
        
        if base_model_path:
            try:
                load_base_model_checkpoint(global_model, base_model_path, strict=False)
            except Exception as e:
                print(f"[Warning] Failed to load base model: {e}")
        
        # 注入AdaLoRA模块
        replaced_modules = inject_adalora_modules(
            global_model,
            r=adalora_cfg.get('initial_r', 8),
            alpha=adalora_cfg.get('alpha', 16),
            target_modules=adalora_cfg.get('target_modules', ['Linear', 'Embedding'])
        )
        mark_only_adalora_as_trainable(global_model)
        adalora_cfg['replaced_modules'] = replaced_modules
        
    elif args.use_lora:
        # LoRA处理
        lora_cfg = cfg.get('lora', {})
        base_model_path = lora_cfg.get('base_model_path')
        
        if base_model_path:
            try:
                load_base_model_checkpoint(global_model, base_model_path, strict=False)
            except Exception as e:
                print(f"[Warning] Failed to load base model: {e}")
        
        # 注入LoRA模块
        replaced_modules = inject_lora_modules(
            global_model,
            r=lora_cfg.get('r', 8),
            alpha=lora_cfg.get('alpha', 16),
            target_modules=lora_cfg.get('target_modules', ['Linear', 'Embedding'])
        )
        mark_only_lora_as_trainable(global_model)
        lora_cfg['replaced_modules'] = replaced_modules
    
    # 准备数据集
    if cfg['dataset']['name'] == 'mnist':
        train_dataset, test_dataset = get_mnist_datasets(
            data_cache_dir=args.data_cache_dir if not args.no_cache else None
        )
        client_datasets = partition_mnist_label_shift(
            train_dataset, 
            num_clients=cfg['federated']['num_clients']
        )
    else:  # imdb
        train_dataset, test_dataset = get_imdb_splits(
            data_cache_dir=args.data_cache_dir if not args.no_cache else None
        )
        client_datasets = partition_imdb_label_shift(
            train_dataset,
            num_clients=cfg['federated']['num_clients']
        )
    
    # 选择使用策略模式还是工厂模式
    if args.use_strategy_pattern:
        print("[Info] Using Strategy Pattern")
        
        # 创建训练策略
        if args.use_adalora:
            training_strategy = AdaLoRATrainingStrategy(adalora_cfg)
            aggregation_strategy = AdaLoRAFedAvgStrategy()
        elif args.use_lora:
            training_strategy = LoRATrainingStrategy(lora_cfg)
            aggregation_strategy = LoRAFedAvgStrategy()
        else:
            training_strategy = StandardTrainingStrategy()
            aggregation_strategy = FedAvgStrategy()
        
        # 使用策略模式创建服务器
        from src.implementations.servers.federated_server import FederatedServer
        
        # 创建客户端列表
        clients = []
        for cid, dataset in enumerate(client_datasets):
            client = FederatedClient(
                client_id=cid,
                train_loader=dataset['train_loader'],
                model_constructor=lambda: deepcopy(global_model),
                device=str(device),
                config=cfg['optimizer'],
                training_strategy=training_strategy
            )
            clients.append(client)
        
        # 创建服务器
        server = FederatedServer(
            model_constructor=lambda: deepcopy(global_model),
            clients=clients,
            path_manager=pm,
            config={
                'federated': cfg['federated'],
                'lora_cfg': lora_cfg,
                'adalora_cfg': adalora_cfg,
                'save_client_each_round': cfg.get('save_client_each_round', True),
                'model_info': {
                    'dataset': cfg['dataset']['name'],
                    'model_type': cfg['model']['name']
                }
            },
            device=str(device),
            aggregation_strategy=aggregation_strategy
        )
        
    else:
        print("[Info] Using Factory Pattern")
        
        # 使用工厂模式创建整个联邦学习系统
        factory = ComponentFactory()
        
        # 配置工厂
        factory_config = {
            'dataset': {
                'name': cfg['dataset']['name'],
                'config': {
                    'batch_size': cfg['dataset']['batch_size'],
                    'num_clients': cfg['federated']['num_clients'],
                    'partition_method': 'label_shift' if cfg['dataset']['name'] == 'mnist' else 'balanced',
                    'data_cache_dir': args.data_cache_dir if not args.no_cache else None
                }
            },
            'model': {
                'constructor': lambda: create_model(
                    dataset=cfg['dataset']['name'],
                    model=cfg['model']['name'],
                    cfg=cfg['model'],
                    extra={'device': device}
                ),
                'config': cfg['model']
            },
            'server': {
                'type': 'federated_server',
                'config': {
                    'federated': cfg['federated'],
                    'lora_cfg': lora_cfg,
                    'adalora_cfg': adalora_cfg,
                    'save_client_each_round': cfg.get('save_client_each_round', True),
                    'model_info': {
                        'dataset': cfg['dataset']['name'],
                        'model_type': cfg['model']['name']
                    }
                }
            },
            'training': {
                'mode': 'adalora' if args.use_adalora else 'lora' if args.use_lora else 'standard',
                'config': adalora_cfg if args.use_adalora else lora_cfg if args.use_lora else {}
            },
            'path_manager': pm,
            'device': str(device)
        }
        
        # 创建联邦学习系统
        server = factory.create_federated_system(factory_config)
    
    # 运行联邦学习
    print(f"[Info] Starting federated learning with {len(server.clients)} clients")
    print(f"[Info] Dataset: {cfg['dataset']['name']}, Model: {cfg['model']['name']}")
    print(f"[Info] Training mode: {'AdaLoRA' if args.use_adalora else 'LoRA' if args.use_lora else 'Standard'}")
    
    server.run(
        num_rounds=cfg['federated']['num_rounds'],
        local_epochs=cfg['federated']['local_epochs']
    )
    
    print("[Info] Federated learning completed successfully!")
    
    # 自动评估
    if args.auto_eval:
        print("[Info] Running automatic evaluation...")
        # TODO: 实现自动评估逻辑
        pass


if __name__ == '__main__':
    main()