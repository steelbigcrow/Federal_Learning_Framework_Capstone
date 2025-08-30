"""
联邦学习训练脚本

该脚本是联邦学习框架的主入口程序，负责协调整个联邦学习训练过程。
支持标准联邦学习和LoRA微调的联邦学习，可以处理MNIST和IMDB数据集。
"""

import os
from copy import deepcopy
import sys

# 允许直接以 `python scripts/fed_train.py` 运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_two_configs, build_argparser, validate_training_config
from src.utils.seed import set_seed
from src.utils.paths import PathManager
from src.utils.device import get_device
from src.datasets.mnist import get_mnist_datasets
from src.datasets.imdb import get_imdb_splits
from src.datasets.partition import partition_mnist_label_shift, partition_imdb_label_shift
from src.models.registry import create_model
from src.training.lora_utils import inject_lora_modules, mark_only_lora_as_trainable, load_base_model_checkpoint
from src.federated.client import Client
from src.federated.server import Server


def main():
	"""
	主函数：执行联邦学习训练流程

	该函数负责：
	1. 解析命令行参数和配置文件
	2. 验证配置的合法性
	3. 准备数据集和模型
	4. 配置LoRA参数（如果使用）
	5. 创建客户端和服务器
	6. 执行联邦学习训练
	"""
	# 设置命令行参数解析器
	parser = build_argparser()
	parser.add_argument('--use-lora', action='store_true', help='Enable LoRA fine-tuning')
	parser.add_argument('--no-cache', action='store_true', help='Disable data caching, load directly from HF')
	parser.add_argument('--data-cache-dir', type=str, default='./data_cache', help='Data cache directory')
	parser.add_argument('--auto-eval', action='store_true', help='Automatically evaluate model after training')
	args = parser.parse_args()

	# 加载配置文件（只使用双配置方式）
	cfg = load_two_configs(args.arch_config, args.train_config, args.override)

	# 验证训练配置的合法性
	try:
		validate_training_config(cfg)
	except ValueError as e:
		print(f"\nConfiguration validation failed:\n{e}")
		print("\nPlease fix the configuration and try again.")
		return

	# 设置随机种子确保结果可重复
	set_seed(cfg.get('seed', 42))

	# 自动检测最佳可用设备（GPU/CPU）
	device = get_device()

	# 获取数据集和模型名称
	ds_name = cfg.get('dataset')
	model_name = cfg.get('model')

	# 检查LoRA使用情况（以配置文件为准，命令行参数作为额外确认）
	config_use_lora = cfg.get('use_lora', False)
	if args.use_lora and not config_use_lora:
		print("Warning: --use-lora specified but config use_lora=false, using config setting")
	elif not args.use_lora and config_use_lora:
		print("Warning: config use_lora=true but --use-lora not specified, consider keeping them consistent")

	use_lora = config_use_lora
	
	# 创建路径管理器，用于管理输出文件路径
	pm = PathManager(
		root=cfg.get('logging', {}).get('root', './outputs'),
		dataset_name=ds_name,
		model_name=model_name,
		use_lora=use_lora
	)

	# 从配置中提取训练参数
	batch_size = cfg.get('train', {}).get('batch_size', cfg['federated']['batch_size'])
	local_epochs = cfg.get('train', {}).get('local_epochs', cfg['federated']['local_epochs'])
	num_clients = cfg.get('train', {}).get('num_clients', cfg['federated']['num_clients'])
	num_rounds = cfg.get('train', {}).get('num_rounds', cfg['federated']['num_rounds'])
	num_workers = cfg.get('data', {}).get('num_workers', 0)
	pin_memory = cfg.get('data', {}).get('pin_memory', False)
	save_client_each_round = cfg.get('checkpoint', {}).get('save_client_each_round', True)

	# 根据数据集类型准备数据和创建全局模型
	if ds_name == 'mnist':
		# MNIST数据集处理
		use_cache = not args.no_cache
		print(f"[Data] Using cache: {use_cache}, cache directory: {args.data_cache_dir}")
		train_ds = get_mnist_datasets(root=args.data_cache_dir, use_cache=use_cache)
		parts = partition_mnist_label_shift(train_ds, num_clients=num_clients)

		# 创建模型构造函数
		def model_ctor():
			return create_model('mnist', model_name, cfg)
		global_model = model_ctor()

	elif ds_name == 'imdb':
		# IMDB数据集处理
		use_cache = not args.no_cache
		print(f"[Data] Using cache: {use_cache}, cache directory: {args.data_cache_dir}")
		train_iter, test_iter, vocab, text_to_ids, pad_idx = get_imdb_splits(
			root=args.data_cache_dir,
			max_seq_len=cfg.get('max_seq_len', 256),
			min_freq=cfg.get('vocab_min_freq', 2),
			use_cache=use_cache
		)
		parts = partition_imdb_label_shift(train_iter, num_clients=num_clients)

		# 创建模型构造函数
		def model_ctor():
			return create_model('imdb', model_name, cfg, extra={'vocab_size': len(vocab), 'pad_idx': pad_idx})
		global_model = model_ctor()

	else:
		raise ValueError('Unknown dataset')

	# 初始化LoRA配置变量
	lora_cfg_effective = {}
	base_model_metadata = {}

	# LoRA和基模处理
	replaced_modules = []
	if use_lora:
		lora_cfg = cfg.get('lora', {})

		# 步骤1：加载基模（必须通过配置文件指定基模路径）
		base_model_path = lora_cfg.get('base_model_path')

		if base_model_path:
			# 构建完整路径
			if not os.path.isabs(base_model_path):
				base_model_path = os.path.join(cfg.get('logging', {}).get('root', './outputs'), base_model_path)
		else:
			print("[Error] LoRA微调必须在配置文件中指定 base_model_path")
			print("[Error] 请在 configs/federated.yaml 的 lora.base_model_path 中设置有效路径")
			return

			try:
				base_model_metadata = load_base_model_checkpoint(global_model, base_model_path, strict=False)
				print(f"[LoRA] Base model loaded successfully: {base_model_path}")
			except Exception as e:
				print(f"[Error] Failed to load base model: {e}")
				print(f"[LoRA] Will train from scratch")

		# 步骤2：注入LoRA模块（支持Linear和Embedding等通用组件）
		target_modules = lora_cfg.get('target_modules', ['Linear', 'Embedding'])
		replaced_modules = inject_lora_modules(
			global_model,
			r=lora_cfg.get('r', 8),
			alpha=lora_cfg.get('alpha', 16),
			dropout=lora_cfg.get('dropout', 0.0),
			target_modules=target_modules
		)

		# 步骤3：标记只有LoRA参数可训练
		mark_only_lora_as_trainable(global_model, lora_cfg.get('train_classifier_head', True))

		# 构建有效的LoRA配置字典
		lora_cfg_effective = {
			**lora_cfg,
			'replaced_modules': replaced_modules,
			'base_model_path': base_model_path,
			'base_model_metadata': base_model_metadata
		}

		print(f"[LoRA] Injected LoRA into {len(replaced_modules)} modules: {replaced_modules}")

		# 显示模型参数统计信息
		total_params = sum(p.numel() for p in global_model.parameters())
		trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
		print(f"[LoRA] Total params: {total_params:,}, trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

	# 创建联邦学习客户端
	clients = []
	if ds_name == 'mnist':
		# 为MNIST数据集创建客户端
		for cid in range(num_clients):
			train_subset = parts[cid]  # 每个客户端的训练数据子集
			train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
			clients.append(Client(cid, model_ctor, train_loader, str(device), cfg['optimizer']))

	elif ds_name == 'imdb':
		# 为IMDB数据集创建客户端
		from src.datasets.text_utils import CollateText
		for cid in range(num_clients):
			train_list = parts[cid]  # 每个客户端的训练数据列表
			collate = CollateText(text_to_ids, pad_idx, cfg.get('max_seq_len', 256))
			train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
			clients.append(Client(cid, model_ctor, train_loader, str(device), cfg['optimizer']))

	# 构建模型信息字典
	model_info = {
		'dataset': ds_name,
		'model_type': model_name,
		'model_config': cfg,
		'device': str(device)
	}

	# 创建联邦学习服务器并开始训练
	server = Server(lambda: deepcopy(global_model), clients, pm, device=str(device), lora_cfg=lora_cfg_effective, save_client_each_round=save_client_each_round, model_info=model_info)
	
	# 记录训练开始时间（用于自动评估）
	import time
	training_start_time = time.time()
	
	# 执行联邦学习训练
	server.run(num_rounds, local_epochs)
	
	# 自动评估（如果启用）
	if args.auto_eval:
		print("\n[auto-eval] Starting automatic evaluation...")
		try:
			from src.evaluation import ModelEvaluator
			
			evaluator = ModelEvaluator(
				arch_config_path=args.arch_config,
				outputs_root=cfg.get('logging', {}).get('root', './outputs'),
				device=str(device)
			)
			
			success = evaluator.auto_evaluate_after_training(
				train_config_path=args.train_config,
				use_lora=use_lora,
				run_name=cfg.get('run_name'),
				started_at=training_start_time
			)
			
			if not success:
				print("[auto-eval] Automatic evaluation failed.")
		except Exception as e:
			print(f"[auto-eval] Error during automatic evaluation: {e}")


if __name__ == '__main__':
	main()
