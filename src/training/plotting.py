"""
绘图模块，用于生成客户端指标图表

该模块提供联邦学习过程中客户端训练指标的可视化功能，
包括训练准确率、F1分数、损失值等指标的趋势图表。
"""
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于无GUI环境


def plot_client_metrics(client_id: int, round_metrics_history: List[Dict], save_path: str) -> None:
    """
    为指定客户端绘制5个指标的图表
    
    Args:
        client_id: 客户端ID
        round_metrics_history: 历史轮次指标列表，每个元素包含一轮的指标
        save_path: 图片保存路径
    """
    if not round_metrics_history:
        return
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 提取每个epoch的数据（新格式）
    all_epochs = []
    train_acc = []
    train_f1 = []
    train_loss = []
    test_acc = []
    test_f1 = []
    
    epoch_counter = 0
    for round_idx, round_data in enumerate(round_metrics_history):
        # 检查是否有epoch_history（新格式）
        if 'epoch_history' in round_data and round_data['epoch_history']:
            for epoch_data in round_data['epoch_history']:
                epoch_counter += 1
                all_epochs.append(epoch_counter)
                train_acc.append(epoch_data.get('train_acc', 0))
                train_f1.append(epoch_data.get('train_f1', 0))
                train_loss.append(epoch_data.get('train_loss', 0))
                test_acc.append(epoch_data.get('test_acc', 0))
                test_f1.append(epoch_data.get('test_f1', 0))
        else:
            # 兼容旧格式：每个round作为一个数据点
            epoch_counter += 1
            all_epochs.append(epoch_counter)
            train_acc.append(round_data.get('train_acc', 0))
            train_f1.append(round_data.get('train_f1', 0))
            train_loss.append(round_data.get('train_loss', 0))
            test_acc.append(round_data.get('test_acc', 0))
            test_f1.append(round_data.get('test_f1', 0))
    
    if not all_epochs:
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Client {client_id} Metrics Overview', fontsize=16, fontweight='bold')
    
    # 设置中文字体（可选，如果系统支持）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 绘制训练准确率
    axes[0, 0].plot(all_epochs, train_acc, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Training Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # 绘制训练F1
    axes[0, 1].plot(all_epochs, train_f1, 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Training F1 Score', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # 绘制训练损失
    axes[0, 2].plot(all_epochs, train_loss, 'r-o', linewidth=2, markersize=4)
    axes[0, 2].set_title('Training Loss', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 绘制测试准确率
    axes[1, 0].plot(all_epochs, test_acc, 'c-o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Test Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 绘制测试F1
    axes[1, 1].plot(all_epochs, test_f1, 'm-o', linewidth=2, markersize=4)
    axes[1, 1].set_title('Test F1 Score', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # 绘制训练vs测试准确率对比
    axes[1, 2].plot(all_epochs, train_acc, 'b-o', label='Train Acc', linewidth=2, markersize=4)
    axes[1, 2].plot(all_epochs, test_acc, 'c-o', label='Test Acc', linewidth=2, markersize=4)
    axes[1, 2].set_title('Accuracy Comparison', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 释放内存


def load_client_metrics_history(client_metrics_dir: str, current_round: int) -> List[Dict]:
    """
    加载客户端的历史指标数据

    Args:
        client_metrics_dir: 客户端指标目录路径
        current_round: 当前轮次

    Returns:
        历史指标数据列表，每个元素包含一轮的训练指标
    """
    history = []
    
    for round_num in range(1, current_round + 1):
        metrics_file = os.path.join(client_metrics_dir, f"round_{round_num}.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    history.append(metrics)
            except Exception as e:
                # 添加空字典以保持轮次对齐
                history.append({})
        else:
            # 添加空字典以保持轮次对齐
            history.append({})
    
    return history


def plot_all_clients_metrics(client_ids: List[int], metrics_clients_dir: str, plots_dir: str, current_round: int) -> None:
    """
    为所有客户端生成指标图表

    Args:
        client_ids: 客户端ID列表
        metrics_clients_dir: 客户端指标根目录路径
        plots_dir: 图片保存根目录路径
        current_round: 当前轮次，用于确定要加载的指标数据范围
    """
    for client_id in client_ids:
        client_metrics_dir = os.path.join(metrics_clients_dir, f"client_{client_id}")
        metrics_history = load_client_metrics_history(client_metrics_dir, current_round)
        
        if metrics_history:
            # 生成图片文件名
            plot_filename = f"client_{client_id}_round_{current_round}_metrics.png"
            plot_path = os.path.join(plots_dir, f"client_{client_id}", plot_filename)
            
            # 绘制图表
            plot_client_metrics(client_id, metrics_history, plot_path)
