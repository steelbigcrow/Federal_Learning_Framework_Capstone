"""
AdaLoRA零补全聚合算法

实现基于SVD的零填充聚合策略，用于处理异构秩的AdaLoRA联邦学习聚合。
核心思想：
1. 将AdaLoRA参数重构为完整矩阵并进行SVD分解
2. 对所有客户端的U, S, VT进行零填充对齐
3. 分别对U, S, VT进行加权平均
4. 分发SVD三元组，客户端自适应重构
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def extract_adalora_matrices(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    从AdaLoRA state_dict中提取并重构完整权重矩阵
    
    Args:
        state_dict: AdaLoRA模型的状态字典
        
    Returns:
        重构后的完整权重矩阵字典 {layer_name: reconstructed_matrix}
    """
    matrices = {}
    
    # 分组AdaLoRA参数
    adalora_groups = defaultdict(dict)
    
    for key, value in state_dict.items():
        if 'lora_A' in key:
            layer_name = key.replace('.lora_A', '')
            adalora_groups[layer_name]['A'] = value
        elif 'lora_B' in key:
            layer_name = key.replace('.lora_B', '')
            adalora_groups[layer_name]['B'] = value
        elif 'ranknum' in key:
            layer_name = key.replace('.ranknum', '')
            adalora_groups[layer_name]['ranknum'] = value
    
    # 重构每一层的完整矩阵
    for layer_name, params in adalora_groups.items():
        if 'A' in params and 'B' in params:
            # W = B @ A (AdaLoRA的重构公式)
            A = params['A']  # shape: (r, d_in)
            B = params['B']  # shape: (d_out, r)
            
            # 如果有ranknum，使用动态秩
            if 'ranknum' in params:
                r = int(params['ranknum'].item())
                A = A[:r, :]
                B = B[:, :r]
            
            reconstructed_matrix = torch.matmul(B, A)  # shape: (d_out, d_in)
            matrices[layer_name] = reconstructed_matrix
    
    return matrices


def pad_svd_to_rank(U: torch.Tensor, S: torch.Tensor, VT: torch.Tensor, 
                   target_rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将SVD分解结果零填充到目标秩
    
    Args:
        U: 左奇异向量矩阵 (m, r)
        S: 奇异值向量 (r,)
        VT: 右奇异向量矩阵转置 (r, n)
        target_rank: 目标秩
        
    Returns:
        填充后的 (U_pad, S_pad, VT_pad)
    """
    current_rank = S.size(0)
    device = S.device
    
    if current_rank >= target_rank:
        # 截断到目标秩
        U_pad = U[:, :target_rank]
        S_pad = S[:target_rank]
        VT_pad = VT[:target_rank, :]
    else:
        # 零填充到目标秩
        m, n = U.size(0), VT.size(1)
        
        # 填充奇异值
        S_pad = torch.zeros(target_rank, device=device, dtype=S.dtype)
        S_pad[:current_rank] = S
        
        # 填充U矩阵
        U_pad = torch.zeros(m, target_rank, device=device, dtype=U.dtype)
        U_pad[:, :current_rank] = U
        
        # 填充VT矩阵
        VT_pad = torch.zeros(target_rank, n, device=device, dtype=VT.dtype)
        VT_pad[:current_rank, :] = VT
    
    return U_pad, S_pad, VT_pad


def zero_padding_svd_fusion(matrices_list: List[Dict[str, torch.Tensor]], 
                          weights: List[float]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    使用零填充策略融合多个客户端的AdaLoRA矩阵
    
    Args:
        matrices_list: 各客户端重构的权重矩阵列表
        weights: 各客户端的聚合权重
        
    Returns:
        融合后的SVD三元组字典 {layer_name: (U_avg, S_avg, VT_avg)}
    """
    if not matrices_list:
        raise ValueError("Empty matrices list")
    
    num_clients = len(matrices_list)
    assert len(weights) == num_clients
    
    # 确保权重归一化
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    
    # 获取所有层名
    layer_names = set(matrices_list[0].keys())
    for matrices in matrices_list[1:]:
        layer_names &= set(matrices.keys())  # 取交集
    
    fused_svd = {}
    
    for layer_name in layer_names:
        # 收集该层所有客户端的矩阵
        layer_matrices = [matrices[layer_name] for matrices in matrices_list]
        
        # 对每个矩阵进行SVD分解
        svd_list = []
        max_rank = 0
        
        for matrix in layer_matrices:
            # SVD分解
            U, S, VT = torch.svd(matrix)
            
            # 移除数值误差造成的小奇异值
            valid_mask = S > 1e-10
            U = U[:, valid_mask]
            S = S[valid_mask]
            VT = VT[valid_mask, :]
            
            svd_list.append((U, S, VT))
            max_rank = max(max_rank, S.size(0))
        
        # 零填充所有SVD到最大秩
        padded_svd_list = []
        for U, S, VT in svd_list:
            U_pad, S_pad, VT_pad = pad_svd_to_rank(U, S, VT, max_rank)
            padded_svd_list.append((U_pad, S_pad, VT_pad))
        
        # 加权平均U, S, VT
        U_weighted = torch.zeros_like(padded_svd_list[0][0])
        S_weighted = torch.zeros_like(padded_svd_list[0][1])
        VT_weighted = torch.zeros_like(padded_svd_list[0][2])
        
        for i, (U_pad, S_pad, VT_pad) in enumerate(padded_svd_list):
            w = weights[i].to(U_pad.device)
            U_weighted += w * U_pad
            S_weighted += w * S_pad
            VT_weighted += w * VT_pad
        
        fused_svd[layer_name] = (U_weighted, S_weighted, VT_weighted)
    
    return fused_svd


def reconstruct_from_svd_triplet(U: torch.Tensor, S: torch.Tensor, VT: torch.Tensor, 
                               target_rank: int, lora_alpha: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从SVD三元组重构AdaLoRA参数
    
    Args:
        U: 融合后的左奇异向量 (m, r_max)
        S: 融合后的奇异值 (r_max,)
        VT: 融合后的右奇异向量转置 (r_max, n)
        target_rank: 目标重构秩
        lora_alpha: LoRA alpha参数
        
    Returns:
        重构的LoRA参数 (lora_A, lora_B)
    """
    # 根据能量选择前target_rank个奇异值
    if target_rank < S.size(0):
        # 选择最大的target_rank个奇异值
        top_indices = torch.topk(S, target_rank).indices
        U_selected = U[:, top_indices]
        S_selected = S[top_indices]
        VT_selected = VT[top_indices, :]
    else:
        U_selected = U[:, :target_rank]
        S_selected = S[:target_rank]
        VT_selected = VT[:target_rank, :]
    
    # 重构为AdaLoRA格式: W = B @ A
    # 我们将奇异值分配给A和B
    sqrt_S = torch.sqrt(S_selected + 1e-10)  # 避免数值问题
    
    # lora_A: (r, n), lora_B: (m, r)
    lora_A = torch.diag(sqrt_S) @ VT_selected  # (r, n)
    lora_B = U_selected @ torch.diag(sqrt_S)   # (m, r)
    
    # 应用LoRA scaling
    scaling = lora_alpha / target_rank
    lora_A = lora_A * scaling
    
    return lora_A, lora_B


def adalora_zero_padding_aggregation(state_dicts: List[Dict[str, torch.Tensor]], 
                                   weights: List[float]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    AdaLoRA零补全聚合的主函数
    
    Args:
        state_dicts: 各客户端的AdaLoRA状态字典
        weights: 聚合权重
        
    Returns:
        融合后的SVD三元组字典
    """
    # 步骤1: 从每个客户端提取并重构完整矩阵
    matrices_list = []
    for state_dict in state_dicts:
        matrices = extract_adalora_matrices(state_dict)
        matrices_list.append(matrices)
    
    # 步骤2: 零填充SVD融合
    fused_svd = zero_padding_svd_fusion(matrices_list, weights)
    
    return fused_svd


def reconstruct_adalora_state_dict(svd_triplets: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                                 original_state_dict: Dict[str, torch.Tensor],
                                 target_ranks: Dict[str, int] = None,
                                 lora_alpha: int = 16) -> Dict[str, torch.Tensor]:
    """
    从SVD三元组重构完整的AdaLoRA状态字典
    
    Args:
        svd_triplets: 融合后的SVD三元组
        original_state_dict: 原始状态字典（用于获取非AdaLoRA参数）
        target_ranks: 各层的目标秩，如果为None则使用原始秩
        lora_alpha: LoRA alpha参数
        
    Returns:
        重构后的完整状态字典
    """
    new_state_dict = {}
    
    # 复制非AdaLoRA参数
    for key, value in original_state_dict.items():
        if not any(keyword in key for keyword in ['lora_A', 'lora_B', 'ranknum']):
            new_state_dict[key] = value.clone()
    
    # 重构AdaLoRA参数
    for layer_name, (U, S, VT) in svd_triplets.items():
        # 确定目标秩
        if target_ranks and layer_name in target_ranks:
            target_rank = target_ranks[layer_name]
        else:
            # 使用原始秩
            original_A_key = f"{layer_name}.lora_A"
            if original_A_key in original_state_dict:
                target_rank = original_state_dict[original_A_key].size(0)
            else:
                target_rank = min(8, S.size(0))  # 默认秩
        
        # 重构LoRA参数
        lora_A, lora_B = reconstruct_from_svd_triplet(U, S, VT, target_rank, lora_alpha)
        
        # 添加到状态字典
        new_state_dict[f"{layer_name}.lora_A"] = lora_A
        new_state_dict[f"{layer_name}.lora_B"] = lora_B
        
        # 更新ranknum（如果存在）
        ranknum_key = f"{layer_name}.ranknum"
        if ranknum_key in original_state_dict:
            new_state_dict[ranknum_key] = torch.tensor(float(target_rank))
    
    return new_state_dict