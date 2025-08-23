"""
设备检测工具模块
"""
import torch


def get_device() -> torch.device:
    """
    自动检测并返回最佳可用设备

    Returns:
        torch.device: 如果CUDA可用返回cuda，否则返回cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA detected, using GPU device")
        print(f"[Device] GPU name: {torch.cuda.get_device_name()}")
        print(f"[Device] GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        return device
    else:
        print(f"[Device] CUDA not available, using CPU device")
        return torch.device("cpu")


def get_device_str() -> str:
    """
    返回设备字符串表示

    Returns:
        str: "cuda" 或 "cpu"
    """
    return get_device().type
