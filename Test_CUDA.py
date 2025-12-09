#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CUDA 安装与可用性：
- 打印 torch 版本、CUDA 版本、is_available
- 打印当前设备、GPU 名称和数量
- 进行一次简单的 Tensor 计算并报告耗时（如有 CUDA）
"""

import time
import torch
from torchvision import datasets, transforms

def main() -> None:
    print("datasets 版本:", datasets)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"当前设备: {torch.cuda.current_device()} -> {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("未检测到可用 GPU，后续计算将在 CPU 上进行。")

    # 简单计算测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(5000, 5000, device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()
    y = x @ x
    torch.cuda.synchronize() if device.type == "cuda" else None
    t2 = time.time()
    print(f"矩阵乘耗时: {t2 - t1:.4f} 秒，设备: {device}")


if __name__ == "__main__":
    main()
