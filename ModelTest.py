#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载 trainResult 中保存的全局参数，对 MNIST 测试集做简单推理评估。
"""
import os
import json
import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from DO.DO import DO  # 复用模型定义


def _assign_flat_to_model(model: nn.Module, flat, pad_or_cut: bool = True) -> None:
    """将一维参数向量写回模型，长度不足则补零，过长则截断。"""
    with torch.no_grad():
        offset = 0
        total_params = sum(p.numel() for p in model.parameters())
        if pad_or_cut and len(flat) < total_params:
            flat = list(flat) + [0.0] * (total_params - len(flat))
        if pad_or_cut and len(flat) > total_params:
            flat = flat[:total_params]
        for param in model.parameters():
            numel = param.numel()
            if offset >= len(flat):
                break
            take = min(numel, len(flat) - offset)
            new_vals = torch.tensor(flat[offset:offset + take], dtype=param.dtype, device=param.device)
            param.view(-1)[:take].copy_(new_vals)
            offset += take


def load_params(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("params") or data
    return params


def build_model(params):
    model = DO._SimpleMNISTCNN()
    _assign_flat_to_model(model, params, pad_or_cut=True)
    return model


def get_test_loader(batch_size: int = 256, max_batches: int = 10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "data"), train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return loader, max_batches


def evaluate(model, loader, max_batches: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batches = 0

    with torch.no_grad():
        for data, target in loader:
            if batches >= max_batches:
                break
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            loss = loss_fn(logits, target)
            preds = logits.argmax(dim=1)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == target).sum().item()
            total_samples += batch_size
            batches += 1

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc, batches


def main():
    parser = argparse.ArgumentParser(description="使用保存的全局参数对 MNIST 测试集做推理评估")
    parser.add_argument("--params", type=str, default=os.path.join("trainResult", "global_params_round10.json"),
                        help="保存的全局参数文件路径")
    parser.add_argument("--batch-size", type=int, default=256, help="评估批大小")
    parser.add_argument("--max-batches", type=int, default=10, help="评估使用的批次数上限（减轻耗时）")
    args = parser.parse_args()

    if not os.path.exists(args.params):
        raise FileNotFoundError(f"参数文件不存在: {args.params}")

    print(f"加载参数文件: {args.params}")
    params = load_params(args.params)
    print(f"参数维度: {len(params)}，前5项: {params[:5]}")

    model = build_model(params)
    loader, max_batches = get_test_loader(batch_size=args.batch_size, max_batches=args.max_batches)
    avg_loss, acc, used_batches = evaluate(model, loader, max_batches)

    print("\n===== 推理评估结果 =====")
    print(f"使用批次数: {used_batches}")
    print(f"平均交叉熵损失: {avg_loss:.6f}")
    print(f"准确率: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
