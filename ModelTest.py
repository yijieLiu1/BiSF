#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载 trainResult 中保存的全局参数，对拆分出的验证集（训练集的后 20% 索引）做推理评估。
仅支持 4 个模型：cnn / lenet / resnet18 / resnet20
仅支持 3 个数据集：mnist / cifar10 / cifar100
拆分索引由 split_datasets.py 生成，存放在 data/splits/<dataset>_train_idx.json 与 *_val_idx.json
"""
import os
import json
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from DO.DO import DO  # 复用模型定义

# 支持的模型/数据集枚举
MODEL_BUILDERS = {
    "cnn": lambda ic, inp, nc: DO._SimpleMNISTCNN(in_channels=ic, input_size=inp, num_classes=nc),
    "lenet": lambda ic, inp, nc: DO._LeNet(in_channels=ic, input_size=inp, num_classes=nc),
    "resnet18": lambda ic, _inp, nc: DO._ResNet18(in_channels=ic, num_classes=nc),
    "resnet20": lambda ic, _inp, nc: DO._ResNet20(in_channels=ic, num_classes=nc),
}
DATASET_CHOICES = {"mnist", "cifar10", "cifar100"}


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
    meta = {}
    if isinstance(data, dict):
        for k in ("model_name", "dataset_name"):
            if k in data:
                meta[k] = data[k]
    return params, meta


def build_model(params, model_name: str, dataset_name: str):
    name = model_name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"仅支持模型 {sorted(MODEL_BUILDERS.keys())}，收到: {model_name}")
    if dataset_name.lower() not in DATASET_CHOICES:
        raise ValueError(f"仅支持数据集 {sorted(DATASET_CHOICES)}，收到: {dataset_name}")
    meta = DO._get_dataset_meta(dataset_name)
    in_channels = meta["in_channels"]
    input_size = meta["input_size"]
    num_classes = meta["num_classes"]
    model = MODEL_BUILDERS[name](in_channels, input_size, num_classes)
    _assign_flat_to_model(model, params, pad_or_cut=True)
    return model


def _needs_bn_calib(model_name: str) -> bool:
    """是否需要对 BN 统计量做校准（仅针对 ResNet 系列）"""
    name = model_name.lower()
    return name.startswith("resnet")


def recalibrate_bn(model: nn.Module, dataset_name: str, data_root: str, max_batches: int = 20) -> None:
    """
    使用训练集的若干 batch 来更新 BN 的 running_mean/var（不反传）。
    仅在需要时调用（ResNet），避免 BN 统计与聚合权重不匹配。
    """
    meta = DO._get_dataset_meta(dataset_name)
    ds_cls = meta["dataset_cls"]
    transform = meta["transform"]
    root = data_root or os.path.join(os.path.dirname(__file__), "data")
    dataset = ds_cls(root=root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=False)

    model.train()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= max_batches:
                break
            data = data.to(device)
            _ = model(data)
    model.eval()


def _get_test_transform(dataset_name: str):
    name = dataset_name.lower()
    if name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    if name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    if name == "cifar100":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    raise ValueError(f"仅支持数据集 {sorted(DATASET_CHOICES)}，收到: {dataset_name}")


def get_eval_loader(dataset_name: str, split: str = "val", batch_size: int = 256, max_batches: int = 10, data_root: str = None):
    transform = _get_test_transform(dataset_name)
    root = data_root or os.path.join(os.path.dirname(__file__), "data")
    ds_cls = DO._get_dataset_meta(dataset_name)["dataset_cls"]
    dataset = ds_cls(root=root, train=True, download=True, transform=transform)
    split_dir = os.path.join(root, "splits")
    idx_path = os.path.join(split_dir, f"{dataset_name.lower()}_{split}_idx.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"评估索引未找到: {idx_path}，请先运行 split_datasets.py 生成 80/20 拆分索引")
    with open(idx_path, "r", encoding="utf-8") as f:
        print(f"[ModelTest] 加载{split}集索引: ", idx_path)
        idx = json.load(f)
    subset = Subset(dataset, idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return loader, max_batches


def get_test_loader(dataset_name: str, batch_size: int = 256, max_batches: int = 10, data_root: str = None):
    """兼容旧接口，默认读取 val 拆分"""
    return get_eval_loader(dataset_name, split="val", batch_size=batch_size, max_batches=max_batches, data_root=data_root)


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


def evaluate_params(params, model_name: str, dataset_name: str, batch_size: int = 256, max_batches: int = 100, data_root: str = None, split: str = "val", bn_calib_batches: int = 0):
    """供 Train 复用：直接对内存参数向量做评估；必要时先做 BN 校准"""
    model = build_model(params, model_name=model_name, dataset_name=dataset_name)
    if _needs_bn_calib(model_name) and bn_calib_batches > 0:
        recalibrate_bn(model, dataset_name, data_root or os.path.join(os.path.dirname(__file__), "data"), max_batches=bn_calib_batches)
    loader, max_batches = get_eval_loader(dataset_name, split=split, batch_size=batch_size, max_batches=max_batches, data_root=data_root)
    return evaluate(model, loader, max_batches)


def main():
    parser = argparse.ArgumentParser(
        description="使用保存的全局参数对验证集做推理评估（cnn/lenet/resnet18/resnet20 + mnist/cifar10/cifar100，使用训练集拆分的 20% 验证索引）"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=os.path.join("trainResult", "cnn_mnist_global_params_round10.json"),
        help="保存的全局参数文件路径",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn",
        help="模型名称：cnn/lenet/resnet18/resnet20，留空则用文件元数据或默认 resnet18",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mnist",
        help="数据集名称：mnist/cifar10/cifar100，留空则用文件元数据或默认 cifar10",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default=None,
        help="验证集根目录，默认使用项目根目录下的 data/（读取 splits 中的 20% 索引）",
    )
    parser.add_argument(
        "--bn-calib-batches",
        type=int,
        default=0,
        help="BN 校准用的训练集 batch 数（仅 resnet 有效，0 表示不校准）",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="评估批大小")
    parser.add_argument("--max-batches", type=int, default=100, help="评估使用的批次数上限（减轻耗时）")
    args = parser.parse_args()

    if not os.path.exists(args.params):
        raise FileNotFoundError(f"参数文件不存在: {args.params}")

    print(f"加载参数文件: {args.params}")
    params, meta = load_params(args.params)
    print(f"参数维度: {len(params)}，前5项: {params[:5]}")

    model_name = args.model_name or meta.get("model_name") or "resnet18"
    dataset_name = args.dataset_name or meta.get("dataset_name") or "cifar10"
    if model_name.lower() not in MODEL_BUILDERS:
        raise ValueError(f"仅支持模型 {sorted(MODEL_BUILDERS.keys())}，收到: {model_name}")
    if dataset_name.lower() not in DATASET_CHOICES:
        raise ValueError(f"仅支持数据集 {sorted(DATASET_CHOICES)}，收到: {dataset_name}")

    print(f"评估使用模型: {model_name}, 数据集: {dataset_name}")
    model = build_model(params, model_name=model_name, dataset_name=dataset_name)
    loader, max_batches = get_test_loader(
        dataset_name,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        data_root=args.test_data_root,
    )
    if _needs_bn_calib(model_name) and args.bn_calib_batches > 0:
        recalibrate_bn(model, dataset_name, args.test_data_root or os.path.join(os.path.dirname(__file__), "data"), max_batches=args.bn_calib_batches)
    avg_loss, acc, used_batches = evaluate(model, loader, max_batches)

    print("\n===== 推理评估结果 =====")
    print(f"使用批次数: {used_batches}")
    print(f"平均交叉熵损失: {avg_loss:.6f}")
    print(f"准确率: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
