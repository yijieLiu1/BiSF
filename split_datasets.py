#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载并拆分 MNIST / CIFAR10 / CIFAR100 训练集（保持官方二进制文件）：
- 仅生成索引文件：前 80% 训练索引、后 20% 验证索引
- 数据仍存放在 torchvision 的默认二进制结构下 data/
索引文件保存到 data/splits/<dataset>_train_idx.json 与 *_val_idx.json
"""
import os
import random
import json
from typing import Tuple, List

from torchvision import datasets


def split_and_save(
    name: str,
    dataset_cls,
    train: bool,
    ratio_holdout: float,
    root_train: str = "data",
    seed: int = 42,
) -> None:
    random.seed(seed)
    ds = dataset_cls(root=root_train, train=train, download=True, transform=None)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    cut = int(len(indices) * (1 - ratio_holdout))
    keep_idx = indices[:cut]
    holdout_idx = indices[cut:]

    # 保存索引
    split_dir = os.path.join(root_train, "splits")
    os.makedirs(split_dir, exist_ok=True)
    train_idx_path = os.path.join(split_dir, f"{name}_train_idx.json")
    val_idx_path = os.path.join(split_dir, f"{name}_val_idx.json")

    with open(train_idx_path, "w", encoding="utf-8") as f:
        json.dump(keep_idx, f)
    with open(val_idx_path, "w", encoding="utf-8") as f:
        json.dump(holdout_idx, f)

    print(f"[{name}] 索引生成完成：train {len(keep_idx)} -> {train_idx_path}, val {len(holdout_idx)} -> {val_idx_path}")


def main() -> None:
    ratio = 0.2
    datasets_to_process: Tuple[Tuple[str, object], ...] = (
        ("MNIST", datasets.MNIST),
        ("cifar10", datasets.CIFAR10),
        # ("cifar100", datasets.CIFAR100),
    )
    for name, cls in datasets_to_process:
        split_and_save(name=name, dataset_cls=cls, train=True, ratio_holdout=ratio)


if __name__ == "__main__":
    main()
