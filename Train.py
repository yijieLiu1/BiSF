#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速无加密训练脚本：
- 支持选择模型（mnist_cnn / lenet）和数据集（mnist / cifar10）
- 可指定某个 DO 在整个训练过程中持续投毒（与 Test.py 的思路一致）
- 仅做明文训练与简单平均聚合，结果保存到 trainResult/do_train_params.json
"""

import argparse
import json
import os
from pathlib import Path
import random
from typing import List, Optional

from TA.TA import TA
from DO.DO import DO  # 复用模型与数据集工厂


def infer_model_size(model_name: str, dataset_name: str) -> int:
    """通过实例化模型估算参数量，用于设置 model_size"""
    meta = DO._get_dataset_meta(dataset_name)
    in_channels = meta["in_channels"]
    input_size = meta["input_size"]
    num_classes = meta["num_classes"]
    name = model_name.lower()
    if name in ("mnist_cnn", "cnn", "simple_cnn"):
        model = DO._SimpleMNISTCNN(in_channels=in_channels, input_size=input_size, num_classes=num_classes)
    elif name in ("lenet", "lenet5", "lenet_cifar"):
        model = DO._LeNet(in_channels=in_channels, input_size=input_size, num_classes=num_classes)
    elif name in ("resnet18", "resnet18_cifar", "resnet_cifar"):
        model = DO._ResNet18(in_channels=in_channels, num_classes=num_classes)
    elif name in ("resnet20", "resnet20_cifar"):
        model = DO._ResNet20(in_channels=in_channels, num_classes=num_classes)
    elif name in ("resnet_small", "resnet_tiny", "resnet18_small"):
        model = DO._ResNetSmall(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    return sum(p.numel() for p in model.parameters())


def load_initial_params(path: Optional[str], model_size: int) -> List[float]:
    """可选地加载初始参数，按 model_size 截断/填充"""
    if not path:
        return [0.0] * model_size
    try:
        if not Path(path).exists():
            print(f"[Train] 初始参数文件不存在，使用零向量: {path}")
            return [0.0] * model_size
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        params = data.get("params") if isinstance(data, dict) else None
        if params is None and isinstance(data, list):
            params = data
        if params is None:
            print(f"[Train] 参数文件缺少 params 字段，使用零向量: {path}")
            return [0.0] * model_size
        if len(params) < model_size:
            params = list(params) + [0.0] * (model_size - len(params))
        elif len(params) > model_size:
            params = params[:model_size]
        print(f"[Train] 已加载初始参数: {path}，维度 {len(params)}")
        return list(map(float, params))
    except Exception as e:
        print(f"[Train] 加载初始参数失败，使用零向量: {e}")
        return [0.0] * model_size


def apply_poison(vec: List[float], attack_type: str, attack_lambda: float, attack_sigma: float, rng: random.Random) -> List[float]:
    """简单投毒：stealth、random、signflip、lie_stat（退化为均值偏移）"""
    label = attack_type.lower()
    if label == "stealth":
        return [(1.0 + attack_lambda) * x for x in vec]
    if label == "random":
        return [rng.gauss(0.0, attack_sigma) for _ in vec]
    if label == "signflip":
        return [-(1.0 + attack_lambda) * x for x in vec]
    if label == "lie_stat":
        mean = sum(vec) / max(1, len(vec))
        return [mean + attack_lambda for _ in vec]
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description="DO 明文训练（支持投毒），输出全局参数")
    parser.add_argument("--rounds", type=int, default=30, help="训练轮数")
    parser.add_argument("--num-do", type=int, default=4, help="DO 数量")
    parser.add_argument("--model-name", type=str, default="resnet20", help="模型名称（mnist_cnn/lenet/resnet18/resnet20")
    parser.add_argument("--dataset-name", type=str, default="cifar10", help="数据集名称（mnist/cifar10）")
    parser.add_argument("--batch-size", type=int, default=128, help="训练批大小")
    parser.add_argument("--max-batches", type=int, default=400, help="每轮使用的批次数上限")
    parser.add_argument("--poison-do-id", type=int, default=None, help="指定持续投毒的 DO id（默认不投毒）")
    parser.add_argument("--attack-type", type=str, default=None, help="投毒类型（stealth/random/signflip/lie_stat）")
    parser.add_argument("--attack-lambda", type=float, default=0.2, help="投毒放大系数")
    parser.add_argument("--attack-sigma", type=float, default=1.0, help="随机投毒的标准差")
    parser.add_argument("--save-path", type=str, default=os.path.join("trainResult", "do_resnet20_train_params.json"), help="保存参数路径")
    parser.add_argument("--initial-params-path", type=str, default="trainResult/do_resnet20_train_params.json", help="trainResult/do_train_params.json")
    args = parser.parse_args()

    model_size = infer_model_size(args.model_name, args.dataset_name)
    print(f"[Train] 模型参数量估计: {model_size}")
    # 构建 TA（仅为 DO 初始化提供必要接口，不使用加密流程）
    # 注意：orthogonal_vector_count 不能为 0，否则 TA 内部生成正交向量会出错，这里给一个最小值 1
    ta = TA(num_do=args.num_do, model_size=model_size, orthogonal_vector_count=1, bit_length=512)

    # 构建 DO 列表（明文训练）
    do_list = [
        DO(
            i,
            ta,
            model_size=model_size,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        for i in range(args.num_do)
    ]

    # 初始全局参数
    global_params = load_initial_params(args.initial_params_path, model_size)
    rng = random.Random(2025)

    for r in range(1, args.rounds + 1):
        print(f"\n----- Round {r}/{args.rounds} -----")
        updates = {}
        for do in do_list:
            local = do._local_train(global_params)
            if args.poison_do_id is not None and do.id == args.poison_do_id:
                poisoned = apply_poison(local, args.attack_type, args.attack_lambda, args.attack_sigma, rng)
                updates[do.id] = poisoned
                print(f"[Train] DO {do.id} 执行投毒: {args.attack_type}")
            else:
                updates[do.id] = local
        # 简单平均
        agg = [0.0] * model_size
        active = len(updates)
        for vec in updates.values():
            for i, v in enumerate(vec):
                agg[i] += v
        if active > 0:
            agg = [v / active for v in agg]
        global_params = agg
        print(f"[Train] 全局参数前5项: {global_params[:5]}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_path).write_text(
        json.dumps(
            {
                "rounds": args.rounds,
                "model_size": len(global_params),
                "params": global_params,
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "poison_do_id": args.poison_do_id,
                "attack_type": args.attack_type,
                "attack_lambda": args.attack_lambda,
                "attack_sigma": args.attack_sigma,
            }
        ),
        encoding="utf-8",
    )
    print(f"[Train] 训练完成，参数已保存: {args.save_path}")


if __name__ == "__main__":
    main()
