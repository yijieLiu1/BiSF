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
from typing import List, Optional, Dict
import time
import io
import contextlib
import math
import torch
import numpy as np

from TA.TA import TA
from DO.DO import DO  # 复用模型与数据集工厂
from CSP.CSP import CSP  # 复用投毒检测逻辑（Multi-Krum / GeoMedian / Cluster / LASA-lite）
import ModelTest  # 复用评估逻辑
from utils.detection_metrics import DetectionMetricsTracker
from utils.logging_utils import write_train_log


def infer_model_size(model_name: str, dataset_name: str) -> int:
    """通过实例化模型估算参数量，用于设置 model_size"""
    meta = DO._get_dataset_meta(dataset_name)
    in_channels = meta["in_channels"]
    input_size = meta["input_size"]
    num_classes = meta["num_classes"]
    name = model_name.lower()
    # 仅保留两类：标准 LeNet-5 与 ResNet20；"cnn" 等同于 LeNet-5
    if name in ("lenet", "lenet5", "lenet_cifar", "cnn", "mnist_cnn", "simple_cnn"):
        model = DO._LeNet(in_channels=in_channels, input_size=input_size, num_classes=num_classes)
    elif name in ("resnet20", "resnet20_cifar", "resnet"):
        model = DO._ResNet20(in_channels=in_channels, num_classes=num_classes)
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


def _build_resnet20_compression_spec(dataset_name: str) -> Dict[str, object]:
    """构造 ResNet20 的检测特征提取规格：FC + 全量 BN gamma/beta + Stage3 通道范数。"""
    meta = DO._get_dataset_meta(dataset_name)
    model = DO._ResNet20(in_channels=meta["in_channels"], num_classes=meta["num_classes"])
    bn_modules = {name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)}
    fc_ranges: List[tuple] = []
    bn_ranges: List[tuple] = []
    stage3_convs: List[Dict[str, object]] = []

    offset = 0
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        numel = param.numel()
        module_name = name.rsplit(".", 1)[0]
        if module_name == "fc":
            fc_ranges.append((offset, offset + numel))
        if module_name in bn_modules:
            bn_ranges.append((offset, offset + numel))
        if module_name.startswith("layer3") and "conv" in module_name and len(shape) == 4:
            stage3_convs.append({"start": offset, "shape": shape})
        offset += numel

    compressed_dim = sum(end - start for start, end in fc_ranges)
    compressed_dim += sum(end - start for start, end in bn_ranges)
    compressed_dim += sum(int(item["shape"][0]) for item in stage3_convs)

    return {
        "input_dim": offset,
        "compressed_dim": compressed_dim,
        "fc_ranges": fc_ranges,
        "bn_ranges": bn_ranges,
        "stage3_convs": stage3_convs,
    }


def _compress_vector(vec: List[float], spec: Dict[str, object]) -> List[float]:
    """提取 ResNet20 检测特征：FC + BN gamma/beta + Stage3 通道范数。"""
    if len(vec) != spec["input_dim"]:
        raise ValueError(f"压缩向量长度不匹配: {len(vec)} != {spec['input_dim']}")
    out: List[float] = []
    for start, end in spec["fc_ranges"]:
        out.extend(vec[start:end])
    for start, end in spec["bn_ranges"]:
        out.extend(vec[start:end])
    for item in spec["stage3_convs"]:
        start = int(item["start"])
        shape = item["shape"]
        out_channels = int(shape[0])
        per_channel = int(shape[1] * shape[2] * shape[3])
        for oc in range(out_channels):
            base = start + oc * per_channel
            s = 0.0
            for i in range(per_channel):
                v = vec[base + i]
                s += v * v
            out.append(math.sqrt(s))
    return out


def _generate_orthogonal_vectors(dim: int, count: int, seed: int = 2025) -> List[List[float]]:
    """生成 dim×count 的正交向量组（列正交），用于压缩向量的投影检测。"""
    if dim <= 0 or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(dim, count))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return [Q[:, i].astype(float).tolist() for i in range(Q.shape[1])]


def _project_vector_chunked(
    vec: List[float],
    orthogonal_vectors: List[List[float]],
    block_size: int,
) -> List[float]:
    """Stream orthogonal projection in blocks to reduce peak memory."""
    total = len(orthogonal_vectors)
    if total == 0:
        return []
    if block_size <= 0 or block_size >= total:
        block_size = total
    projection: List[float] = []
    for start in range(0, total, block_size):
        block = orthogonal_vectors[start:start + block_size]
        for u_vec in block:
            s = 0.0
            for x, y in zip(vec, u_vec):
                s += x * y
            projection.append(s)
    return projection


def main() -> None:
    #MNIST训练集按照batchsize=128时，大约切成468个batch，每个DO取400个batch进行训练，相当于差不多一次一个DO跑一个epoch
    parser = argparse.ArgumentParser(description="DO 明文训练（支持投毒），输出全局参数")
    parser.add_argument("--rounds", type=int, default=150, help="训练轮数")
    parser.add_argument("--num-do", type=int, default=10, help="DO 数量")
    parser.add_argument("--model-name", type=str, default="resnet20", help="模型名称（cnn/lenet/resnet20，对应 LeNet-5 或 ResNet20）")
    parser.add_argument("--dataset-name", type=str, default="cifar10", help="数据集名称（mnist/cifar10）")
    parser.add_argument("--batch-size", type=int, default=64, help="训练批大小")
    parser.add_argument("--max-batches", type=int, default=300, help="每轮使用的批次数上限")
    parser.add_argument("--partition-mode",type=str,default="iid",help="数据划分模式：iid / mild / extreme（控制各 DO 训练数据的 IID/non-IID 程度）",)
    # target label flip 投毒设置（与 Test.py 对齐）
    parser.add_argument("--poison-do-id",type=str,default="1,2,4,7",help="攻击 DO id（同时用于 label flip 和 untarget 投毒），逗号分隔如 0,1；留空则不投毒",)
    parser.add_argument("--source-label", type=int, default=1, help="若需 label flip，指定源标签")
    parser.add_argument("--target-label", type=int, default=99, help="若需 label flip，指定目标标签，超出数据label范围，则直接随机选择")
    parser.add_argument("--attack-rounds",type=str,default="all",help="label flip 的攻击轮次，逗号分隔如 3,4,5，或 all 表示每轮；留空默认不触发",)
    parser.add_argument("--poison-ratio",type=float,default=1.0,help="label flip 翻转比例，默认 0.3（源类样本内随机抽该比例改成目标标签）",)
    # 梯度投毒设置（stealth/random/signflip/lie_stat）
    parser.add_argument("--attack-type", type=str, default=None, help="untarget 投毒类型（stealth/random/signflip/lie_stat）")
    parser.add_argument("--attack-round-untarget", type=str,default=None,help="untarget 投毒触发轮次：单个数字、逗号列表或 all；留空则不触发",)
    parser.add_argument("--attack-lambda", type=float, default=0.25, help="投毒放大系数")
    parser.add_argument("--attack-sigma", type=float, default=1.0, help="随机投毒的标准差")
    # backdoor (BadNets-style) 配置
    parser.add_argument("--bd-enable",type=lambda x: str(x).lower() == "true",default=False,help="启用 backdoor (BadNets) 投毒，显式传 True 才会开启，如 --bd-enable True",)
    parser.add_argument("--bd-target-label", type=int, default=9, help="backdoor 目标标签")
    parser.add_argument("--bd-ratio", type=float, default=0.3, help="backdoor 在源标签样本中的注入比例")
    parser.add_argument("--bd-trigger-size", type=int, default=2, help="触发器方块尺寸（像素）")
    parser.add_argument("--bd-trigger-value", type=float, default=3.0, help="触发器像素值（归一化前）")
    #结果保存（Plain 明文训练结果，文件名中带 plain 以便区分加密版本）
    parser.add_argument("--save-path",type=str,default=None,help="保存目录（默认按 {model}_{dataset}_{numdo}do_{attack}_{rounds}r_plain 自动生成）",)
    parser.add_argument("--initial-params-path", type=str, default=None, help="trainResult/do_train_params.json")
    #模型推理测试
    parser.add_argument("--eval-batch-size", type=int, default=256, help="每轮推理评估批大小")
    parser.add_argument("--eval-batches", type=int, default=50, help="每轮评估使用的批次数上限（train/val）")
    parser.add_argument("--bn-calib-batches", type=int, default=30, help="BN 校准用的训练批次数（仅 resnet 有效）")
    parser.add_argument("--enable-compressed-detect", type=lambda x: str(x).lower() == "true", default=True, help="是否启用压缩向量投毒检测（仅 ResNet20 有效）")
    parser.add_argument("--proj-block-size", type=int, default=512, help="正交投影分块大小，默认 256，<=0 表示不分块")
    parser.add_argument("--refresh-orthogonal-each-round", type=lambda x: str(x).lower() == "true", default=False, help="是否每轮刷新 TA 正交向量组（True/False）")
    parser.add_argument("--enable-all-detection", type=lambda x: str(x).lower() == "true", default=True, help="是否开启全部投毒检测（True/False）")
    parser.add_argument("--detection-methods", type=str, default="all", help="投毒检测方案：all 或 multi,geo,cluster（逗号分隔）")
    args = parser.parse_args()
    enable_all_detection = bool(args.enable_all_detection)
    def _parse_detection_methods(raw: str) -> List[str]:
        if raw is None:
            return []
        text = str(raw).strip().lower()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts or "all" in parts:
            return ["multi_krum", "geomedian", "clustering"]
        mapping = {
            "multi": "multi_krum",
            "multi_krum": "multi_krum",
            "geo": "geomedian",
            "geomedian": "geomedian",
            "cluster": "clustering",
            "clustering": "clustering",
        }
        selected: List[str] = []
        for p in parts:
            method = mapping.get(p)
            if method and method not in selected:
                selected.append(method)
        if not selected:
            return ["multi_krum", "geomedian", "clustering"]
        return selected
    selected_methods = _parse_detection_methods(args.detection_methods)

    # 解析 target label flip 轮次
    attack_rounds = None
    if args.attack_rounds:
        if args.attack_rounds.lower() == "all":
            attack_rounds = "all"
        else:
            attack_rounds = [int(x) for x in args.attack_rounds.split(",") if x.strip()]

    # 解析攻击 DO 列表
    poison_do_ids: List[int] = []
    if args.poison_do_id:
        poison_do_ids = [int(x) for x in str(args.poison_do_id).split(",") if x.strip()]

    # 解析 untarget 投毒轮次
    attack_round_parsed: Optional[int] = None
    attack_rounds_untarget: Optional[object] = None
    if args.attack_round_untarget:
        if args.attack_round_untarget.lower() == "all":
            print("untarget 梯度/参数投毒将在所有轮次触发")
            attack_rounds_untarget = "all"
        elif "," in args.attack_round_untarget:
            attack_rounds_untarget = [int(x) for x in args.attack_round_untarget.split(",") if x.strip()]
        else:
            try:
                attack_round_parsed = int(args.attack_round_untarget)
            except ValueError:
                attack_round_parsed = None

    model_size = infer_model_size(args.model_name, args.dataset_name)
    print(f"[Train] 模型参数量估计: {model_size}")
    compression_spec = None
    compressed_orthogonal_vectors: Optional[List[List[float]]] = None
    if args.enable_compressed_detect and args.model_name.lower() in ("resnet20", "resnet20_cifar", "resnet"):
        compression_spec = _build_resnet20_compression_spec(args.dataset_name)
        print(
            f"[Train] 启用压缩检测：原始维度 {compression_spec['input_dim']} -> "
            f"{compression_spec['compressed_dim']}"
        )
        if enable_all_detection:
            comp_dim = int(compression_spec["compressed_dim"])
            comp_k = min(1024, comp_dim)
            if comp_k > 0:
                compressed_orthogonal_vectors = _generate_orthogonal_vectors(comp_dim, comp_k)
    # 构建 TA：用于 DO 初始化与正交向量生成（与 Test.py 保持一致，使用 2048 维投影）
    ta = TA(num_do=args.num_do, model_size=model_size, orthogonal_vector_count=1024, bit_length=512)

    # 仅用于投毒检测的 CSP：复用 Multi-Krum / GeoMedian / Cluster / LASA-lite 逻辑
    csp_detector = CSP(ta, model_size=model_size, precision=10 ** 6, initial_params_path=None)

    # 构建 DO 列表（明文训练），同步 Test.py 的 label flip 配置
    do_list = []
    for i in range(args.num_do):
        attack_cfg: Dict[str, object] = {}
        if args.source_label is not None and args.target_label is not None and i in poison_do_ids:
            attack_cfg.update(
                {
                    "attack_type": "label_flip",
                    "attacker_do_id": i,
                    "attack_rounds": attack_rounds if attack_rounds else "all",
                    "source_label": args.source_label,
                    "target_label": args.target_label,
                    "poison_ratio": args.poison_ratio,
                }
            )
        if args.bd_enable and args.bd_target_label is not None and i in poison_do_ids:
            attack_cfg.update(
                {
                    "bd_enable": True,
                    "bd_target_label": args.bd_target_label,
                    "bd_ratio": args.bd_ratio,
                    "bd_trigger_size": args.bd_trigger_size,
                    "bd_trigger_value": args.bd_trigger_value,
                }
            )
        do_list.append(
            DO(
                i,
                ta,
                model_size=model_size,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                partition_mode=args.partition_mode,
                attack_config=attack_cfg if attack_cfg else None,
            )
        )

    # 初始全局参数
    global_params = load_initial_params(args.initial_params_path, model_size)
    rng = random.Random(2025)
    prev_global = None
    convergence_history = []
    train_metric_history = []
    val_metric_history = []
    round_times = []
    detection_logs_raw: List[Dict[str, str]] = []
    detection_logs_proj: List[Dict[str, str]] = []
    detection_logs_comp_raw: List[Dict[str, str]] = []
    detection_logs_comp_proj: List[Dict[str, str]] = []
    best_params = None
    best_val_acc = -1.0
    detection_methods = tuple(selected_methods) if selected_methods else ("multi_krum", "geomedian", "clustering")
    raw_metrics = DetectionMetricsTracker(args.rounds, detection_methods)
    proj_metrics = DetectionMetricsTracker(args.rounds, detection_methods)
    comp_raw_metrics = DetectionMetricsTracker(args.rounds, detection_methods)
    comp_proj_metrics = DetectionMetricsTracker(args.rounds, detection_methods)


    def _run_detection_suite(vector_map: Dict[int, List[float]], label: str, temporary: bool = True) -> Dict[str, object]:
        """
        在明文训练下执行一次投毒检测：
        - vector_map: {do_id: 向量}，通常为原始模型参数或已经计算好的投影向量
        - label: 日志标签（如 "原始向量" / "正交投影向量"）
        使用 CSP 的投毒检测接口，但不涉及 SafeMul 或一致性检验。
        """
        if not vector_map:
            return {"log": "", "suspects": {}}
        backup = None
        if temporary:
            backup = csp_detector.do_projection_map
            csp_detector.do_projection_map = {k: list(v) for k, v in vector_map.items()}
        buf = io.StringIO()
        suspects_multi: List[int] = []
        suspects_geo: List[int] = []
        suspects_cluster: List[int] = []
        with contextlib.redirect_stdout(buf):
            print(f"\n----- 投毒检测（{label}，Plain）-----")
            active_count = len(csp_detector.do_projection_map)
            if active_count >= 2:
                # 这里参数与 Test.py 中保持一致，后续可按需调整敏感度
                if "multi_krum" in detection_methods:
                    suspects_multi = csp_detector.detect_poison_multi_krum(f=4, alpha=1.5)
                if "geomedian" in detection_methods:
                    suspects_geo = csp_detector.detect_poison_geomedian(beta=1.5)
                if "clustering" in detection_methods:
                    suspects_cluster = csp_detector.detect_poison_clustering(k=min(3, active_count), alpha=1.5)
                # csp_detector.detect_poison_lasa_lite(angle_threshold=0.0, beta=1.5)
            else:
                print("[检测] 在线 DO 数不足，跳过。")
        output = buf.getvalue()
        print(output, end="")
        if temporary:
            csp_detector.do_projection_map = backup
        return {
            "log": output,
            "suspects": {
                "multi_krum": suspects_multi,
                "geomedian": suspects_geo,
                "clustering": suspects_cluster,
            },
        }

    def _should_poison_round(round_idx: int, do_id: int) -> bool:
        """untarget 梯度/参数投毒的触发判定：支持 all / 列表 / 单次"""
        if not args.attack_type or do_id not in poison_do_ids:
            return False
        if attack_rounds_untarget == "all":
            return True
        if isinstance(attack_rounds_untarget, list):
            return round_idx in attack_rounds_untarget
        if attack_round_parsed is not None:
            return round_idx == attack_round_parsed
        return False

    def _should_label_flip_round(round_idx: int, do_id: int) -> bool:
        if args.source_label is None or args.target_label is None:
            return False
        if do_id not in poison_do_ids:
            return False
        if attack_rounds == "all":
            return True
        if isinstance(attack_rounds, list):
            return round_idx in attack_rounds
        return False

    def _should_backdoor_round(do_id: int) -> bool:
        if not args.bd_enable:
            return False
        if do_id not in poison_do_ids:
            return False
        if args.bd_target_label is None:
            return False
        if args.bd_ratio is None or float(args.bd_ratio) <= 0.0:
            return False
        return True

    def _get_poisoned_ids(round_idx: int, active_ids: List[int]) -> set:
        poisoned = set()
        for do_id in active_ids:
            if _should_poison_round(round_idx, do_id):
                poisoned.add(do_id)
            if _should_label_flip_round(round_idx, do_id):
                poisoned.add(do_id)
            if _should_backdoor_round(do_id):
                poisoned.add(do_id)
        return poisoned

    total_start = time.time()
    cached_orthogonal_vectors = None
    for r in range(1, args.rounds + 1):
        round_start = time.time()
        print(f"\n----- Round {r}/{args.rounds} -----")
        # 与 Test.py 保持一致：从第 2 轮开始，每轮刷新 TA 的密钥与正交向量组
        if args.refresh_orthogonal_each_round and r > 1:
            try:
                ta.update_keys_for_new_round()
            except Exception as e:
                print(f"[Train] 第 {r} 轮刷新 TA 正交向量失败: {e}")
        updates = {}
        for do in do_list:
            local = do._local_train(global_params)
            # untarget 梯度/参数投毒（stealth/random/signflip/lie_stat），与 Test.py 逻辑保持一致
            if _should_poison_round(r, do.id):
                poisoned = apply_poison(local, args.attack_type, args.attack_lambda, args.attack_sigma, rng)
                updates[do.id] = poisoned
                print(f"[Train] DO {do.id} 执行 untarget 投毒: {args.attack_type}")
            else:
                updates[do.id] = local

        active_ids = list(updates.keys())
        poisoned_ids = _get_poisoned_ids(r, active_ids)

        # Poison detection (raw/compressed/projection, Plain)
        compressed_updates = None
        if enable_all_detection:
            det_raw = _run_detection_suite(updates, "raw_vectors")
            detection_logs_raw.append({"round": r, "log": det_raw["log"]})
            raw_suspects = det_raw.get("suspects", {})
            for method in detection_methods:
                raw_metrics.update(
                    method, r, raw_suspects.get(method), poisoned_ids, active_ids
                )
            if compression_spec is not None:
                compressed_updates = {do_id: _compress_vector(vec, compression_spec) for do_id, vec in updates.items()}
                det_comp_raw = _run_detection_suite(compressed_updates, "compressed_raw_vectors")
                detection_logs_comp_raw.append({"round": r, "log": det_comp_raw["log"]})
                comp_raw_suspects = det_comp_raw.get("suspects", {})
                for method in detection_methods:
                    comp_raw_metrics.update(
                        method, r, comp_raw_suspects.get(method), poisoned_ids, active_ids
                    )
                if compressed_orthogonal_vectors:
                    comp_proj_map: Dict[int, List[float]] = {}
                    for do_id, vec in compressed_updates.items():
                        proj: List[float] = []
                        for u_vec in compressed_orthogonal_vectors:
                            s = 0.0
                            for x, y in zip(vec, u_vec):
                                s += x * y
                            proj.append(s)
                        comp_proj_map[do_id] = proj
                    det_comp_proj = _run_detection_suite(comp_proj_map, "compressed_projection")
                    detection_logs_comp_proj.append({"round": r, "log": det_comp_proj["log"]})
                    comp_proj_suspects = det_comp_proj.get("suspects", {})
                    comp_active_ids = list(comp_proj_map.keys())
                    for method in detection_methods:
                        comp_proj_metrics.update(
                            method, r, comp_proj_suspects.get(method), poisoned_ids, comp_active_ids
                        )

        if enable_all_detection:
            # 使用 TA 生成的完整正交向量组 U（形状约为 2048×model_size），直接做明文点积 w·U 得到 2048 维投影
            print(f"\n===== Round {r}: 正交投影计算（Plain，在线 DO）=====")
            t1 = time.time()
            proj_map: Dict[int, List[float]] = {}
            if args.refresh_orthogonal_each_round and r > 1:
                cached_orthogonal_vectors = None
            if cached_orthogonal_vectors is None:
                cached_orthogonal_vectors = ta.get_orthogonal_vectors()
            U = cached_orthogonal_vectors  # List[List[float]]，形状: [orthogonal_count][model_size]
            if  U:
                for do_id, vec in updates.items():
                    # 1×d 向量与 k×d 正交向量组逐个点积，按分块流式处理
                    proj = _project_vector_chunked(vec, U, args.proj_block_size)
                    proj_map[do_id] = proj
                    print(f" DO {do_id} 投影向量(长度{len(proj)})")
                # 更新 detector 的投影映射并执行检测
                csp_detector.do_projection_map = proj_map
                det_proj = _run_detection_suite(proj_map, "orthogonal_projection", temporary=False)
                detection_logs_proj.append({"round": r, "log": det_proj["log"]})
                proj_active_ids = list(proj_map.keys())
                proj_suspects = det_proj.get("suspects", {})
                for method in detection_methods:
                    proj_metrics.update(
                        method, r, proj_suspects.get(method), poisoned_ids, proj_active_ids
                    )
            t2 = time.time()
            print(f"[Train][Round {r}] 正交投影耗时 {t2 - t1:.4f}s")

        # 简单平均
        agg = [0.0] * model_size
        active = len(updates)
        for vec in updates.values():
            for i, v in enumerate(vec):
                agg[i] += v
        if active > 0:
            agg = [v / active for v in agg]
        global_params = agg

        # 收敛指标：与上一轮全局参数的差异
        if prev_global is None:
            print("[Train] 首轮，无收敛对比指标")
        else:
            sq = 0.0
            m_abs = 0.0
            for x, y in zip(prev_global, global_params):
                d = y - x
                sq += d * d
                ad = abs(d)
                if ad > m_abs:
                    m_abs = ad
            l2 = sq ** 0.5
            convergence_history.append({"round": r, "l2": l2, "linf": m_abs})
            print(f"[Train] 收敛指标：L2差={l2:.6f}, L∞差={m_abs:.6f}")
        prev_global = list(global_params)

        print(f"[Train] 全局参数前5项: {global_params[:5]}")
        # 训练/验证评估
        try:
            train_loss, train_acc, tb, train_asr, train_src_acc, train_bd_asr = ModelTest.evaluate_params(
                global_params,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                batch_size=args.eval_batch_size,
                max_batches=args.eval_batches,
                data_root=os.path.join(os.path.dirname(__file__), "data"),
                split="train",
                bn_calib_batches=args.bn_calib_batches,
                source_label=args.source_label,
                target_label=args.target_label,
                bd_target_label=args.bd_target_label if args.bd_enable else None,
                bd_trigger_size=args.bd_trigger_size,
                bd_trigger_value=args.bd_trigger_value,
                bd_inject_ratio=args.bd_ratio if args.bd_enable else 0.0,
            )
            val_loss, val_acc, vb, val_asr, val_src_acc, val_bd_asr = ModelTest.evaluate_params(
                global_params,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                batch_size=args.eval_batch_size,
                max_batches=args.eval_batches,
                data_root=os.path.join(os.path.dirname(__file__), "data"),
                split="val",
                bn_calib_batches=args.bn_calib_batches,
                source_label=args.source_label,
                target_label=args.target_label,
                bd_target_label=args.bd_target_label if args.bd_enable else None,
                bd_trigger_size=args.bd_trigger_size,
                bd_trigger_value=args.bd_trigger_value,
                bd_inject_ratio=args.bd_ratio if args.bd_enable else 0.0,
            )
            train_metric_history.append({"round": r, "loss": train_loss, "acc": train_acc, "asr": train_asr, "src_acc": train_src_acc, "bd_asr": train_bd_asr})
            val_metric_history.append({"round": r, "loss": val_loss, "acc": val_acc, "asr": val_asr, "src_acc": val_src_acc, "bd_asr": val_bd_asr})
            extra_train = ""
            extra_val = ""
            if train_asr is not None and train_src_acc is not None:
                extra_train = f", ASR(s->{args.target_label})={train_asr*100:.2f}%, src_acc={train_src_acc*100:.2f}%"
            if train_bd_asr is not None:
                extra_train += f", bd_ASR={train_bd_asr*100:.2f}%"
            if val_asr is not None and val_src_acc is not None:
                extra_val = f", ASR(s->{args.target_label})={val_asr*100:.2f}%, src_acc={val_src_acc*100:.2f}%"
            if val_bd_asr is not None:
                extra_val += f", bd_ASR={val_bd_asr*100:.2f}%"
            print(f"[Eval][Round {r}] Train loss={train_loss:.4f}, acc={train_acc*100:.2f}% (batches {tb}){extra_train}")
            print(f"[Eval][Round {r}] Val   loss={val_loss:.4f}, acc={val_acc*100:.2f}% (batches {vb}){extra_val}")
            # 记录最优（按 val acc，否则按 train acc）
            target_acc = val_acc if val_metric_history else train_acc
            if target_acc > best_val_acc:
                best_val_acc = target_acc
                best_params = list(global_params)
        except Exception as e:
            print(f"[Eval][Round {r}] 评估失败: {e}")
        round_elapsed = time.time() - round_start
        round_times.append(round_elapsed)
        print(f"[Train] Round {r} 用时: {round_elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    model_tag = args.model_name.lower()
    dataset_tag = args.dataset_name.lower()
    if args.attack_type:
        attack_tag = args.attack_type.lower()
    elif args.bd_enable:
        attack_tag = "backdoor"
    elif poison_do_ids:
        attack_tag = "label-flip"
    else:
        attack_tag = "normal"
    exp_name = f"{model_tag}_{dataset_tag}_{args.num_do}do_{attack_tag}_{args.rounds}r_plain"
    output_dir = Path(args.save_path) if args.save_path else Path("trainResult") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "params.json"
    best_path = output_dir / "best-params.json"
    log_path = output_dir / "log.txt"

    params_path.write_text(
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
                "source_label": args.source_label,
                "target_label": args.target_label,
                "bd_enable": args.bd_enable,
                "bd_target_label": args.bd_target_label,
                "bd_ratio": args.bd_ratio,
                "bd_trigger_size": args.bd_trigger_size,
                "bd_trigger_value": args.bd_trigger_value,
            }
        ),
        encoding="utf-8",
    )
    print(f"[Train] 训练完成，参数已保存: {params_path}")
    # 额外保存最优参数（按 val acc，否则 train acc）
    if best_params is not None:
        best_path.write_text(
            json.dumps(
                {
                    "rounds": args.rounds,
                    "model_size": len(best_params),
                    "params": best_params,
                    "model_name": args.model_name,
                    "dataset_name": args.dataset_name,
                    "best_metric": best_val_acc,
                    "source_label": args.source_label,
                    "target_label": args.target_label,
                }
            ),
            encoding="utf-8",
        )
        print(f"[Train] 最优参数已保存: {best_path}")
    # 保存日志（收敛、评估、用时）
    summary_lines = []
    if enable_all_detection:
        summary_lines.extend(raw_metrics.format_summary("raw_vectors"))
        summary_lines.extend(proj_metrics.format_summary("orthogonal_projection"))
        if compression_spec is not None:
            summary_lines.extend(comp_raw_metrics.format_summary("compressed_raw_vectors"))
            summary_lines.extend(comp_proj_metrics.format_summary("compressed_projection"))
    write_train_log(
        log_path=str(log_path),
        total_elapsed=total_elapsed,
        round_times=round_times,
        convergence_history=convergence_history,
        train_metric_history=train_metric_history,
        val_metric_history=val_metric_history,
        detection_logs_raw=detection_logs_raw if enable_all_detection else [],
        detection_logs_proj=detection_logs_proj if enable_all_detection else [],
        detection_logs_comp_raw=detection_logs_comp_raw if (enable_all_detection and compression_spec is not None) else None,
        detection_logs_comp_proj=detection_logs_comp_proj if (enable_all_detection and compression_spec is not None) else None,
        detection_summary_lines=summary_lines if enable_all_detection else None,
    )
    print(f"[Train] 日志已保存: {log_path}")
    print(f"[Train] 总用时: {total_elapsed:.2f}s")
    if convergence_history:
        print("\n===== 收敛指标汇总（相邻轮差） =====")
        for item in convergence_history:
            print(f"Round {item['round']}: L2={item['l2']:.6f}, L∞={item['linf']:.6f}")
        if train_metric_history and val_metric_history:
            print("\n===== 训练/验证指标汇总（按轮） =====")
            for tm, vm in zip(train_metric_history, val_metric_history):
                extra_train = ""
                extra_val = ""
                if tm.get("asr") is not None and tm.get("src_acc") is not None:
                    extra_train = f", ASR={tm['asr']*100:.2f}%, src_acc={tm['src_acc']*100:.2f}%"
                if tm.get("bd_asr") is not None:
                    extra_train += f", bd_ASR={tm['bd_asr']*100:.2f}%"
                if vm.get("asr") is not None and vm.get("src_acc") is not None:
                    extra_val = f", ASR={vm['asr']*100:.2f}%, src_acc={vm['src_acc']*100:.2f}%"
                if vm.get("bd_asr") is not None:
                    extra_val += f", bd_ASR={vm['bd_asr']*100:.2f}%"
                print(f"Round {tm['round']}: Train loss={tm['loss']:.4f}, acc={tm['acc']*100:.2f}%{extra_train} | Val loss={vm['loss']:.4f}, acc={vm['acc']*100:.2f}%{extra_val}")
        if enable_all_detection:
            print("\n===== Poison detection summary (raw vectors) =====")
            for line in raw_metrics.format_summary("raw_vectors"):
                print(line)
            print("\n===== Poison detection summary (orthogonal projection) =====")
            for line in proj_metrics.format_summary("orthogonal_projection"):
                print(line)
            if compression_spec is not None:
                print("\n===== Poison detection summary (compressed raw vectors) =====")
                for line in comp_raw_metrics.format_summary("compressed_raw_vectors"):
                    print(line)
                print("\n===== Poison detection summary (compressed projection) =====")
                for line in comp_proj_metrics.format_summary("compressed_projection"):
                    print(line)


if __name__ == "__main__":
    main()
