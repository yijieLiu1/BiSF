#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可配置的联邦学习模拟，支持：
- 指定掉线的 DO（按轮次）
- 在指定轮次对某个 DO 施加基于自身梯度的 Lie Attack（放大 1+lambda）
- 在每轮执行 CSP 的投毒检测（Multi-Krum 与 GeoMedian）
"""

import os
import sys
import time
import random
import math
import numpy as np
import torch
import json
import argparse
import io
import contextlib
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from TA.TA import TA
from CSP.CSP import CSP
from ASP.ASP import ASP
from DO.DO import DO
import ModelTest
from utils.detection_metrics import DetectionMetricsTracker
from utils.logging_utils import write_test_log

# 默认参数。作为“库函数”时可单独调用；命令行入口会显式传入同名参数（保持默认值一致）。
def _build_resnet20_compression_spec(dataset_name: str) -> Dict[str, object]:
    """Build ResNet20 compression spec: FC + BN gamma/beta + stage3 channel norms."""
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
    """Extract compressed vector features for ResNet20 detection."""
    if len(vec) != spec["input_dim"]:
        raise ValueError(f"Compressed vector length mismatch: {len(vec)} != {spec['input_dim']}")
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

def run_federated_simulation(
    num_rounds: int = 30,
    num_do: int = 10,
    model_size: int = 10_000,
    orthogonal_vector_count: int = 2_048,
    bit_length: int = 512,
    precision: int = 10 ** 6,
    dropouts: Optional[Dict[int, List[int]]] = None,
    dropout_round: Optional[int] = None,
    dropout_do_ids: Optional[List[int]] = None,
    # untarget 梯度/参数投毒的单轮触发配置
    attack_round: Optional[int] = None,
    attack_do_ids: Optional[List[int]] = None,
    # stealth 隐蔽投毒 / random 随机噪声 / signflip 反向梯度 / lie_stat 统计型放大
    # 注意：label flip 使用的是 DO 内部的 attack_config，与此处 attack_type 解耦。
    attack_type: Optional[str] = None,
    attack_lambda: float = 0.2,
    attack_sigma: float = 1.0,
    # 模型名称和数据集名称："lenet/mnist_cnn","cnn/mnist","lenet/cifar10" 等
    model_name: str = "cnn",
    dataset_name: str = "mnist",
    # DO 训练的 batch 选项
    train_batch_size: Optional[int] = 64,
    train_max_batches: Optional[int] = 300,
    # 数据划分模式：iid / mild / extreme（控制各 DO 的本地数据分布）
    partition_mode: str = "iid",
    # 可选的模型参数，可以指定使用已训练过的全局模型参数作为初始参数：
    initial_params_path: Optional[str] = None,
    # 评估相关：每轮在 train/val 上做推理（明文），可选 BN 校准（ResNet）
    # 是否每一轮都进行一次推理，查看准确率
    eval_each_round: bool = True,
    eval_batch_size: int = 256,
    eval_batches: int = 50,
    bn_calib_batches: int = 30,
    # label flip 攻击配置（target 攻击）
    source_label: Optional[int] = None,
    target_label: Optional[int] = None,
    attack_rounds: Optional[List[int]] = None,
    poison_ratio: float = 0.3,
    # untarget 梯度/参数投毒的轮次配置："all" / List[int] / 单个 int / None
    attack_rounds_untarget: Optional[object] = "all",
    bd_enable: bool = True,
    bd_target_label: Optional[int] = 9,
    bd_ratio: float = 0.3,
    bd_trigger_size: int = 2,
    bd_trigger_value: float = 3.0,
    enable_compressed_detect: bool = True,
    refresh_orthogonal_each_round: bool = False,
    safe_mul_block_size: int = 256,
    enable_all_detection: bool = True,
    detection_methods: Optional[List[str]] = None,
    audit_round: Optional[int] = None,
    audit_do_ids: Optional[List[int]] = None,
    audit_simulate_dropout: bool = False,
    audit_simulate_mismatch: bool = False,
) -> None:
    """
    运行一次联邦学习流程。

    - dropouts / dropout_round + dropout_do_ids：配置掉线 DO
    - label flip（目标攻击）：
        * 通过 DO 内部的 attack_config 控制：
          - attack_do_ids: 哪些 DO 做 label flip
          - source_label / target_label / attack_rounds / poison_ratio
    - untarget 梯度/参数投毒：
        * 本文件内部在聚合前直接篡改明文更新：
          - attack_type: stealth / random / signflip / lie_stat
          - attack_do_ids: 哪些 DO 做 untarget 投毒
          - attack_round / attack_rounds_untarget: 触发轮次
    - 审计模拟（单轮）：可指定 DO 在特定轮次
        * audit_simulate_dropout: 直接掉线（不参与加密与 SafeMul）
        * audit_simulate_mismatch: SafeMul 使用与加密不一致的本地参数

    注意：同一个 attack_do_ids 列表同时被 label flip 和 untarget 两类攻击复用，
    是否真正“生效”取决于是否同时满足各自所需的参数组合。
    """
    dropouts = dropouts or {}
    if dropout_round is not None and dropout_do_ids:
        dropouts = dict(dropouts)
        dropouts[dropout_round] = dropout_do_ids
    _rng = random.Random(2025)

    def _vector_gap(new_vec: List[float], old_vec: List[float]) -> Dict[str, float]:
        if not new_vec or not old_vec:
            return {"l2": 0.0, "linf": 0.0}
        l2_sq = 0.0
        linf = 0.0
        for x, y in zip(new_vec, old_vec):
            d = x - y
            l2_sq += d * d
            ad = abs(d)
            if ad > linf:
                linf = ad
        return {"l2": math.sqrt(l2_sq), "linf": linf}

    def _vector_cos(new_vec: List[float], old_vec: List[float]) -> float:
        if not new_vec or not old_vec:
            return 0.0
        num = 0.0
        n_norm_sq = 0.0
        o_norm_sq = 0.0
        for x, y in zip(new_vec, old_vec):
            num += x * y
            n_norm_sq += x * x
            o_norm_sq += y * y
        if n_norm_sq == 0.0 or o_norm_sq == 0.0:
            return 0.0
        return num / (math.sqrt(n_norm_sq) * math.sqrt(o_norm_sq))

    def _vector_summary(vec: List[float]) -> Dict[str, float]:
        if not vec:
            return {"mean": 0.0, "abs_max": 0.0}
        total = 0.0
        abs_max = 0.0
        for v in vec:
            total += v
            av = abs(v)
            if av > abs_max:
                abs_max = av
        return {"mean": total / len(vec), "abs_max": abs_max}

    print("===== 初始化 TA / CSP / DO 列表 =====")
    ta = TA(
        num_do=num_do,
        model_size=model_size,
        orthogonal_vector_count=orthogonal_vector_count,
        bit_length=bit_length,
        precision=precision,
    )
    audit_weight_scale = 1000
    asp = ASP.from_ta(ta, weight_scale=audit_weight_scale)
    csp = CSP(
        ta,
        model_size=model_size,
        precision=precision,
        initial_params_path=initial_params_path,
        asp=asp,
        audit_weight_scale=audit_weight_scale,
    )
    compression_spec = None
    compressed_orthogonal_vectors: Optional[List[List[float]]] = None
    if enable_compressed_detect and model_name.lower() in ("resnet20", "resnet20_cifar", "resnet"):
        compression_spec = _build_resnet20_compression_spec(dataset_name)
        print(
            f"[Test] 启用压缩检测：原始维度 {compression_spec['input_dim']} -> "
            f"{compression_spec['compressed_dim']}"
        )
        comp_dim = int(compression_spec["compressed_dim"])
        comp_k = min(int(orthogonal_vector_count), comp_dim)
        if comp_k > 0:
            compressed_orthogonal_vectors = _generate_orthogonal_vectors(comp_dim, comp_k)
    attack_do_ids = attack_do_ids or []
    audit_do_ids = audit_do_ids or []

    do_list: List[Optional[DO]] = []
    for i in range(num_do):
        attack_cfg: Dict[str, object] = {}
        if source_label is not None and target_label is not None and i in attack_do_ids:
            attack_cfg.update(
                {
                    "attack_type": "label_flip",
                    "attacker_do_id": i,
                    "attack_rounds": attack_rounds if attack_rounds else "all",
                    "source_label": source_label,
                    "target_label": target_label,
                    "poison_ratio": poison_ratio,
                }
            )
        if bd_enable and bd_target_label is not None and i in attack_do_ids:
            attack_cfg.update(
                {
                    "bd_enable": True,
                    "bd_target_label": bd_target_label,
                    "bd_ratio": bd_ratio,
                    "bd_trigger_size": bd_trigger_size,
                    "bd_trigger_value": bd_trigger_value,
                }
            )
        do_list.append(
            DO(
                i,
                ta,
                model_size=model_size,
                precision=precision,
                model_name=model_name,
                dataset_name=dataset_name,
                batch_size=train_batch_size,
                max_batches=train_max_batches,
                partition_mode=partition_mode,
                attack_config=attack_cfg if attack_cfg else None,
            )
        )

    round_stats: List[Dict[str, float]] = []

    def run_detection_suite(vector_map: Dict[int, List[float]], label: str, temporary: bool = True) -> Dict[str, object]:
        if not vector_map:
            return {"log": "", "suspects": {}}
        backup = None
        if temporary:
            backup = csp.do_projection_map
            csp.do_projection_map = {k: list(v) for k, v in vector_map.items()}
        buf = io.StringIO()
        suspects_multi: List[int] = []
        suspects_geo: List[int] = []
        suspects_cluster: List[int] = []
        with contextlib.redirect_stdout(buf):
            print(f"\n----- Detection ({label}) -----")
            active_count = len(csp.do_projection_map)
            if active_count >= 2:
                if "multi_krum" in detection_methods:
                    suspects_multi = csp.detect_poison_multi_krum(f=2, alpha=1.5)
                if "geomedian" in detection_methods:
                    suspects_geo = csp.detect_poison_geomedian(beta=1.5)
                if "clustering" in detection_methods:
                    suspects_cluster = csp.detect_poison_clustering(k=min(3, active_count), alpha=1.5)
            else:
                print("[Detect] Not enough active DOs, skip.")
        output = buf.getvalue()
        print(output, end="")
        if temporary:
            csp.do_projection_map = backup
        return {
            "log": output,
            "suspects": {
                "multi_krum": suspects_multi,
                "geomedian": suspects_geo,
                "clustering": suspects_cluster,
            },
        }

    round_times = []
    round_timing_details: List[Dict[str, object]] = []
    eval_history: List[Dict[str, float]] = []
    detection_logs_raw: List[Dict[str, str]] = []
    detection_logs_proj: List[Dict[str, str]] = []
    detection_logs_comp_raw: List[Dict[str, str]] = []
    detection_logs_comp_proj: List[Dict[str, str]] = []
    if detection_methods:
        detection_methods = tuple(detection_methods)
    else:
        detection_methods = ("multi_krum", "geomedian", "clustering")
    raw_metrics = DetectionMetricsTracker(num_rounds, detection_methods)
    proj_metrics = DetectionMetricsTracker(num_rounds, detection_methods)
    comp_raw_metrics = DetectionMetricsTracker(num_rounds, detection_methods)
    comp_proj_metrics = DetectionMetricsTracker(num_rounds, detection_methods)
    best_params = None
    best_metric = -1.0  # 优先用 val acc，如果没有则用 train acc
    # 构造统一的运行标签：模型_数据集_do数_场景_轮数
    def _build_run_tag() -> str:
        model_tag = model_name.lower().replace(" ", "_")
        dataset_tag = dataset_name.lower().replace(" ", "_")
        scenario_parts = []

        # target 攻击（label flip）：只要配置了攻击 DO 且给出了源/目标标签，就认为开启
        has_target_attack = bool(attack_do_ids) and source_label is not None and target_label is not None
        # untarget 攻击（stealth/random/signflip/lie_stat）：只要 attack_type 与攻击 DO 同时非空，就认为开启
        has_untarget_attack = bool(attack_do_ids) and bool(attack_type)

        has_backdoor = bool(attack_do_ids) and bd_enable

        if has_target_attack:
            if attack_rounds == "all":
                scenario_parts.append("target_labelflip_all")
            elif isinstance(attack_rounds, list) and attack_rounds:
                scenario_parts.append(f"target_labelflip_{len(attack_rounds)}r")
            else:
                scenario_parts.append("target_labelflip")

        if has_untarget_attack:
            scenario_parts.append(f"untarget_{attack_type.lower()}")

        if has_backdoor:
            scenario_parts.append("backdoor")

        scenario_tag = "_".join(scenario_parts) if scenario_parts else "normal"
        return f"{model_tag}_{dataset_tag}_{num_do}do_{scenario_tag}_{num_rounds}r"

    run_tag = _build_run_tag()

    total_start = time.time()

    def _should_poison_round(round_idx: int, do_id: int) -> bool:
        """untarget 梯度/参数投毒的触发判定：支持 all / 列表 / 单次"""
        if not attack_type or do_id not in attack_do_ids:
            return False
        if attack_rounds_untarget == "all":
            return True
        if isinstance(attack_rounds_untarget, list):
            return round_idx in attack_rounds_untarget
        if attack_round is not None:
            return round_idx == attack_round
        return False

    def _should_label_flip_round(round_idx: int, do_id: int) -> bool:
        if source_label is None or target_label is None:
            return False
        if do_id not in attack_do_ids:
            return False
        if attack_rounds == "all":
            return True
        if isinstance(attack_rounds, list):
            return round_idx in attack_rounds
        return False

    def _should_backdoor_round(do_id: int) -> bool:
        if not bd_enable:
            return False
        if do_id not in attack_do_ids:
            return False
        if bd_target_label is None:
            return False
        if bd_ratio is None or float(bd_ratio) <= 0.0:
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

    safe_mul_ctx_cache = None
    safe_mul_block_ctx_cache: Dict[tuple, Dict[str, object]] = {}
    for round_idx in range(1, num_rounds + 1):
        if refresh_orthogonal_each_round and round_idx > 1:
            try:
                ta.update_keys_for_new_round()
            except Exception as e:
                print(f"[Test] Round {round_idx} TA refresh failed: {e}")
        round_start = time.time()
        print(f"\n===== Round {round_idx}: 广播参数 =====")
        global_params = csp.broadcast_params(refresh_orthogonal_each_round)

        round_timing = {
            "round": round_idx,
            "round_total": 0.0,
            "do_encrypt_times": {},
            "do_safe_mul_round2_times": {},
            "csp_safe_mul_prepare": 0.0,
            "csp_safe_mul_finalize_times": {},
        }

        # 掉线处理（含审计模拟掉线）
        working_do_list: List[Optional[DO]] = list(do_list)
        for drop_id in dropouts.get(round_idx, []):
            if 0 <= drop_id < len(working_do_list):
                working_do_list[drop_id] = None
        if audit_round is not None and round_idx == audit_round and audit_simulate_dropout:
            for drop_id in audit_do_ids:
                if 0 <= drop_id < len(working_do_list):
                    working_do_list[drop_id] = None
        offline_ids = [i for i, d in enumerate(working_do_list) if d is None]
        if offline_ids:
            print(f"[Round {round_idx}] 模拟掉线 DO: {offline_ids}")

        # DO 训练并上传
        print(f"\n===== Round {round_idx}: DO 训练并加密 =====")
        do_cipher_map: Dict[int, List[int]] = {}
        clean_update_map: Dict[int, List[float]] = {}
        audit_mismatch_map: Dict[int, List[float]] = {}
        for do in [d for d in working_do_list if d is not None]:
            t_enc_start = time.time()
            ciphertexts = do.train_and_encrypt(global_params)
            t_enc_end = time.time()
            do_cipher_map[do.id] = ciphertexts
            clean_update_map[do.id] = do.get_last_updates()
            if audit_round is not None and round_idx == audit_round and audit_simulate_mismatch:
                if do.id in audit_do_ids:
                    base_vec = do.get_last_updates()
                    audit_mismatch_map[do.id] = [-v for v in base_vec]
                    print(f"[Round {round_idx}] 审计模拟：DO {do.id} SafeMul 使用不一致参数")
            enc_time = getattr(do, "last_encrypt_time", None)
            if enc_time is None:
                enc_time = t_enc_end - t_enc_start
            round_timing["do_encrypt_times"][do.id] = enc_time

        for attacker_id in [d.id for d in working_do_list if d is not None and d.id in attack_do_ids]:
            if attacker_id not in clean_update_map or not _should_poison_round(round_idx, attacker_id):
                continue
            base_vec = clean_update_map[attacker_id]
            dim = len(base_vec)
            poisoned: Optional[List[float]] = None
            label = attack_type.lower()

            if label == "stealth":
                poisoned = [(1.0 + attack_lambda) * x for x in base_vec]
                print(f"[Round {round_idx}] DO {attacker_id} 执行 Stealth Lie Attack（仅篡改密文），放大系数 1+λ={1.0 + attack_lambda}")
            elif label == "random":
                poisoned = [_rng.gauss(0.0, attack_sigma) for _ in range(dim)]
                print(f"[Round {round_idx}] DO {attacker_id} 执行 Random Attack，σ={attack_sigma}")
            elif label == "signflip":
                poisoned = [-(1.0 + attack_lambda) * x for x in base_vec]
                print(f"[Round {round_idx}] DO {attacker_id} 执行 Sign-Flip Attack，系数=-(1+λ)")
            elif label == "lie_stat":
                vectors = list(clean_update_map.values())
                count = len(vectors)
                mu = [0.0] * dim
                for vec in vectors:
                    for i, val in enumerate(vec):
                        mu[i] += val
                mu = [m / count for m in mu]
                sigma_vals = [0.0] * dim
                for vec in vectors:
                    for i, val in enumerate(vec):
                        diff = val - mu[i]
                        sigma_vals[i] += diff * diff
                sigma_vals = [math.sqrt(s / max(1, count - 1)) for s in sigma_vals]
                poisoned = [mu[i] + attack_lambda * sigma_vals[i] for i in range(dim)]
                print(f"[Round {round_idx}] DO {attacker_id} 执行统计型 Lie Attack，λ={attack_lambda}")
            else:
                print(f"[Round {round_idx}] DO {attacker_id} 未知攻击类型 {attack_type}，保持原样")

            if poisoned is not None:
                attacker = next((d for d in working_do_list if d is not None and d.id == attacker_id), None)
                if attacker is not None:
                    share = 1.0
                    if hasattr(ta, "get_data_share_for_do"):
                        try:
                            share = float(ta.get_data_share_for_do(attacker_id))
                        except Exception:
                            share = 1.0
                    if share <= 0.0:
                        share = 1.0
                    upload_poisoned = poisoned
                    if share != 1.0:
                        upload_poisoned = [share * v for v in poisoned]
                    do_cipher_map[attacker_id] = [attacker._encrypt_value(v) for v in upload_poisoned]
                    if label != "stealth":
                        clean_update_map[attacker_id] = list(poisoned)
                        if attacker.training_history:
                            attacker.training_history[-1]['local_updates'] = list(poisoned)
                            attacker.training_history[-1]['uploaded_updates'] = list(upload_poisoned)

        active_ids = [d.id for d in working_do_list if d is not None]
        poisoned_ids = _get_poisoned_ids(round_idx, active_ids)
        compressed_updates = None
        if enable_all_detection:
            # 原始向量投毒检测（未映射）
            det_raw = run_detection_suite(clean_update_map, "raw_vectors", temporary=True)
            detection_logs_raw.append({"round": round_idx, "log": det_raw["log"]})
            raw_suspects = det_raw.get("suspects", {})
            for method in detection_methods:
                raw_metrics.update(method, round_idx, raw_suspects.get(method), poisoned_ids, active_ids)
            if compression_spec is not None:
                compressed_updates = {do_id: _compress_vector(vec, compression_spec) for do_id, vec in clean_update_map.items()}
                det_comp_raw = run_detection_suite(compressed_updates, "compressed_raw_vectors", temporary=True)
                detection_logs_comp_raw.append({"round": round_idx, "log": det_comp_raw["log"]})
                comp_raw_suspects = det_comp_raw.get("suspects", {})
                for method in detection_methods:
                    comp_raw_metrics.update(method, round_idx, comp_raw_suspects.get(method), poisoned_ids, active_ids)
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
                    det_comp_proj = run_detection_suite(comp_proj_map, "compressed_projection", temporary=True)
                    detection_logs_comp_proj.append({"round": round_idx, "log": det_comp_proj["log"]})
                    comp_proj_suspects = det_comp_proj.get("suspects", {})
                    comp_active_ids = list(comp_proj_map.keys())
                    for method in detection_methods:
                        comp_proj_metrics.update(
                            method, round_idx, comp_proj_suspects.get(method), poisoned_ids, comp_active_ids
                        )

        print(f"\n===== Round {round_idx}: SafeMul 投影计算（在线 DO）=====")
        t1 = time.time()
        csp.do_projection_map.clear()
        online_dos = [d for d in working_do_list if d is not None]
        block_size = safe_mul_block_size
        use_block = block_size > 0 and block_size < orthogonal_vector_count
        if refresh_orthogonal_each_round and round_idx > 1:
            safe_mul_ctx_cache = None
            safe_mul_block_ctx_cache.clear()
        if not use_block:
            r1t = time.time()
            if safe_mul_ctx_cache is None:
                safe_mul_ctx_cache = csp.safe_mul_prepare_payload()
            ctx = safe_mul_ctx_cache
            r1te = time.time()
            round_timing["csp_safe_mul_prepare"] = r1te - r1t
            print(f"[Round {round_idx}] SafeMul 准备阶段耗时 {r1te - r1t:.4f}s")
            for do in online_dos:
                b_vec = audit_mismatch_map.get(do.id, do.get_last_updates())
                payload = {'p': ctx['p'], 'alpha': ctx['alpha'], 'C_all': ctx['C_all']}
                r2t = time.time()
                resp = do.safe_mul_round2_process(payload, b_vec)
                r2te = time.time()
                round_timing["do_safe_mul_round2_times"][do.id] = r2te - r2t
                print(f"[Round {round_idx}] DO {do.id} SafeMul 第二轮处理耗时 {r2te - r2t:.4f}s")

                projection = csp.safe_mul_finalize(ctx, resp['D_sums'], resp['do_part'], do.id)
                r3t = time.time()
                round_timing["csp_safe_mul_finalize_times"][do.id] = r3t - r2te
                print(f"[Round {round_idx}] CSP SafeMul 最终化 DO {do.id} 投影耗时 {r3t - r2te:.4f}s")
                print(f" DO {do.id} 投影向量(长度{len(projection)})")
        else:
            proj_map: Dict[int, List[float]] = {d.id: [] for d in online_dos}
            round_timing["csp_safe_mul_prepare"] = 0.0
            for start in range(0, orthogonal_vector_count, block_size):
                end = min(start + block_size, orthogonal_vector_count)
                r1t = time.time()
                cache_key = (start, end)
                if cache_key not in safe_mul_block_ctx_cache:
                    safe_mul_block_ctx_cache[cache_key] = csp.safe_mul_prepare_payload_block(start, end)
                ctx = safe_mul_block_ctx_cache[cache_key]
                r1te = time.time()
                print(f"[Round {round_idx}] SafeMul 准备阶段区间[{start},{end})耗时 {r1te - r1t:.4f}s")
                
                round_timing["csp_safe_mul_prepare"] += r1te - r1t
                for do in online_dos:
                    b_vec = audit_mismatch_map.get(do.id, do.get_last_updates())
                    payload = {'p': ctx['p'], 'alpha': ctx['alpha'], 'C_all': ctx['C_all']}
                    r2t = time.time()
                    resp = do.safe_mul_round2_process_block(payload, b_vec, start, end)
                    print(f"[Round {round_idx}] DO {do.id} SafeMul 第二轮区间[{start},{end})处理耗时 {time.time() - r2t:.4f}s")
                    r2te = time.time()
                    round_timing["do_safe_mul_round2_times"][do.id] = round_timing["do_safe_mul_round2_times"].get(do.id, 0.0) + (r2te - r2t)
                    r3t = time.time()
                    block_proj = csp.safe_mul_finalize_block(ctx, resp['D_sums'], resp['do_part'])
                    r3te = time.time()
                    round_timing["csp_safe_mul_finalize_times"][do.id] = round_timing["csp_safe_mul_finalize_times"].get(do.id, 0.0) + (r3te - r3t)
                    proj_map[do.id].extend(block_proj)
            csp.do_projection_map = proj_map
            for do_id, projection in proj_map.items():
                print(f" DO {do_id} 投影向量(长度{len(projection)})")
        t2 = time.time()
        print(f"[Round {round_idx}] SafeMul 投影耗时 {t2 - t1:.4f}s")

        # 投毒检测（映射后）
        if enable_all_detection:
            det_proj = run_detection_suite(csp.do_projection_map, "orthogonal_projection", temporary=False)
            detection_logs_proj.append({"round": round_idx, "log": det_proj["log"]})
            proj_suspects = det_proj.get("suspects", {})
            proj_active_ids = list(csp.do_projection_map.keys())
            for method in detection_methods:
                proj_metrics.update(method, round_idx, proj_suspects.get(method), poisoned_ids, proj_active_ids)

        print(f"\n===== Round {round_idx}: CSP 聚合 + 解密更新 =====")
        updated_params = csp.round_aggregate_and_update(working_do_list, do_cipher_map)
        print(f"[Round {round_idx}] 更新后的全局参数前5: {updated_params[:5]}")

        # 明文评估（可选，与 Train 同步）：每轮 train/val 指标，支持 BN 校准
        gap = _vector_gap(updated_params, global_params)
        summary = _vector_summary(updated_params)
        cos = _vector_cos(updated_params, global_params)
        print(
            f"[Round {round_idx}] 参数变化: ΔL2={gap['l2']:.6f}, Δ∞={gap['linf']:.6f}, cos(prev)={cos:.6f}; "
            f"当前均值={summary['mean']:.6f}, |max|={summary['abs_max']:.6f}"
        )
        # 记录本轮统计（proxy_loss 用 ΔL2 平均化作为无数据集的收敛近似指标）
        proxy_loss = (gap['l2'] ** 2) / max(1, len(updated_params))
        round_stats.append({
            "round": round_idx,
            "delta_l2": gap['l2'],
            "delta_inf": gap['linf'],
            "cos": cos,
            "mean": summary['mean'],
            "abs_max": summary['abs_max'],
            "proxy_loss": proxy_loss,
        })
        # 明文评估（可选，与 Train 同步）：每轮 train/val 指标，支持 BN 校准
        if eval_each_round:
            try:
                train_loss, train_acc, _, train_asr, train_src_acc, train_bd_asr = ModelTest.evaluate_params(
                    updated_params,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    data_root=os.path.join(os.path.dirname(__file__), "data"),
                    split="train",
                    bn_calib_batches=bn_calib_batches,
                    source_label=source_label,
                    target_label=target_label,
                    bd_target_label=bd_target_label if bd_enable else None,
                    bd_trigger_size=bd_trigger_size,
                    bd_trigger_value=bd_trigger_value,
                    bd_inject_ratio=bd_ratio if bd_enable else 0.0,
                )
                val_loss, val_acc, _, val_asr, val_src_acc, val_bd_asr = ModelTest.evaluate_params(
                    updated_params,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    data_root=os.path.join(os.path.dirname(__file__), "data"),
                    split="val",
                    bn_calib_batches=bn_calib_batches,
                    source_label=source_label,
                    target_label=target_label,
                    bd_target_label=bd_target_label if bd_enable else None,
                    bd_trigger_size=bd_trigger_size,
                    bd_trigger_value=bd_trigger_value,
                    bd_inject_ratio=bd_ratio if bd_enable else 0.0,
                )
                eval_history.append({
                    "round": round_idx,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_asr": train_asr,
                    "train_src_acc": train_src_acc,
                    "train_bd_asr": train_bd_asr,
                    "val_asr": val_asr,
                    "val_src_acc": val_src_acc,
                    "val_bd_asr": val_bd_asr,
                })
                extra_train = ""
                extra_val = ""
                if train_asr is not None and train_src_acc is not None:
                    extra_train = f", ASR(s->{target_label})={train_asr*100:.2f}%, src_acc={train_src_acc*100:.2f}%"
                if train_bd_asr is not None:
                    extra_train += f", bd_ASR={train_bd_asr*100:.2f}%"
                if val_asr is not None and val_src_acc is not None:
                    extra_val = f", ASR(s->{target_label})={val_asr*100:.2f}%, src_acc={val_src_acc*100:.2f}%"
                if val_bd_asr is not None:
                    extra_val += f", bd_ASR={val_bd_asr*100:.2f}%"
                print(f"[Eval][Round {round_idx}] Train loss={train_loss:.4f}, acc={train_acc*100:.2f}%{extra_train} | Val loss={val_loss:.4f}, acc={val_acc*100:.2f}%{extra_val}")
                metric = val_acc if not math.isnan(val_acc) else train_acc
                if metric > best_metric:
                    best_metric = metric
                    best_params = list(updated_params)
            except Exception as e:
                print(f"[Eval][Round {round_idx}] 评估失败: {e}")
        round_elapsed = time.time() - round_start
        round_times.append(round_elapsed)
        round_timing["round_total"] = round_elapsed
        round_timing_details.append(round_timing)
        print(f"[Round {round_idx}] 用时 {round_elapsed:.2f}s")
    total_elapsed = time.time() - total_start
    print("\n===== 全部轮次结束 =====")
    print(f"最终全局参数前5: {csp.global_params[:5]}")
    final_summary = _vector_summary(csp.global_params)
    print(
        f"最终参数统计: 均值={final_summary['mean']:.6f}, |max|={final_summary['abs_max']:.6f}"
    )
    # 可选：最终明文推理评估（与 Train 同步：支持 BN 校准）
    if eval_each_round:
        try:
            val_loss, val_acc, _, val_asr, val_src_acc, val_bd_asr = ModelTest.evaluate_params(
                csp.global_params,
                model_name=model_name,
                dataset_name=dataset_name,
                batch_size=eval_batch_size,
                max_batches=eval_batches,
                data_root=os.path.join(os.path.dirname(__file__), "data"),
                split="val",
                bn_calib_batches=bn_calib_batches,
                source_label=source_label,
                target_label=target_label,
                bd_target_label=bd_target_label if bd_enable else None,
                bd_trigger_size=bd_trigger_size,
                bd_trigger_value=bd_trigger_value,
                bd_inject_ratio=bd_ratio if bd_enable else 0.0,
            )
            extra_val = ""
            if val_asr is not None and val_src_acc is not None:
                extra_val = f", ASR(s->{target_label})={val_asr*100:.2f}%, src_acc={val_src_acc*100:.2f}%"
            if val_bd_asr is not None:
                extra_val += f", bd_ASR={val_bd_asr*100:.2f}%"
            print(f"[Final Eval] Val loss={val_loss:.4f}, acc={val_acc*100:.2f}%{extra_val}")
        except Exception as e:
            print(f"[Final Eval] 评估失败: {e}")
    # 保存最终全局参数
    try:
        # 为本次运行单独创建目录：trainResult/<run_tag>/
        base_dir = os.path.join("trainResult", run_tag)
        os.makedirs(base_dir, exist_ok=True)
        result_path = os.path.join(base_dir, "params.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "rounds": num_rounds,
                "model_size": len(csp.global_params),
                "params": csp.global_params,
                "model_name": model_name,
                "dataset_name": dataset_name,
            }, f)
        print(f"全局参数已保存至: {result_path}")
        # 保存最优参数（如评估开启）
        if best_params is not None:
            best_path = os.path.join(base_dir, "best_params.json")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rounds": num_rounds,
                    "model_size": len(best_params),
                    "params": best_params,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "best_metric": best_metric,
                }, f)
            print(f"最优全局参数已保存至: {best_path}")
    except Exception as e:
        print(f"保存全局参数失败: {e}")
    # 汇总各轮指标
    if round_stats:
        print("\n===== 训练指标汇总（逐轮） =====")
        for stat in round_stats:
            print(
                f"Round {stat['round']}: ΔL2={stat['delta_l2']:.6f}, Δ∞={stat['delta_inf']:.6f}, "
                f"cos(prev)={stat['cos']:.6f}, 均值={stat['mean']:.6f}, |max|={stat['abs_max']:.6f}, "
                f"proxy_loss={stat['proxy_loss']:.6e}"
            )
    # 记录日志到 txt（时间、评估、收敛）
    try:
        base_dir = os.path.join("trainResult", run_tag)
        os.makedirs(base_dir, exist_ok=True)
        log_path = os.path.join(base_dir, "log.txt")
        summary_lines = []
        if enable_all_detection:
            summary_lines.extend(raw_metrics.format_summary("raw_vectors"))
            summary_lines.extend(proj_metrics.format_summary("orthogonal_projection"))
            if compression_spec is not None:
                summary_lines.extend(comp_raw_metrics.format_summary("compressed_raw_vectors"))
                summary_lines.extend(comp_proj_metrics.format_summary("compressed_projection"))
        write_test_log(
            log_path=log_path,
            total_elapsed=total_elapsed,
            round_times=round_times,
            eval_history=eval_history,
            detection_logs_raw=detection_logs_raw if enable_all_detection else [],
            detection_logs_proj=detection_logs_proj if enable_all_detection else [],
            round_stats=round_stats,
            detection_logs_comp_raw=detection_logs_comp_raw if (enable_all_detection and compression_spec is not None) else None,
            detection_logs_comp_proj=detection_logs_comp_proj if (enable_all_detection and compression_spec is not None) else None,
            detection_summary_lines=summary_lines if enable_all_detection else None,
            timing_details=round_timing_details,
        )
        print(f"Log saved: {log_path}")
    except Exception as e:
        print(f"保存日志失败: {e}")

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
    parser = argparse.ArgumentParser(description="联邦学习模拟（加密流程），可选初始参数与 BN 校准")
    parser.add_argument("--initial-params-path", type=str, default=None,help="可选初始全局参数文件路径")
    parser.add_argument("--bn-calib-batches", type=int, default=30, help="评估时 BN 校准用的训练批次数（仅 ResNet 有效）")
    # 以下保持原默认示例参数，可按需扩展更多 CLI
    # num_rounds: 轮次 / num_do: DO 数 / model_size: 模型参数规模
    parser.add_argument("--num-rounds", type=int, default=15)#联邦轮次
    parser.add_argument("--num-do", type=int, default=3)#在线DO数
    parser.add_argument("--model-size", type=int, default=272474)  # cnn:61706 / lenet:81086 / resnet20:272474
    parser.add_argument("--orthogonal-vector-count", type=int, default=2048)
    parser.add_argument("--refresh-orthogonal-each-round", type=lambda x: str(x).lower() == "true", default=False, help="是否每轮刷新 TA 正交向量组（True/False）")
    parser.add_argument("--model-name", type=str, default="resnet20")    #模型名称：cnn /resnet20
    parser.add_argument("--dataset-name", type=str, default="cifar10")#数据集名称：mnist /cifar10
    # DO 训练相关，batch size / max batches
    parser.add_argument("--train-batch-size", type=int, default=64)#训练batch size
    parser.add_argument("--train-max-batches", type=int, default=300)#训练最大batch数
    parser.add_argument("--partition-mode",type=str,default="iid",help="数据划分模式：iid / mild / extreme（控制各 DO 的本地数据是否 non-IID）")
    # 评估相关
    parser.add_argument("--eval-each-round", dest="eval_each_round", action="store_true", default=True, help="开启每轮明文评估（会额外耗时，默认开启）")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=50)
    
    
    # untarget 梯度/参数投毒配置（与 label flip 解耦；留空则完全不启用 untarget）
    parser.add_argument("--attack-type",type=str,default=None,help="untarget 梯度/参数投毒类型：stealth/random/signflip/lie_stat；留空则不启用")
    parser.add_argument("--attack-round-untarget",type=str,default=None,help="untarget 梯度/参数投毒触发轮次：单个数字、逗号列表或 all；留空则不触发")
    parser.add_argument("--attack-lambda", type=float, default=0.2, help="untarget 投毒放大系数（stealth/signflip/lie_stat 使用）")
    parser.add_argument("--attack-sigma", type=float, default=1.0, help="untarget random 投毒的标准差")
    
    
    #target攻击： label flip 相关参数 source_label:源标签 / target_label:目标标签 / attack_do_id:攻击 DO id / attack_rounds:攻击轮次 / poison_ratio:翻转比例：该 batch 的源类样本中随机抽 30% 改标签，其余不动
    parser.add_argument("--attack-do-id",type=str,default=None,help="攻击 DO id-0,1（同时用于 label flip 和 untarget 投毒）如0,1；留空则不攻击（无需引号）")
    parser.add_argument("--source-label", type=int, default=None, help="若需监测 label flip，指定源标签，如2，代表把数据集中的2类数据投毒成3类数据")
    parser.add_argument("--target-label", type=int, default=None, help="若需监测 label flip，指定目标标签，如3，代表把数据集中的2类数据投毒成3类数据") 
    parser.add_argument("--attack-rounds", type=str, default="all", help="label flip 的攻击轮次，逗号分隔如 3,4,5，或 all 表示每轮；留空默认不触发")
    parser.add_argument("--poison-ratio", type=float, default=1.0, help="label flip 翻转比例，默认 0.3（源类样本内随机抽该比例改成目标标签）")
    # backdoor (BadNets-style) config
    parser.add_argument("--bd-enable", type=lambda x: str(x).lower() == "true", default=False, help="Enable backdoor (BadNets) when set to True, e.g. --bd-enable True")
    parser.add_argument("--bd-target-label", type=int, default=9, help="Backdoor target label")
    parser.add_argument("--bd-ratio", type=float, default=0.3, help="Backdoor injection ratio in source class")
    parser.add_argument("--bd-trigger-size", type=int, default=2, help="Backdoor trigger size (pixels)")
    parser.add_argument("--bd-trigger-value", type=float, default=3.0, help="Backdoor trigger value (pre-normalization)")
    
    #压缩向量检测相关，是否分块，是否开启投毒检测
    parser.add_argument("--enable-compressed-detect", type=lambda x: str(x).lower() == "true", default=False, help="Enable compressed vector detection (ResNet20 only)")
    parser.add_argument("--enable-all-detection", type=lambda x: str(x).lower() == "true", default=True, help="是否开启全部投毒检测（True/False）")
    parser.add_argument("--detection-methods", type=str, default="all", help="投毒检测方案：all 或 multi,geo,cluster（逗号分隔）")
    parser.add_argument("--safe-mul-block-size", type=int, default=512, help="SafeMul 分块大小，<=0 表示不分块")
    # 审计模拟相关：DO掉线，DO恶意使用不一致模型参数
    parser.add_argument("--audit-round", type=int, default=1, help="审计模拟轮次（单轮）")
    parser.add_argument("--audit-do-id", type=str, default="1", help="审计模拟的 DO id，逗号分隔如 0,1")
    parser.add_argument("--audit-simulate-dropout", type=lambda x: str(x).lower() == "true", default=False, help="审计模拟：是否让指定 DO 掉线")
    parser.add_argument("--audit-simulate-mismatch", type=lambda x: str(x).lower() == "true", default=True, help="审计模拟：SafeMul 与加密参数不一致")
    args = parser.parse_args()

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

    attack_rounds = None
    if args.attack_rounds:
        if args.attack_rounds.lower() == "all":
            attack_rounds = "all"
        else:
            attack_rounds = [int(x) for x in args.attack_rounds.split(",") if x.strip()]

    attack_do_ids: List[int] = []
    if args.attack_do_id:
        attack_do_ids = [int(x) for x in args.attack_do_id.split(",") if x.strip()]

    attack_round_parsed: Optional[int] = None
    attack_rounds_untarget: Optional[object] = None
    if args.attack_round_untarget:
        if args.attack_round_untarget.lower() == "all":
            print("untarget 梯度/参数投毒将在所有轮次触发")
            attack_rounds_untarget = "all"
        elif "," in args.attack_round_untarget:
            # 支持形如 "3,4,5" 的多轮触发
            attack_rounds_untarget = [
                int(x) for x in args.attack_round_untarget.split(",") if x.strip()
            ]
        else:
            # 单个数字：只在该轮触发
            try:
                attack_round_parsed = int(args.attack_round_untarget)
            except ValueError:
                attack_round_parsed = None

    run_federated_simulation(
        num_rounds=args.num_rounds,
        num_do=args.num_do,
        model_size=args.model_size,
        orthogonal_vector_count=args.orthogonal_vector_count,
        bit_length=512,
        precision=10 ** 6,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        train_batch_size=args.train_batch_size,
        train_max_batches=args.train_max_batches,
        initial_params_path=args.initial_params_path,
        bn_calib_batches=args.bn_calib_batches,
        eval_each_round=args.eval_each_round,
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        source_label=args.source_label,
        target_label=args.target_label,
        attack_do_ids=attack_do_ids,
        attack_rounds=attack_rounds,
        poison_ratio=args.poison_ratio,
        partition_mode=args.partition_mode,
        attack_type=args.attack_type,
        attack_round=attack_round_parsed,
        attack_rounds_untarget=attack_rounds_untarget,
        attack_lambda=args.attack_lambda,
        attack_sigma=args.attack_sigma,
        bd_enable=args.bd_enable,
        bd_target_label=args.bd_target_label,
        bd_ratio=args.bd_ratio,
        bd_trigger_size=args.bd_trigger_size,
        bd_trigger_value=args.bd_trigger_value,
        enable_compressed_detect=args.enable_compressed_detect,
        refresh_orthogonal_each_round=args.refresh_orthogonal_each_round,
        safe_mul_block_size=args.safe_mul_block_size,
        enable_all_detection=args.enable_all_detection,
        detection_methods=selected_methods,
        audit_round=args.audit_round,
        audit_do_ids=[int(x) for x in args.audit_do_id.split(",") if x.strip()] if args.audit_do_id else [],
        audit_simulate_dropout=args.audit_simulate_dropout,
        audit_simulate_mismatch=args.audit_simulate_mismatch,
    )
