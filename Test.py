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
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from TA.TA import TA
from CSP.CSP import CSP
from DO.DO import DO


def run_federated_simulation(
    num_rounds: int = 2,
    num_do: int = 3,
    model_size: int = 10_000,
    orthogonal_vector_count: int = 1_024,
    bit_length: int = 512,
    precision: int = 10 ** 6,
    dropouts: Optional[Dict[int, List[int]]] = None,
    dropout_round: Optional[int] = None,
    dropout_do_ids: Optional[List[int]] = None,
    attack_round: Optional[int] = 1,
    attack_do_id: Optional[int] = 2,
    attack_type: str = "random",
    #stealth隐蔽投毒，二分找到恶意DO / random随机数 / signflip反向梯度 /lie_stat放大梯度
    attack_lambda: float = 0.2,
    attack_sigma: float = 1.0,
) -> None:
    """
    运行一次联邦学习流程。
    - dropouts / dropout_round+dropout_do_ids：配置掉线 DO
    - attack_round/attack_do_id/attack_lambda：在指定轮次对指定 DO 进行 Lie Attack（基于自身梯度放大 1+lambda）
    """
    dropouts = dropouts or {}
    if dropout_round is not None and dropout_do_ids:
        dropouts = dict(dropouts)
        dropouts[dropout_round] = dropout_do_ids
    _rng = random.Random(2025)

    print("===== 初始化 TA / CSP / DO 列表 =====")
    ta = TA(
        num_do=num_do,
        model_size=model_size,
        orthogonal_vector_count=orthogonal_vector_count,
        bit_length=bit_length,
        precision=precision,
    )
    csp = CSP(ta, model_size=model_size, precision=precision)
    do_list: List[Optional[DO]] = [DO(i, ta, model_size=model_size, precision=precision) for i in range(num_do)]

    def run_detection_suite(vector_map: Dict[int, List[float]], label: str, temporary: bool = True) -> None:
        if not vector_map:
            return
        backup = None
        if temporary:
            backup = csp.do_projection_map
            csp.do_projection_map = {k: list(v) for k, v in vector_map.items()}
        print(f"\n----- 投毒检测（{label}）-----")
        active_count = len(csp.do_projection_map)
        if active_count >= 2:
            csp.detect_poison_multi_krum(f=1, alpha=1.5)
            csp.detect_poison_geomedian(beta=1.5)
            csp.detect_poison_clustering(k=min(3, active_count), alpha=1.5)
            csp.detect_poison_lasa_lite(angle_threshold=0.0, beta=1.5)
        else:
            print("[检测] 在线 DO 数不足，跳过。")
        if temporary:
            csp.do_projection_map = backup

    for round_idx in range(1, num_rounds + 1):
        print(f"\n===== Round {round_idx}: 广播参数 =====")
        global_params = csp.broadcast_params()

        # 掉线处理
        working_do_list: List[Optional[DO]] = list(do_list)
        for drop_id in dropouts.get(round_idx, []):
            if 0 <= drop_id < len(working_do_list):
                working_do_list[drop_id] = None
        offline_ids = [i for i, d in enumerate(working_do_list) if d is None]
        if offline_ids:
            print(f"[Round {round_idx}] 模拟掉线 DO: {offline_ids}")

        # DO 训练并上传
        print(f"\n===== Round {round_idx}: DO 训练并加密 =====")
        do_cipher_map: Dict[int, List[int]] = {}
        clean_update_map: Dict[int, List[float]] = {}
        for do in [d for d in working_do_list if d is not None]:
            ciphertexts = do.train_and_encrypt(global_params)
            do_cipher_map[do.id] = ciphertexts
            clean_update_map[do.id] = do.get_last_updates()

        if (
            attack_round is not None
            and attack_do_id is not None
            and round_idx == attack_round
            and attack_do_id in clean_update_map
        ):
            base_vec = clean_update_map[attack_do_id]
            dim = len(base_vec)
            poisoned: Optional[List[float]] = None
            label = attack_type.lower()

            if label == "stealth":
                poisoned = [(1.0 + attack_lambda) * x for x in base_vec]
                print(f"[Round {round_idx}] DO {attack_do_id} 执行 Stealth Lie Attack（仅篡改密文），放大系数 1+λ={1.0 + attack_lambda}")
            elif label == "random":
                poisoned = [_rng.gauss(0.0, attack_sigma) for _ in range(dim)]
                print(f"[Round {round_idx}] DO {attack_do_id} 执行 Random Attack，σ={attack_sigma}")
            elif label == "signflip":
                poisoned = [-(1.0 + attack_lambda) * x for x in base_vec]
                print(f"[Round {round_idx}] DO {attack_do_id} 执行 Sign-Flip Attack，系数=-(1+λ)")
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
                print(f"[Round {round_idx}] DO {attack_do_id} 执行统计型 Lie Attack，λ={attack_lambda}")
            else:
                print(f"[Round {round_idx}] DO {attack_do_id} 未知攻击类型 {attack_type}，保持原样")

            if poisoned is not None:
                attacker = next((d for d in working_do_list if d is not None and d.id == attack_do_id), None)
                if attacker is not None:
                    do_cipher_map[attack_do_id] = [attacker._encrypt_value(v) for v in poisoned]
                    if label != "stealth":
                        clean_update_map[attack_do_id] = list(poisoned)
                        if attacker.training_history:
                            attacker.training_history[-1]['local_updates'] = list(poisoned)

        # 原始向量投毒检测（未映射）
        run_detection_suite(clean_update_map, "原始向量", temporary=True)

        # SafeMul 投影
        print(f"\n===== Round {round_idx}: SafeMul 投影计算（在线 DO）=====")
        t1 = time.time()
        csp.do_projection_map.clear()
        ctx = csp.safe_mul_prepare_payload()
        for do in [d for d in working_do_list if d is not None]:
            b_vec = do.get_last_updates()
            payload = {'p': ctx['p'], 'alpha': ctx['alpha'], 'C_all': ctx['C_all']}
            resp = do.safe_mul_round2_process(payload, b_vec)
            projection = csp.safe_mul_finalize(ctx, resp['D_sums'], resp['do_part'], do.id)
            print(f" DO {do.id} 投影向量(长度{len(projection)})")
        t2 = time.time()
        print(f"[Round {round_idx}] SafeMul 投影耗时 {t2 - t1:.4f}s")

        # 投毒检测（映射后）
        run_detection_suite(csp.do_projection_map, "正交投影向量", temporary=False)

        # 聚合 + 解密 + 更新
        print(f"\n===== Round {round_idx}: CSP 聚合 + 解密更新 =====")
        updated_params = csp.round_aggregate_and_update(working_do_list, do_cipher_map)
        print(f"[Round {round_idx}] 更新后的全局参数前5: {updated_params[:5]}")

    print("\n===== 全部轮次结束 =====")
    print(f"最终全局参数前5: {csp.global_params[:5]}")


if __name__ == "__main__":
    # 示例：2 轮，5 个 DO；第 2 轮让 DO2 掉线，DO1 做 Lie Attack（放大 1.2）
    run_federated_simulation(
        num_rounds=2,
        num_do=30,
        model_size=10_000,
        orthogonal_vector_count=1_024,
        bit_length=512,
        precision=10 ** 6,
        attack_type= "signflip",
        # dropouts={1: [2]},
        attack_round=1,
        attack_do_id=4,
        attack_lambda=0.2,
    )
