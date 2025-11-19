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
    attack_round: Optional[int] = None,
    attack_do_id: Optional[int] = None,
    attack_lambda: float = 0.2,
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
        for do in [d for d in working_do_list if d is not None]:
            ciphertexts = do.train_and_encrypt(global_params)
            if (
                attack_round is not None
                and attack_do_id is not None
                and round_idx == attack_round
                and do.id == attack_do_id
            ):
                poisoned = [(1.0 + attack_lambda) * x for x in do.get_last_updates()]
                if do.training_history:
                    do.training_history[-1]['local_updates'] = list(poisoned)
                ciphertexts = [do._encrypt_value(v) for v in poisoned]
                print(f"[Round {round_idx}] DO {do.id} 执行 Lie Attack，放大系数 1+λ={1.0 + attack_lambda}")
            do_cipher_map[do.id] = ciphertexts

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

        # 投毒检测
        csp.detect_poison_multi_krum(f=1, alpha=1.5)
        csp.detect_poison_geomedian(beta=1.5)

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
        num_do=5,
        model_size=50_000,
        orthogonal_vector_count=2_048,
        bit_length=512,
        precision=10 ** 6,
        # dropouts={2: [2]},
        attack_round=2,
        attack_do_id=1,
        attack_lambda=0.2,
    )
