#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, List, Optional

# 确保可以导入项目内模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from TA.TA import TA
from CSP.CSP import CSP
from DO.DO import DO


def run_federated_simulation(num_rounds: int = 10,
                             num_do: int = 5,
                             model_size: int = 5,
                             orthogonal_vector_count: int = 5,
                             bit_length: int = 512,
                             precision: int = 10 ** 6,
                             dropout_round: int = 9,
                             dropout_do_id: int = 2) -> None:
    """
    运行联邦学习模拟：
    - 使用 ImprovedPaillier
    - 5 个 DO，门限 = 3（TA中按 2/3 n 计算，5 -> 3）
    - 正交向量组 5x5
    - 共 10 轮；第 3 轮模拟 1 个 DO 掉线
    - 每轮密钥由 TA 更新，DO 基于全局参数哈希派生轮次密钥
    """

    print("===== 初始化 TA / CSP / DO 列表 =====")
    ta = TA(num_do=num_do,
            model_size=model_size,
            orthogonal_vector_count=orthogonal_vector_count,
            bit_length=bit_length,
            precision=precision)
    # 断言门限为 3
    assert ta.get_threshold() == 3, f"期望阈值为3，实际为 {ta.get_threshold()}"

    csp = CSP(ta, model_size=model_size, precision=precision)
    do_list: List[Optional[DO]] = [DO(i, ta, model_size=model_size, precision=precision) for i in range(num_do)]

    print("===== 开始联邦学习回合 =====")
    global_params: List[float] = csp.global_params[:]  # 初始全局参数

    for round_idx in range(1, num_rounds + 1):
        print(f"\n===== Round {round_idx}: 广播参数 =====")
        # CSP 广播参数时，TA 会更新密钥（R_t 与各 DO 的基础私钥）
        global_params = csp.broadcast_params()

        # 构造当轮 DO 列表（第 dropout_round 轮模拟掉线）
        if round_idx == dropout_round:
            print(f"[TEST] 模拟 DO {dropout_do_id} 在本轮掉线")
            working_do_list: List[Optional[DO]] = [d for d in do_list]
            working_do_list[dropout_do_id] = None
        else:
            working_do_list = do_list

        # 在线 DO 训练并加密上传
        print(f"\n===== Round {round_idx}: 在线 DO 训练并加密 =====")
        do_cipher_map: Dict[int, List[int]] = {}
        for do in [d for d in working_do_list if d is not None]:
            ciphertexts = do.train_and_encrypt(global_params)
            do_cipher_map[do.id] = ciphertexts

        # CSP 聚合与（必要时）门限恢复解密并更新
        print(f"\n===== Round {round_idx}: CSP 聚合 + 解密更新 =====")
        updated_params = csp.round_aggregate_and_update(working_do_list, do_cipher_map)
        print(f">>> Round {round_idx} 结束，新全局参数: {updated_params}")

    print("\n===== 全部轮次完成 =====")
    print(f"最终全局参数: {csp.global_params}")


if __name__ == "__main__":
    run_federated_simulation()
