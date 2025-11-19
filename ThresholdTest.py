#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from TA.TA import TA
from CSP.CSP import CSP
from DO.DO import DO


def run_threshold_recovery_demo(num_do: int = 3, model_size: int = 5, bit_length: int = 512):
    print("===== ThresholdTest: 掉线恢复 n_i 并继续聚合 =====")

    # 1) 初始化 TA / CSP / DO
    ta = TA(num_do=num_do, model_size=model_size, orthogonal_vector_count=3, bit_length=bit_length)
    csp = CSP(ta, model_size=model_size)
    do_list = [DO(i, ta, model_size=model_size) for i in range(num_do)]

    # 2) 广播参数（Round 1）
    print("\n[Round 1] 广播全局参数")
    global_params = csp.broadcast_params()

    # 3) 各 DO 训练并加密（Round 1）
    do_cipher_map = {}
    for do in do_list:
        do_cipher_map[do.id] = do.train_and_encrypt(global_params)

    # 4) 聚合与解密（Round 1）
    print("\n[Round 1] 聚合与解密")
    _ = csp.round_aggregate_and_update(do_list, do_cipher_map)

    # 5) 模拟 DO 掉线（Round 2）
    missing_id = 1 if num_do >= 2 else 0
    print(f"\n[Round 2] 模拟 DO{missing_id} 掉线")
    global_params = csp.broadcast_params()
    do_list_with_missing = do_list.copy()
    do_list_with_missing[missing_id] = None

    # 6) 在线 DO 上传密文（Round 2）
    do_cipher_map = {}
    for do in [d for d in do_list_with_missing if d is not None]:
        do_cipher_map[do.id] = do.train_and_encrypt(global_params)

    # 7) CSP 侧进行 n_i 恢复并完成聚合解密
    print("\n[Round 2] 聚合与带恢复的解密")
    updated_params = csp.round_aggregate_and_update(do_list_with_missing, do_cipher_map)

    # 8) 校验：CSP 恢复到的 n_i 与 TA 中的 n_i 是否一致（门限满足：应恢复成功）
    if missing_id in csp.recoveredNiValues:
        recovered_n = csp.recoveredNiValues[missing_id]
        ground_truth_n = ta.get_ni(missing_id)
        print(f"\n[校验] 恢复的 n_{missing_id}: {recovered_n}")
        print(f"[校验] 真实的 n_{missing_id}: {ground_truth_n}")
        print(f"[校验] 恢复正确: {recovered_n == ground_truth_n}")
    else:
        print("\n[校验] 本轮未恢复到缺失 n_i（可能在线 DO 数量不足阈值）")

    print(f"\n[结果] Round 2 更新后的全局参数: {updated_params}")

    # 9) 场景二：门限不足，无法恢复（让两名 DO 掉线，仅剩 1 名在线，阈值=2 时应失败）
    print("\n[Round 3] 模拟两名 DO 掉线，门限不足，期望无法恢复")
    global_params = csp.broadcast_params()
    do_list_round3 = do_list.copy()
    # 让 DO0、DO1 掉线，仅保留 DO2（若 num_do=3）
    missing_ids_round3 = []
    # 尽量让在线数量 = 1
    online_kept = None
    for i in range(num_do):
        if online_kept is None:
            online_kept = i  # 保留第一个
            continue
        do_list_round3[i] = None
        missing_ids_round3.append(i)

    do_cipher_map = {}
    for do in [d for d in do_list_round3 if d is not None]:
        do_cipher_map[do.id] = do.train_and_encrypt(global_params)

    print(f"[Round 3] 掉线 DO: {missing_ids_round3}，仅在线 DO: {online_kept}")
    updated_params_r3 = csp.round_aggregate_and_update(do_list_round3, do_cipher_map)

    # 校验：由于仅 1 份分片 (< threshold)，应无法恢复任何缺失的 n_i
    threshold_needed = ta.get_threshold()
    if csp.recoveredNiValues:
        print(f"[校验-异常] 仍然恢复到了 {csp.recoveredNiValues}，但门限需要 {threshold_needed}，请检查实现。")
    else:
        print(f"[校验-预期] 门限不足（需要 {threshold_needed}），未能恢复缺失 n_i。")

    print(f"\n[结果] Round 3 更新后的全局参数: {updated_params_r3}")
    print("\n===== ThresholdTest 结束 =====")


if __name__ == "__main__":
    run_threshold_recovery_demo()


