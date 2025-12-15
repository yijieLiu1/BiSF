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
import json
import argparse
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from TA.TA import TA
from CSP.CSP import CSP
from DO.DO import DO
import ModelTest

#默认参数。Test调用时还会传参
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
    #stealth隐蔽投毒，二分找到恶意DO / random随机数 / signflip反向梯度 /lie_stat放大梯度
    attack_type: str = None,
    attack_lambda: float = 0.2,
    attack_sigma: float = 1.0,
    #模型名称和数据集名称："lenet/mnist_cnn","mnist/cifar10"
    model_name: str = "mnist_cnn",
    dataset_name: str = "mnist",
    #DO训练的batch选项
    train_batch_size: Optional[int] = None,
    train_max_batches: Optional[int] = None,
    #可选的模型参数，可以指定使用已训练过的全局模型参数作为初始参数：
    initial_params_path="trainResult/lenet_cifar10_global_params_round10.json",
    # initial_params_path: Optional[str] = None,
    # 评估相关：每轮在 train/val 上做推理（明文），可选 BN 校准（ResNet）
    #是否每一轮都进行一次推理，查看准确率
    eval_each_round: bool = True,
    eval_batch_size: int = 256,
    eval_batches: int = 50,
    bn_calib_batches: int = 0,
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
    csp = CSP(ta, model_size=model_size, precision=precision, initial_params_path=initial_params_path)
    do_list: List[Optional[DO]] = [
        DO(
            i,
            ta,
            model_size=model_size,
            precision=precision,
            model_name=model_name,
            dataset_name=dataset_name,
            batch_size=train_batch_size,
            max_batches=train_max_batches,
        )
        for i in range(num_do)
    ]
    round_stats: List[Dict[str, float]] = []

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

    round_times = []
    eval_history: List[Dict[str, float]] = []
    best_params = None
    best_metric = -1.0  # 优先用 val acc，如果没有则用 train acc
    # 构造统一的运行标签：模型_数据集_do数_场景_轮数
    def _build_run_tag() -> str:
        model_tag = model_name.lower().replace(" ", "_")
        dataset_tag = dataset_name.lower().replace(" ", "_")
        if attack_type and attack_round is not None:
            scenario_tag = f"{attack_type.lower()}投毒加检测"
        else:
            scenario_tag = "正常"
        scenario_tag = scenario_tag.replace(" ", "")
        return f"{model_tag}_{dataset_tag}_{num_do}do_{scenario_tag}_{num_rounds}r"

    run_tag = _build_run_tag()

    total_start = time.time()
    for round_idx in range(1, num_rounds + 1):
        round_start = time.time()
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

        # 明文评估（可选，与 Train 同步）：每轮 train/val 指标，支持 BN 校准
        if eval_each_round:
            try:
                train_loss, train_acc, _ = ModelTest.evaluate_params(
                    updated_params,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    data_root=os.path.join(os.path.dirname(__file__), "data"),
                    split="train",
                    bn_calib_batches=bn_calib_batches,
                )
                val_loss, val_acc, _ = ModelTest.evaluate_params(
                    updated_params,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    data_root=os.path.join(os.path.dirname(__file__), "data"),
                    split="val",
                    bn_calib_batches=bn_calib_batches,
                )
                print(f"[Eval][Round {round_idx}] Train loss={train_loss:.4f}, acc={train_acc*100:.2f}% | Val loss={val_loss:.4f}, acc={val_acc*100:.2f}%")
            except Exception as e:
                print(f"[Eval][Round {round_idx}] 评估失败: {e}")

        # 训练效果评估（无测试集版本）：看参数值收敛性
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
                train_loss, train_acc, _ = ModelTest.evaluate_params(
                    updated_params,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    data_root=os.path.join(os.path.dirname(__file__), "data"),
                    split="train",
                    bn_calib_batches=bn_calib_batches,
                )
                val_loss, val_acc, _ = ModelTest.evaluate_params(
                    updated_params,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    data_root=os.path.join(os.path.dirname(__file__), "data"),
                    split="val",
                    bn_calib_batches=bn_calib_batches,
                )
                eval_history.append({
                    "round": round_idx,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                print(f"[Eval][Round {round_idx}] Train loss={train_loss:.4f}, acc={train_acc*100:.2f}% | Val loss={val_loss:.4f}, acc={val_acc*100:.2f}%")
                metric = val_acc if not math.isnan(val_acc) else train_acc
                if metric > best_metric:
                    best_metric = metric
                    best_params = list(updated_params)
            except Exception as e:
                print(f"[Eval][Round {round_idx}] 评估失败: {e}")
        round_elapsed = time.time() - round_start
        round_times.append(round_elapsed)
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
            val_loss, val_acc, _ = ModelTest.evaluate_params(
                csp.global_params,
                model_name=model_name,
                dataset_name=dataset_name,
                batch_size=eval_batch_size,
                max_batches=eval_batches,
                data_root=os.path.join(os.path.dirname(__file__), "data"),
                split="val",
                bn_calib_batches=bn_calib_batches,
            )
            print(f"[Final Eval] Val loss={val_loss:.4f}, acc={val_acc*100:.2f}%")
        except Exception as e:
            print(f"[Final Eval] 评估失败: {e}")
    # 保存最终全局参数
    try:
        os.makedirs("trainResult", exist_ok=True)
        result_path = os.path.join("trainResult", f"{run_tag}_params.json")
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
            best_path = os.path.join("trainResult", f"{run_tag}_best_params.json")
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
        log_path = os.path.join("trainResult", f"{run_tag}_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"总用时: {total_elapsed:.2f}s\n")
            f.write("按轮用时(s): " + ", ".join(f"{t:.2f}" for t in round_times) + "\n\n")
            if eval_history:
                f.write("评估指标（train/val）:\n")
                for ev in eval_history:
                    f.write(
                        f"Round {ev['round']}: Train loss={ev['train_loss']:.4f}, acc={ev['train_acc']*100:.2f}%; "
                        f"Val loss={ev['val_loss']:.4f}, acc={ev['val_acc']*100:.2f}%\n"
                    )
                f.write("\n")
            if round_stats:
                f.write("收敛指标（ΔL2, ΔL∞）:\n")
                for stat in round_stats:
                    f.write(
                        f"Round {stat['round']}: ΔL2={stat['delta_l2']:.6f}, Δ∞={stat['delta_inf']:.6f}, "
                        f"cos(prev)={stat['cos']:.6f}, 均值={stat['mean']:.6f}, |max|={stat['abs_max']:.6f}, "
                        f"proxy_loss={stat['proxy_loss']:.6e}\n"
                    )
            print(f"日志已保存: {log_path}")
    except Exception as e:
        print(f"保存日志失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="联邦学习模拟（加密流程），可选初始参数与 BN 校准")
    parser.add_argument("--initial-params-path", type=str, default=None,help="可选初始全局参数文件路径")
    parser.add_argument("--bn-calib-batches", type=int, default=30, help="评估时 BN 校准用的训练批次数（仅 ResNet 有效）")
    # 以下保持原默认示例参数，可按需扩展更多 CLI
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--num-do", type=int, default=10)
    parser.add_argument("--model-size", type=int, default=56714)  # cnn:56714 / lenet:81086 / resnet20:272186
    parser.add_argument("--model-name", type=str, default="cnn")#cnn /lenet /resnet20
    parser.add_argument("--dataset-name", type=str, default="mnist")#mnist /cifar10
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-max-batches", type=int, default=300)
    parser.add_argument("--eval-each-round", dest="eval_each_round", action="store_true", default=True, help="开启每轮明文评估（会额外耗时，默认开启）")
    parser.add_argument("--no-eval-each-round", dest="eval_each_round", action="store_false", help="关闭每轮明文评估")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=50)
    args = parser.parse_args()

    run_federated_simulation(
        num_rounds=args.num_rounds,
        num_do=args.num_do,
        model_size=args.model_size,
        orthogonal_vector_count=2_048,
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
    )
