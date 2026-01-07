#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared logging helpers for training and encrypted testing runs.
"""
from typing import Dict, List, Optional, Sequence


def write_train_log(
    log_path: str,
    total_elapsed: float,
    round_times: Sequence[float],
    convergence_history: Sequence[Dict[str, float]],
    train_metric_history: Sequence[Dict[str, float]],
    val_metric_history: Sequence[Dict[str, float]],
    detection_logs_raw: Sequence[Dict[str, str]],
    detection_logs_proj: Sequence[Dict[str, str]],
    detection_logs_comp_raw: Optional[Sequence[Dict[str, str]]] = None,
    detection_logs_comp_proj: Optional[Sequence[Dict[str, str]]] = None,
    detection_summary_lines: Optional[Sequence[str]] = None,
) -> None:
    """Write Train.py style log file."""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"总用时: {total_elapsed:.2f}s\n")
        f.write("按轮用时(s): " + ", ".join(f"{t:.2f}" for t in round_times) + "\n\n")
        if convergence_history:
            f.write("收敛指标（ΔL2, ΔL∞）:\n")
            for item in convergence_history:
                f.write(
                    f"Round {item['round']}: L2={item['l2']:.6f}, L∞={item['linf']:.6f}\n"
                )
            f.write("\n")
        if train_metric_history and val_metric_history:
            f.write("训练/验证指标:\n")
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
                f.write(
                    f"Round {tm['round']}: Train loss={tm['loss']:.4f}, acc={tm['acc']*100:.2f}%{extra_train} | "
                    f"Val loss={vm['loss']:.4f}, acc={vm['acc']*100:.2f}%{extra_val}\n"
                )
        f.write("\n")
        if detection_logs_raw:
            f.write("投毒检测日志（原始向量，Plain）:\n")
            for item in detection_logs_raw:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_logs_proj:
            f.write("投毒检测日志（正交投影向量，Plain）:\n")
            for item in detection_logs_proj:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_logs_comp_raw:
            f.write("投毒检测日志（压缩原始向量，Plain）:\n")
            for item in detection_logs_comp_raw:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_logs_comp_proj:
            f.write("投毒检测日志（压缩正交投影向量，Plain）:\n")
            for item in detection_logs_comp_proj:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_summary_lines:
            for line in detection_summary_lines:
                f.write(line + "\n")
            f.write("\n")


def write_test_log(
    log_path: str,
    total_elapsed: float,
    round_times: Sequence[float],
    eval_history: Sequence[Dict[str, float]],
    detection_logs_raw: Sequence[Dict[str, str]],
    detection_logs_proj: Sequence[Dict[str, str]],
    round_stats: Sequence[Dict[str, float]],
    detection_logs_comp_raw: Optional[Sequence[Dict[str, str]]] = None,
    detection_logs_comp_proj: Optional[Sequence[Dict[str, str]]] = None,
    detection_summary_lines: Optional[Sequence[str]] = None,
    timing_details: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    """Write Test.py style log file."""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"总用时: {total_elapsed:.2f}s\n")
        f.write("按轮用时(s): " + ", ".join(f"{t:.2f}" for t in round_times) + "\n\n")
        if eval_history:
            f.write("评估指标（train/val）:\n")
            for ev in eval_history:
                extra_train = ""
                extra_val = ""
                if ev.get("train_asr") is not None and ev.get("train_src_acc") is not None:
                    extra_train = f", ASR={ev['train_asr']*100:.2f}%, src_acc={ev['train_src_acc']*100:.2f}%"
                if ev.get("train_bd_asr") is not None:
                    extra_train += f", bd_ASR={ev['train_bd_asr']*100:.2f}%"
                if ev.get("val_asr") is not None and ev.get("val_src_acc") is not None:
                    extra_val = f", ASR={ev['val_asr']*100:.2f}%, src_acc={ev['val_src_acc']*100:.2f}%"
                if ev.get("val_bd_asr") is not None:
                    extra_val += f", bd_ASR={ev['val_bd_asr']*100:.2f}%"
                f.write(
                    f"Round {ev['round']}: Train loss={ev['train_loss']:.4f}, acc={ev['train_acc']*100:.2f}%{extra_train}; "
                    f"Val loss={ev['val_loss']:.4f}, acc={ev['val_acc']*100:.2f}%{extra_val}\n"
                )
            f.write("\n")
        if detection_logs_raw:
            f.write("投毒检测日志（原始向量）:\n")
            for item in detection_logs_raw:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_logs_proj:
            f.write("投毒检测日志（正交投影向量）:\n")
            for item in detection_logs_proj:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_logs_comp_raw:
            f.write("投毒检测日志（压缩原始向量）:\n")
            for item in detection_logs_comp_raw:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if detection_logs_comp_proj:
            f.write("投毒检测日志（压缩正交投影向量）:\n")
            for item in detection_logs_comp_proj:
                f.write(f"[Round {item['round']}]\n{item['log']}\n")
            f.write("\n")
        if round_stats:
            f.write("收敛指标（ΔL2, ΔL∞）:\n")
            for stat in round_stats:
                f.write(
                    f"Round {stat['round']}: ΔL2={stat['delta_l2']:.6f}, Δ∞={stat['delta_inf']:.6f}, "
                    f"cos(prev)={stat['cos']:.6f}, 均值={stat['mean']:.6f}, |max|={stat['abs_max']:.6f}, "
                    f"proxy_loss={stat['proxy_loss']:.6e}\n"
                )
        if timing_details:
            f.write("\n逐轮用时明细:\n")
            for item in timing_details:
                f.write(f"Round {item.get('round')}:\n")
                f.write(f"  round_total={item.get('round_total', 0.0):.4f}s\n")
                f.write(f"  csp_safe_mul_prepare={item.get('csp_safe_mul_prepare', 0.0):.4f}s\n")
                enc_map = item.get("do_encrypt_times", {}) or {}
                for do_id, t_val in enc_map.items():
                    f.write(f"  do{do_id}_encrypt={t_val:.4f}s\n")
                r2_map = item.get("do_safe_mul_round2_times", {}) or {}
                for do_id, t_val in r2_map.items():
                    f.write(f"  do{do_id}_safe_mul_round2={t_val:.4f}s\n")
                fin_map = item.get("csp_safe_mul_finalize_times", {}) or {}
                for do_id, t_val in fin_map.items():
                    f.write(f"  csp_safe_mul_finalize_do{do_id}={t_val:.4f}s\n")
        if detection_summary_lines:
            f.write("\n")
            for line in detection_summary_lines:
                f.write(line + "\n")
            f.write("\n")
