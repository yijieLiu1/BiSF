#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASP: 审计服务器，用于辅助 CSP 在密文层面进行一致性相关计算。

ASP 拥有：
- 正交向量组求和值 V^（标量，原始求和，不做缩放）
- 各 DO 的 n_i
- 系统模数 N
- 每轮随机数 R_t

可构造每个 DO 的 R_t^(N - V^ * n_i)。
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Optional
import math
from utils.hash_utils import hash_global_params

class ASP:
    """审计服务器（ASP）。"""

    def __init__(self, N: int, R_t: int, v_hat: float, n_i: List[int], weight_scale: int, sum_weights: int):
        self.N = int(N)
        self.N2 = int(N) * int(N)
        self.R_t = int(R_t)
        self.v_hat = float(v_hat)
        self.n_i = list(n_i)
        self.weight_scale = int(weight_scale)
        self.sum_weights = int(sum_weights)

    @classmethod
    def from_ta(cls, ta, weight_scale: Optional[int] = None) -> "ASP":
        """从 TA 构建 ASP（读取 N、V^、n_i、R_t）。"""
        N = int(ta.get_N())
        R_t = int(ta.get_R_t())
        if weight_scale is None:
            weight_scale = 100 
        weight_scale = int(weight_scale)
        sum_vec = ta.get_orthogonal_sumvectors_for_csp()
        v_hat = float(sum(sum_vec)) if sum_vec else 0.0
        n_i = list(getattr(ta, "n_i", []))
        sum_weights = sum(int(round(v * weight_scale)) for v in sum_vec) if sum_vec else 0
        return cls(
            N=N,
            R_t=R_t,
            v_hat=v_hat,
            n_i=n_i,
            weight_scale=weight_scale,
            sum_weights=sum_weights,
        )

    def get_v_hat(self) -> float:
        """返回 V^ 的原始求和结果（不缩放）。"""
        return self.v_hat

    def get_n_i(self, do_id: int) -> Optional[int]:
        """返回指定 DO 的 n_i。"""
        if 0 <= do_id < len(self.n_i):
            return int(self.n_i[do_id])
        return None

    def get_N_minus_vhat_ni(self, do_id: int) -> Optional[float]:
        """返回 N - V^ * n_i（保留接口，当前审计不使用该表达式）。"""
        ni = self.get_n_i(do_id)
        if ni is None:
            return None
        return float(self.N - self.v_hat * ni)

    def get_all_N_minus_vhat_ni(self) -> List[Optional[float]]:
        """返回所有 DO 的 N - V^ * n_i 列表。"""
        out: List[Optional[float]] = []
        for i in range(len(self.n_i)):
            out.append(self.get_N_minus_vhat_ni(i))
        return out

    def get_rt_pow_for_do(self, do_id: int, params_hash: Optional[int] = None) -> Optional[int]:
        """返回 R_t^(N - n_i * sum_weights * hash) mod N^2（若未传 hash 则默认 1）。"""
        ni = self.get_n_i(do_id)
        if ni is None:
            return None
        hash_val = int(params_hash) if params_hash is not None else 1
        exp = int(self.N) - int(ni) * int(self.sum_weights) * hash_val
        return pow(self.R_t, exp, self.N2)



def _ciphertext_weighted_sum(cipher_vec: List[int], weights_int: List[int], N2: int) -> int:
    result = 1
    for c, w in zip(cipher_vec, weights_int):
        if w == 0:
            continue
        if w > 0:
            result = (result * pow(c, w, N2)) % N2
        else:
            inv = pow(c, -1, N2)
            result = (result * pow(inv, -w, N2)) % N2
    return result


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _pow_signed(base: int, exp: int, mod: int) -> int:
    if exp == 0:
        return 1
    if exp > 0:
        return pow(base, exp, mod)
    inv = pow(base, -1, mod)
    return pow(inv, -exp, mod)


if __name__ == "__main__":
    try:
        from TA.TA import TA
        from CSP.CSP import CSP
        from DO.DO import DO
    except ModuleNotFoundError:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from TA.TA import TA
        from CSP.CSP import CSP
        from DO.DO import DO

    print("===== [TEST] ASP 一致性审计模拟 =====")
    model_size = 20
    k = 4
    
    ta = TA(num_do=2, model_size=model_size, orthogonal_vector_count=k, bit_length=1024)
    weight_scale = 100
    asp = ASP.from_ta(ta, weight_scale=weight_scale)
    csp = CSP(ta, model_size=model_size, precision=10 ** 6, initial_params_path=None, asp=asp, audit_weight_scale=weight_scale)
    csp.data_shares = ta.get_data_shares()
    global_params = [0.0] * model_size
    do_list = [
        DO(
            i,
            ta,
            model_size=model_size,
            precision=10 ** 6,
            model_name="lenet",
            dataset_name="mnist",
            batch_size=1,
            max_batches=1,
            partition_mode="iid",
        )
        for i in range(ta.num_do)
    ]
    for do in do_list:
        do.update_key(global_params)
    params_hash = hash_global_params(global_params)
    
    do_ids = list(range(ta.num_do))
    v_star = ta.get_orthogonal_sumvectors_for_csp()
    print(f"正交向量组求和 v*: {v_star}, V*的长度: {len(v_star)}")
    weights_int = [int(round(v * weight_scale)) for v in v_star]
    sum_weights = int(sum(weights_int))
    print(f"sum(weights_int)={sum_weights}")
    do_cipher_map = {}
    projections_map = {}
    sum_abs_v = float(sum(abs(v) for v in v_star))
    print(f"sum(|V*|)={sum_abs_v:.6f}")
    for do_id in do_ids:
        share = float(ta.get_data_share_for_do(do_id))
        params = [0.01 * (i + 1) * (1.0 + 0.1 * do_id) for i in range(model_size)]
        max_abs_param = max(abs(v) for v in params)
        sum_wu_raw = _dot(params, v_star)
        sum_wu = sum_wu_raw * share
        projections = []
        for u_vec in ta.get_orthogonal_vectors():
            projections.append(_dot(params, u_vec))
        print(f"\nDO {do_id} params: {params}")
        print(f"DO {do_id} share={share:.3f}")
        print(f"DO {do_id} max|param|={max_abs_param:.6f}")
        print(f"DO {do_id} w·U projections (len={len(projections)}): {projections}")
        print(f"DO {do_id} sum(w·U projections)={sum(projections)}")
        if sum_abs_v > 0.0:
            bound = max_abs_param * weight_scale * sum_abs_v * (10 ** 6)
            print(f"DO {do_id} est_bound={bound:.6e}, y/2≈{ta.impaillier.y/2:.6e}")

        upload_params = [share * v for v in params]
        do = do_list[do_id]
        cipher_vec = [do._encrypt_value(val) for val in upload_params]
        bad = [j for j, c in enumerate(cipher_vec) if math.gcd(c, ta.impaillier.N2) != 1]
        print(f"DO {do_id} bad idx: {bad}")
        rt_pow = asp.get_rt_pow_for_do(do_id, params_hash=params_hash)
        if rt_pow is not None:
            sk = do.round_private_key
            sk_term = _pow_signed(sk, sum_weights, ta.impaillier.N2)
            check = (sk_term * rt_pow) % ta.impaillier.N2
            rt_n = pow(ta.get_R_t(), ta.get_N(), ta.impaillier.N2)
            ni = int(ta.n_i[do_id])
            n2 = ta.impaillier.N2
            rt = ta.get_R_t()
            exp_hash = ni * params_hash
            exp_hash_w = ni * params_hash * sum_weights
            sk_ok = (sk == pow(rt, exp_hash, n2))
            sk_term_ok = (sk_term == pow(rt, exp_hash_w, n2))
            check_ok = (check == pow(rt, ta.get_N(), n2))
            print(
                f"DO {do_id} SK correction check: {check}, R_t^N: {rt_n} | "
                f"sk_ok={sk_ok}, sk_term_ok={sk_term_ok}, check_ok={check_ok}"
            )
        powered_cipher = []
        for c, w in zip(cipher_vec, weights_int):
            if w == 0:
                powered_cipher.append(1)
            elif w > 0:
                powered_cipher.append(pow(c, w, ta.impaillier.N2))
            else:
                inv = pow(c, -1, ta.impaillier.N2)
                powered_cipher.append(pow(inv, -w, ta.impaillier.N2))
        cipher_bits = [c.bit_length() for c in cipher_vec]
        powered_bits = [c.bit_length() for c in powered_cipher]
        print(f"DO {do_id} 原始密文 bitlen: min={min(cipher_bits)}, max={max(cipher_bits)}")
        print(f"DO {do_id} 幂次密文 bitlen: min={min(powered_bits)}, max={max(powered_bits)}")
        combined = 1
        for val in powered_cipher:
            combined = (combined * val) % ta.impaillier.N2
        print(f"DO {do_id} 密文连乘 bitlen: {combined.bit_length()}\n")

        if rt_pow is not None:
            combined = (combined * rt_pow) % ta.impaillier.N2
        decrypted = ta.impaillier.decrypt(combined) / float(weight_scale)
        diff = abs(sum_wu - decrypted)
        print(f"[OK][DO {do_id}] sum_wu={sum_wu:.6f}, decrypted={decrypted:.6f}, diff={diff:.6e}")

        do_cipher_map[do_id] = cipher_vec
        projections_map[do_id] = projections

    csp.do_projection_map = projections_map
    suspects = csp.audit_all_dos_by_cipher(do_cipher_map, tol=1e-3, params_hash=params_hash)
    print(f"\n[OK] CSP 审计结果: {suspects}")

    # 模拟不一致：篡改 DO 0 的密文
    cipher_vec_bad = list(do_cipher_map[0])
    cipher_vec_bad[0] = (cipher_vec_bad[0] * cipher_vec_bad[0]) % ta.impaillier.N2
    do_cipher_map_bad = dict(do_cipher_map)
    do_cipher_map_bad[0] = cipher_vec_bad
    suspects_bad = csp.audit_all_dos_by_cipher(do_cipher_map_bad, tol=1e-3, params_hash=params_hash)
    print(f"[BAD] CSP 审计结果: {suspects_bad}")
