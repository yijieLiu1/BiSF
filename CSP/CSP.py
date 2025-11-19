# CSP.py
import os, sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hashlib
import math
from typing import List, Dict, Optional
from utils.ImprovedPaillier import ImprovedPaillier
from utils.Threshold import recover_secret
from utils.SafeMul import SafeInnerProduct


class CSP:
    """中心服务器（CSP）"""

    def __init__(self, ta, model_size: Optional[int] = None, precision: int = 10 ** 6):
        self.ta = ta
        if model_size is None and hasattr(self.ta, "get_model_size"):
            try:
                self.model_size = self.ta.get_model_size()
            except Exception:
                self.model_size = 50000
        else:
            self.model_size = model_size if model_size is not None else 50000
        self.precision = precision

        # 全局模型参数管理
        self.global_params: List[float] = [0.0] * self.model_size
        self.global_params_snapshot: List[float] = list(self.global_params)
        self.round_count: int = 0

        # 为掉线恢复缓存
        # 恢复的是掉线DO的n_i值
        self.recoveredNiValues: Dict[int, int] = {}

        # 保存TA生成的正交向量组
        self.orthogonal_vectors_for_csp: List[List[float]] = []
        self.orthogonal_sumvectors_for_csp: List[float] = []
        # 保存每个 DO 的 w·U 映射结果（1×orthogonal_vector_count）
        self.do_projection_map: Dict[int, List[float]] = {}
        self.debug_last_sum: Optional[List[float]] = None
        self._load_orthogonal_vectors()

        # 使用 ImprovedPaillier，但用 TA 的参数覆盖，确保全局一致
        self.impaillier = ImprovedPaillier(m=self.ta.num_do, bit_length=512, precision=self.precision)
        self._sync_paillier_with_ta()

    def _load_orthogonal_vectors(self) -> None:
        self.orthogonal_vectors_for_csp = self.ta.get_orthogonal_vectors_for_csp()
        self.orthogonal_sumvectors_for_csp = self.ta.get_orthogonal_sumvectors_for_csp()
        print(f"[CSP] 已加载正交求和向量，共{len(self.orthogonal_sumvectors_for_csp)}维")
        print(f"[CSP] 已加载正交向量组，共{len(self.orthogonal_vectors_for_csp)}个向量")

    def _sync_paillier_with_ta(self) -> None:
        """同步公参，包括 y 参数"""
        N = self.ta.get_N()
        self.impaillier.N = N
        self.impaillier.N2 = N * N
        self.impaillier.g = self.ta.get_g()
        self.impaillier.h = self.ta.get_h()
        
        # 同步 lambda_ 和 u 参数
        try:
            self.impaillier.lambda_ = self.ta.get_lambda()
        except Exception:
            self.impaillier.lambda_ = getattr(self.ta, 'lambda_val', None)
        
        if hasattr(self.ta, 'u'):
            self.impaillier.u = self.ta.u
        elif hasattr(self.ta, 'mu'):
            self.impaillier.u = self.ta.mu
            
        # **重要：同步 y 参数，确保解密一致性**
        try:
            self.impaillier.y = self.ta.get_y()
        except Exception:
            self.impaillier.y = getattr(self.ta, 'gamma', None)
            
        # print(f"[CSP] 已同步Paillier公参，N={N}, g={self.impaillier.g}, h={self.impaillier.h}, y={self.impaillier.y}")
    # ============== 广播 ==============
    def broadcast_params(self) -> List[float]:
        try:
            self.ta.update_keys_for_new_round()
        except Exception as e:
            print(f"[CSP] 密钥更新失败: {e}")
        # 同步本轮最新的正交向量组
        self._load_orthogonal_vectors()
        self.global_params_snapshot = list(self.global_params)
        self.round_count += 1
        # print(f"[CSP] 第{self.round_count}轮广播参数: {self.global_params_snapshot}")
        return list(self.global_params_snapshot)

    # ============== 收集/聚合 ==============
    def aggregate_ciphertexts(self, do_cipher_map: Dict[int, List[int]]) -> List[int]:
        """逐坐标同态聚合：使用ImprovedPaillier的聚合方法"""
        aggregated: List[int] = []

        for i in range(self.model_size):
            coordinate_ciphertexts = []
            for do_id, c_vec in do_cipher_map.items():
                if c_vec is not None and i < len(c_vec):
                    coordinate_ciphertexts.append(c_vec[i])

            if coordinate_ciphertexts:
                aggregated_cipher = self.impaillier.aggregate(coordinate_ciphertexts)
                aggregated.append(aggregated_cipher)
            else:
                # 加密0作为占位（使用 SK_DO = 1 即不会影响乘积）
                zero_cipher = self.impaillier.encrypt(0.0, 1)
                aggregated.append(zero_cipher)

        print(f"[CSP] 聚合完成，共{len(aggregated)}个坐标")
        return aggregated

    # ============== 掉线恢复 ==============
    # 基于门限恢复缺失 n_i（从在线 DO 收集分片并重构）
    def recover_missing_private_keys(self, missing_ids: List[int], online_dos: List, threshold: int) -> Dict[int, int]:
        self.recoveredNiValues.clear()
        print(f"[CSP] 检测到掉线 DO: {missing_ids}，在线DO: {[d.id for d in online_dos]}")

        for missing_id in missing_ids:
            try:
                shares_map: Dict[int, int] = {}
                prime_used: Optional[int] = None

                for do in online_dos:
                    info = None
                    try:
                        info = do.get_key_share_info(missing_id)
                    except Exception:
                        info = None

                    if info is None:
                        continue

                    share_val = info.get('share')
                    prime = info.get('prime')
                    do_id = info.get('do_id')

                    if share_val is None or prime is None or do_id is None:
                        continue

                    if prime_used is None:
                        prime_used = prime
                    elif prime_used != prime:
                        print(f"[CSP] 分片素数不一致，忽略DO {do_id} 的分片")
                        continue

                    # 注意 shares 的 x 取在线 DO 的编号 + 1（与生成时一致）
                    shares_map[do_id + 1] = share_val

                    if len(shares_map) >= threshold:
                        break

                if len(shares_map) < threshold or prime_used is None:
                    print(f"[CSP] 恢复DO {missing_id} 分片不足：{len(shares_map)}/{threshold}")
                    continue

                recovered_n = recover_secret(shares_map, prime_used)
                self.recoveredNiValues[missing_id] = recovered_n
                print(f"[CSP] 成功恢复DO {missing_id} 的 n_i{recovered_n}")

            except Exception as e:
                print(f"[CSP] 恢复DO {missing_id} 时出错: {e}")

        return self.recoveredNiValues

    # ============== 解密流程（正常/带恢复） ==============
    def _decrypt_vector(self, aggregated: List[int]) -> List[float]:
        results: List[float] = [0.0] * self.model_size
        for i in range(self.model_size):
            results[i] = self.impaillier.decrypt(aggregated[i])
        print(f"[CSP] 正常解密结果维度: {len(results)}")
        print(f"[CSP] 解密结果前10维: {results[:10]}")
        return results

    def _decrypt_with_recovery(self, aggregated: List[int], params_hash: int) -> List[float]:
       
        """恢复缺失 n_i 后解密（保持原逻辑）"""
        N2 = self.impaillier.N2
        N = self.impaillier.N

        # 求和 n_i（由 TA 恢复并存入 self.recoveredNiValues）
        sumNi = sum(self.recoveredNiValues.values())
        sumNi = (sumNi * params_hash) % N  # 避免溢出

        R_t = self.ta.get_R_t()
        Rt_pow = pow(R_t, sumNi, N2)

        results: List[float] = [0.0] * self.model_size
        for i in range(self.model_size):
            modified = (aggregated[i] * Rt_pow) % N2
            results[i] = self.impaillier.decrypt(modified)
            # 对于大向量，只打印前几个坐标的详细信息
            if i < 5:
                print(f"[CSP] 恢复解密坐标 {i}: 原始密文={aggregated[i]}, 修正密文={modified} -> 明文={results[i]}")
        print(f"[CSP] 恢复解密完成，结果维度: {len(results)}，前10维: {results[:10]}")
        return results

    # ============== 主流程 ==============
    def round_aggregate_and_update(self, do_list: List[Optional[object]], do_cipher_map: Dict[int, List[int]]) -> List[
        float]:
        print(f"[CSP] 开始第{self.round_count}轮聚合和更新")
        aggregated = self.aggregate_ciphertexts(do_cipher_map)

        online_dos = [d for d in do_list if d is not None]
        missing_ids = [idx for idx, d in enumerate(do_list) if d is None]
        print(f"[CSP] 在线DO: {[d.id for d in online_dos]}, 掉线DO: {missing_ids}")
        # 计算当前全局参数哈希，如果有DO掉线了，需要使用
        # 对于大向量，使用采样策略提高效率
        if len(self.global_params_snapshot) > 10000:
            sample_size = 3000
            step = len(self.global_params_snapshot) // sample_size
            sampled = self.global_params_snapshot[::max(1, step)][:sample_size]
            sampled = self.global_params_snapshot[:1000] + self.global_params_snapshot[len(self.global_params_snapshot)//2:len(self.global_params_snapshot)//2+1000] + self.global_params_snapshot[-1000:]
            params_bytes = str(sampled).encode('utf-8')
        else:
            params_bytes = str(self.global_params_snapshot).encode('utf-8')
        params_hash = int.from_bytes(hashlib.sha256(params_bytes).digest(), 'big', signed=False)

        if not missing_ids:
            summed = self._decrypt_vector(aggregated)
        else:
            print(f"\n[CSP] 检测到掉线DO，进行密钥恢复\n")
            self.recover_missing_private_keys(missing_ids, online_dos, self.ta.get_threshold())
            summed = self._decrypt_with_recovery(aggregated, params_hash)

        # 聚合后进行一致性对比
       
        self.debug_last_sum = list(summed)
        print(f"[CSP] 计算正交求和向量点积结果{self.compute_sum_with_orthogonal_vector(summed)}")
        print(f"[CSP] 判断正交求和是否一致{self.compare_consistency(summed)}")
        print(f"[CSP] 解密后的聚合结果维度: {len(summed)}")
        print(f"[CSP] 聚合结果前10维: {summed[:10]}")
        print(f"[CSP] 聚合结果范围: [{min(summed):.4f}, {max(summed):.4f}]")

        num_online = max(1, len(online_dos))
        # DO 侧上传的是训练后的完整参数（几万维），
        # 因此这里直接取在线 DO 的平均作为新一轮全局参数。
        next_params = [ (summed[i] / num_online) for i in range(self.model_size) ]

        self.global_params = next_params
        print(f"[CSP] 更新后的全局参数维度: {len(next_params)}")
        print(f"[CSP] 更新后的全局参数前10维: {next_params[:10]}")
        print(f"[CSP] 更新后的全局参数范围: [{min(next_params):.4f}, {max(next_params):.4f}]")

        return next_params

    # ============== 正交求和向量相关 ==============
    def compute_sum_with_orthogonal_vector(self, params: List[float]) -> float:
        """
        将聚合解密后的 1×n 模型参数与 orthogonal_sumvectors_for_csp 做点积，得到单个标量。
        """
        if not self.orthogonal_sumvectors_for_csp:
            raise ValueError("[CSP] 未加载 orthogonal_sumvectors_for_csp")
        if len(params) != len(self.orthogonal_sumvectors_for_csp):
            raise ValueError(f"[CSP] 参数长度不匹配: params={len(params)}, sum_vec={len(self.orthogonal_sumvectors_for_csp)}")
        return float(sum(p * s for p, s in zip(params, self.orthogonal_sumvectors_for_csp)))

    def compare_consistency(self, params: List[float], tol: float = 1e-2) -> bool:
        """
        将 w·U 得到的 1×m 向量求和，与 compute_sum_with_orthogonal_vector(params) 的结果对比。
        """
        if not self.do_projection_map:
            print("[CSP] 警告：尚未记录 DO 的正交映射结果，跳过一致性对比。")
            return False

        sum_wu = float(sum(sum(vec) for vec in self.do_projection_map.values()))
        sum_proj = self.compute_sum_with_orthogonal_vector(params)
        print(f"[CSP] w·U 向量求和结果: {sum_wu}")
        print(f"[CSP] 正交求和向量点积结果: {sum_proj}")
        is_consistent = abs(sum_wu - sum_proj) <= tol
        print(f"[CSP] 一致性判断: {is_consistent}")
        return is_consistent

    # ============== 投毒检测：Multi-Krum 风格 ==============
    def detect_poison_multi_krum(self, f: int = 1, alpha: float = 1.5) -> List[int]:
        """
        基于当前 do_projection_map（每个 DO 的 w·U 投影向量）执行 Multi-Krum 异常检测。
        Args:
            f: 可容忍的恶意客户端个数估计（影响 K = N - f - 2）
            alpha: IQR 放大系数，用于判定“远离大多数”的阈值
        Returns:
            suspects: 被判定为可疑的 DO id 列表
        """
        if not self.do_projection_map:
            print("[CSP] 投毒检测跳过：无投影数据。")
            return []

        ids = sorted(self.do_projection_map.keys())
        vectors = [self.do_projection_map[i] for i in ids]
        n = len(vectors)
        if n < 2:
            print("[CSP] 投毒检测跳过：在线 DO 少于 2 个。")
            return []

        # 计算 pairwise 距离矩阵（对称，平方欧氏距离）
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            vi = vectors[i]
            for j in range(i + 1, n):
                vj = vectors[j]
                s = 0.0
                for a, b in zip(vi, vj):
                    d = a - b
                    s += d * d
                dist[i][j] = dist[j][i] = s

        K = n - f - 2
        if K < 1:
            print(f"[CSP] 投毒检测跳过：K={K} 无意义（n={n}, f={f}）。")
            return []

        # 计算 Krum score
        scores: List[float] = []
        for i in range(n):
            row = [dist[i][j] for j in range(n) if j != i]
            row.sort()
            scores.append(sum(row[:K]))

        # 中位数和 IQR
        def _median(vals: List[float]) -> float:
            vals_sorted = sorted(vals)
            m = len(vals_sorted)
            mid = m // 2
            if m % 2 == 1:
                return vals_sorted[mid]
            return 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])

        def _iqr(vals: List[float]) -> float:
            if len(vals) < 4:
                return 0.0
            vals_sorted = sorted(vals)
            m = len(vals_sorted)
            mid = m // 2
            lower = vals_sorted[:mid]
            upper = vals_sorted[mid + (0 if m % 2 == 0 else 1):]
            if not lower or not upper:
                return 0.0
            return _median(upper) - _median(lower)

        med = _median(scores)
        iqr = _iqr(scores)
        threshold = med + alpha * iqr
        suspects_idx = [idx for idx, s in enumerate(scores) if s > threshold]
        suspects = [ids[idx] for idx in suspects_idx]

        print(f"[CSP] Multi-Krum 检测：scores={scores}, med={med:.6f}, iqr={iqr:.6f}, 阈值={threshold:.6f}")
        if suspects:
            print(f"[CSP] 可疑 DO: {suspects}")
        else:
            print("[CSP] 未发现可疑 DO（根据当前阈值）。")
        return suspects

    def detect_poison_geomedian(self, beta: float = 1.5, max_iter: int = 50, eps: float = 1e-9) -> List[int]:
        """
        基于几何中位数的异常检测：计算鲁棒中心后，按中位数+IQR 标记远离中心的 DO。
        Args:
            beta: IQR 放大系数
            max_iter: Weiszfeld 迭代次数上限
            eps: 距离下界，避免除零
        Returns:
            suspects: 被判定为可疑的 DO id 列表
        """
        if not self.do_projection_map:
            print("[CSP] GeoMedian 检测跳过：无投影数据。")
            return []

        ids = sorted(self.do_projection_map.keys())
        vectors = [self.do_projection_map[i] for i in ids]
        n = len(vectors)
        if n < 2:
            print("[CSP] GeoMedian 检测跳过：在线 DO 少于 2 个。")
            return []

        dim = len(vectors[0])
        # 初始中心：均值
        center = [0.0] * dim
        for v in vectors:
            for i, val in enumerate(v):
                center[i] += val
        center = [c / n for c in center]

        def l2_norm_sq(a, b):
            s = 0.0
            for x, y in zip(a, b):
                d = x - y
                s += d * d
            return s

        # Weiszfeld 迭代
        for _ in range(max_iter):
            weights = []
            for v in vectors:
                dist = max(l2_norm_sq(v, center) ** 0.5, eps)
                weights.append(1.0 / dist)
            new_center = [0.0] * dim
            weight_sum = sum(weights)
            for v, w in zip(vectors, weights):
                for i, val in enumerate(v):
                    new_center[i] += w * val
            new_center = [c / weight_sum for c in new_center]
            # 收敛判定
            shift = l2_norm_sq(center, new_center) ** 0.5
            center = new_center
            if shift < 1e-6:
                break

        # 计算到中心的距离
        dists = [l2_norm_sq(v, center) ** 0.5 for v in vectors]

        def _median(vals: List[float]) -> float:
            vals_sorted = sorted(vals)
            m = len(vals_sorted)
            mid = m // 2
            if m % 2 == 1:
                return vals_sorted[mid]
            return 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])

        def _iqr(vals: List[float]) -> float:
            if len(vals) < 4:
                return 0.0
            vals_sorted = sorted(vals)
            m = len(vals_sorted)
            mid = m // 2
            lower = vals_sorted[:mid]
            upper = vals_sorted[mid + (0 if m % 2 == 0 else 1):]
            if not lower or not upper:
                return 0.0
            return _median(upper) - _median(lower)

        med = _median(dists)
        iqr = _iqr(dists)
        threshold = med + beta * iqr
        suspects_idx = [idx for idx, d in enumerate(dists) if d > threshold]
        suspects = [ids[idx] for idx in suspects_idx]

        print(f"[CSP] GeoMedian 检测：dists={dists}, med={med:.6f}, iqr={iqr:.6f}, 阈值={threshold:.6f}")
        if suspects:
            print(f"[CSP] GeoMedian 可疑 DO: {suspects}")
        else:
            print("[CSP] GeoMedian 未发现可疑 DO（根据当前阈值）。")
        return suspects

    # ============== SafeMul: 1+3轮（PA侧） ==============
    def safe_mul_prepare_payload(self) -> Dict[str, object]:
        """准备安全点积第1轮数据，基于 CSP 的正交向量组发送 (p, alpha, C_all)。"""
        sip = SafeInnerProduct(precision_factor=self.precision)
        p, alpha, C_all, s, s_inv = sip.round1_setup_and_encrypt(self.orthogonal_vectors_for_csp)
        return {'p': p, 'alpha': alpha, 'C_all': C_all, 's_inv': s_inv}

    def safe_mul_finalize(self, ctx: Dict[str, object], D_sums: List[int], do_part: List[float], do_id: int) -> List[float]:
        """执行安全点积第3轮并与 DO 明文部分求和，得到 w·U 的 1×m 向量，并缓存该 DO 的映射结果。"""
        sip = SafeInnerProduct(precision_factor=self.precision)
        p = ctx['p']
        alpha = ctx['alpha']
        s_inv = ctx['s_inv']
        csp_part = sip.round3_decrypt(D_sums, s_inv, alpha, p)
        projection = [csp_part[i] + do_part[i] for i in range(len(csp_part))]
        self.do_projection_map[do_id] = projection
        return projection
# ===========================
# === 测试代码部分 (CSP.py) ===
# ===========================
if __name__ == "__main__":
    print("===== [TEST] 启动 CSP 测试 =====")
    from TA.TA import TA
    from DO.DO import DO

    def plain_sum_params(dos: List[Optional[object]]) -> List[float]:
        active = [d for d in dos if d is not None]
        if not active:
            return []
        total = [0.0] * len(active[0].get_last_updates())
        for do in active:
            w = do.get_last_updates()
            for i, v in enumerate(w):
                total[i] += v
        return total

    def diff_stats(a: List[float], b: List[float]) -> (Optional[float], Optional[float]):
        if not a or not b:
            return None, None
        max_abs = 0.0
        sq = 0.0
        for x, y in zip(a, b):
            d = x - y
            ad = abs(d)
            if ad > max_abs:
                max_abs = ad
            sq += d * d
        return max_abs, math.sqrt(sq)

    def projection_scalar_sum(projection_map: Dict[int, List[float]]) -> Optional[float]:
        if not projection_map:
            return None
        return float(sum(sum(vec) for vec in projection_map.values()))

    ta = TA(num_do=3, model_size=10000, orthogonal_vector_count=1024, bit_length=512)
    csp = CSP(ta)
    do_list = [DO(i, ta) for i in range(3)]

    print("\n===== Round 1: 广播参数 =====")
    global_params = csp.broadcast_params()

    print("\n===== Round 1: 收集各 DO 的密文更新 =====")
    do_cipher_map = {}
    for do in do_list:
        ciphertexts = do.train_and_encrypt(global_params)
        do_cipher_map[do.id] = ciphertexts

    print("\n===== Round 1: SafeMul 投影计算（在线 DO）=====")
    csp.do_projection_map.clear()
    ctx = csp.safe_mul_prepare_payload()
    for do in [d for d in do_list if d is not None]:
        b_vec = do.get_last_updates()
        payload = {'p': ctx['p'], 'alpha': ctx['alpha'], 'C_all': ctx['C_all']}
        resp = do.safe_mul_round2_process(payload, b_vec)
        projection = csp.safe_mul_finalize(ctx, resp['D_sums'], resp['do_part'], do.id)
        print(f" DO {do.id} 投影向量(长度{len(projection)}): {projection}")
    plain_sum_round1 = plain_sum_params(do_list)
    if plain_sum_round1:
        print(f"[DEBUG R1] 明文聚合参数前5: {plain_sum_round1[:5]}")
        plain_dot_round1 = csp.compute_sum_with_orthogonal_vector(plain_sum_round1)
        print(f"[DEBUG R1] 明文聚合与正交求和向量点积: {plain_dot_round1}")

    new_params = csp.round_aggregate_and_update(do_list, do_cipher_map)
    if csp.debug_last_sum:
        max_abs_diff, l2_diff = diff_stats(plain_sum_round1, csp.debug_last_sum)
        if max_abs_diff is not None:
            print(f"[DEBUG R1] 明文聚合 vs 解密聚合差异: max_abs={max_abs_diff:.6f}, l2={l2_diff:.6f}")
            print(f"[DEBUG R1] 解密聚合前5: {csp.debug_last_sum[:5]}")
            decrypt_dot_round1 = csp.compute_sum_with_orthogonal_vector(csp.debug_last_sum)
            print(f"[DEBUG R1] 解密聚合与正交求和向量点积: {decrypt_dot_round1}")
    safe_scalar_round1 = projection_scalar_sum(csp.do_projection_map)
    if safe_scalar_round1 is not None:
        print(f"[DEBUG R1] SafeMul 投影标量求和: {safe_scalar_round1}")

    # print("\n===== Round 2: 模拟 DO2 掉线并恢复 =====")
    # global_params = csp.broadcast_params()
    # do_list[2] = None  # 模拟掉线

    # do_cipher_map = {}
    # for do in [d for d in do_list if d is not None]:
    #     ciphertexts = do.train_and_encrypt(global_params)
    #     do_cipher_map[do.id] = ciphertexts

    # print("\n===== Round 2: SafeMul 投影计算（在线 DO）====")
    # csp.do_projection_map.clear()
    # ctx = csp.safe_mul_prepare_payload()
    # for do in [d for d in do_list if d is not None]:
    #     b_vec = do.get_last_updates()
    #     payload = {'p': ctx['p'], 'alpha': ctx['alpha'], 'C_all': ctx['C_all']}
    #     resp = do.safe_mul_round2_process(payload, b_vec)
    #     projection = csp.safe_mul_finalize(ctx, resp['D_sums'], resp['do_part'], do.id)
    #     print(f" DO {do.id} 投影向量(长度{len(projection)})")

    # plain_sum_round2 = plain_sum_params(do_list)
    # if plain_sum_round2:
    #     print(f"[DEBUG R2] 明文聚合参数前5: {plain_sum_round2[:5]}")
    #     plain_dot_round2 = csp.compute_sum_with_orthogonal_vector(plain_sum_round2)
    #     print(f"[DEBUG R2] 明文聚合与正交求和向量点积: {plain_dot_round2}")

    # new_params = csp.round_aggregate_and_update(do_list, do_cipher_map)
    # if csp.debug_last_sum:
    #     max_abs_diff2, l2_diff2 = diff_stats(plain_sum_round2, csp.debug_last_sum)
    #     if max_abs_diff2 is not None:
    #         print(f"[DEBUG R2] 明文聚合 vs 解密聚合差异: max_abs={max_abs_diff2:.6f}, l2={l2_diff2:.6f}")
    #         print(f"[DEBUG R2] 解密聚合前5: {csp.debug_last_sum[:5]}")
    #         decrypt_dot_round2 = csp.compute_sum_with_orthogonal_vector(csp.debug_last_sum)
    #         print(f"[DEBUG R2] 解密聚合与正交求和向量点积: {decrypt_dot_round2}")
    # safe_scalar_round2 = projection_scalar_sum(csp.do_projection_map)
    # if safe_scalar_round2 is not None:
    #     print(f"[DEBUG R2] SafeMul 投影标量求和: {safe_scalar_round2}")

    print("\n===== 测试完成 =====")
  
