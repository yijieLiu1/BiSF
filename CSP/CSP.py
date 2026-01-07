# CSP.py
import os, sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hashlib
import math
import random
from typing import List, Dict, Optional
from utils.ImprovedPaillier import ImprovedPaillier
from utils.Threshold import recover_secret
from utils.SafeMul import SafeInnerProduct


class CSP:
    """中心服务器（CSP）"""

    def __init__(self, ta, model_size: Optional[int] = None, precision: int = 10 ** 6, initial_params_path: Optional[str] = None):
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
        if initial_params_path:
            self._try_load_initial_params(initial_params_path)

        # 为掉线恢复缓存
        # 恢复的是掉线DO的n_i值
        self.recoveredSecretValues: Dict[int, int] = {}

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

    def _try_load_initial_params(self, path: str) -> None:
        """可选地从文件加载初始全局参数"""
        try:
            import json
            if not os.path.exists(path):
                print(f"[CSP] 初始参数文件不存在，忽略: {path}")
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = data.get("params") if isinstance(data, dict) else None
            if params is None:
                if isinstance(data, list):
                    params = data
                else:
                    print(f"[CSP] 初始参数文件格式不含 params，忽略: {path}")
                    return
            if len(params) < self.model_size:
                params = list(params) + [0.0] * (self.model_size - len(params))
                print(f"[CSP] 初始参数长度不足，已用0填充到 {self.model_size}")
            elif len(params) > self.model_size:
                params = params[:self.model_size]
                print(f"[CSP] 初始参数长度超过模型尺寸，已截断到 {self.model_size}")
            self.global_params = list(map(float, params))
            self.global_params_snapshot = list(self.global_params)
            print(f"[CSP] 已从文件加载初始全局参数: {path}，维度 {len(self.global_params)}")
        except Exception as e:
            print(f"[CSP] 加载初始参数失败（忽略，使用默认0向量）: {e}")

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
        # 第 1 轮沿用初始化时的密钥/正交向量；从第 2 轮开始每轮刷新
        if self.round_count > 0:
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
        time_start = time.time()
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
        time_end = time.time()
        print(f"[CSP] 聚合完成，共{len(aggregated)}个坐标, 用时{time_end - time_start:.4f}秒")
        return aggregated

    # ============== 掉线恢复 ==============
    # 基于门限恢复缺失 n_i（从在线 DO 收集分片并重构）
    def recover_missing_private_keys(self, missing_ids: List[int], online_dos: List, threshold: int) -> Dict[int, int]:
        self.recoveredSecretValues.clear()
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

                recovered_key = recover_secret(shares_map, prime_used)
                self.recoveredSecretValues[missing_id] = recovered_key
                print(f"[CSP] 成功恢复DO {missing_id} 的私钥")

            except Exception as e:
                print(f"[CSP] 恢复DO {missing_id} 时出错: {e}")

        return self.recoveredSecretValues

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

        # 连乘恢复的私钥（每个为 R_t^{n_i} mod N^2），得到 R_t^{sum n_i}
        Rt_pow = 1
        for secret in self.recoveredSecretValues.values():
            Rt_pow = (Rt_pow * secret) % N2
        Rt_pow = pow(Rt_pow, params_hash, N2)
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
        consistency = self.compare_consistency(summed)
        print(f"[CSP] 判断正交求和是否一致{consistency}")
        if not consistency:
            suspects = self.locate_inconsistent_dos(do_list, do_cipher_map, params_hash)
            if suspects:
                print(f"[CSP] 二分定位可疑 DO: {suspects}")
            else:
                print("[CSP] 未能定位具体可疑 DO")
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

    def _subset_projection_sum(self, subset_ids: List[int]) -> Optional[float]:
        if not self.do_projection_map:
            return None
        total = 0.0
        for do_id in subset_ids:
            vec = self.do_projection_map.get(do_id)
            if vec:
                total += float(sum(vec))
        return total

    def _decrypt_vector_quiet(self, cipher_vec: List[int]) -> List[float]:
        return [self.impaillier.decrypt(val) for val in cipher_vec]

    def _subset_consistency(self, subset_ids: List[int], do_cipher_map: Dict[int, List[int]], params_hash: int,
                             tol: float = 1e-2) -> bool:
        if not subset_ids:
            return True
        subset_map = {do_id: do_cipher_map[do_id] for do_id in subset_ids if do_id in do_cipher_map}
        if not subset_map:
            return True

        aggregated = self.aggregate_ciphertexts(subset_map)
        complement_ids = [do_id for do_id in do_cipher_map.keys() if do_id not in subset_ids]
        comp_key = self.ta.get_aggregated_base_key(complement_ids) if complement_ids else 1
        if comp_key in (None, 0):
            Rt_pow = 1
        else:
            Rt_pow = pow(comp_key, params_hash, self.impaillier.N2)
        modified = [(val * Rt_pow) % self.impaillier.N2 for val in aggregated]
        summed = self._decrypt_vector_quiet(modified)
        subset_proj = self._subset_projection_sum(subset_ids)
        if subset_proj is None:
            return True
        sum_proj = self.compute_sum_with_orthogonal_vector(summed)
        is_consistent = abs(subset_proj - sum_proj) <= tol
        if not is_consistent:
            print(f"[CSP] 子集 {subset_ids} 不一致：w·U={subset_proj}, 解密投影={sum_proj}")
        return is_consistent

    def _bisect_inconsistency(self, do_ids: List[int], do_cipher_map: Dict[int, List[int]], params_hash: int,
                              tol: float = 1e-2) -> List[int]:
        if not do_ids:
            return []
        if len(do_ids) == 1:
            return do_ids
        mid = len(do_ids) // 2
        left = do_ids[:mid]
        right = do_ids[mid:]
        if not self._subset_consistency(left, do_cipher_map, params_hash, tol):
            return self._bisect_inconsistency(left, do_cipher_map, params_hash, tol)
        if not self._subset_consistency(right, do_cipher_map, params_hash, tol):
            return self._bisect_inconsistency(right, do_cipher_map, params_hash, tol)
        return []

    def locate_inconsistent_dos(self, do_list: List[Optional[object]], do_cipher_map: Dict[int, List[int]],
                                params_hash: int, tol: float = 1e-2) -> List[int]:
        print(f"\n[CSP] 开始二分定位不一致 DO")
        active_ids = [d.id for d in do_list if d is not None]
        if len(active_ids) <= 1 or not self.do_projection_map:
            return active_ids
        suspects = self._bisect_inconsistency(active_ids, do_cipher_map, params_hash, tol)
        return suspects

    # ============== 投毒检测：Multi-Krum 每个 DO 与其他 DO 的平方欧氏距离矩阵 =============
    # geometric median 基于几何中位数的异常检测 =============
    # clustering 聚类式防御（余弦相似度版）
    # lasa liter 轻量级异常检测 余弦方向和模长双重检查=============
    #  ==============
    def detect_poison_multi_krum(self, f: int = 1, alpha: float = 1.5) -> List[int]:
        """
        基于当前 do_projection_map（每个 DO 的 w·U 投影向量）执行 Multi-Krum 异常检测。
        Args:
            k：对每个候选向量，只取它最近的 K 个邻居来计算分数
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

        # 计算 pairwise 距离矩阵（对称，平方欧氏距离），计算n*n的矩阵
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

        # 计算 Krum score，找到除自己外的其余参数，取最小的k个距离求和
        scores: List[float] = []
        for i in range(n):
            row = [dist[i][j] for j in range(n) if j != i]
            row.sort()
            scores.append(sum(row[:K]))

        # 中位数和 IQR：这里用“上半部分的中位数 - 下半部分的中位数”，最终算出一个阈值
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
            print(f"[CSP] Multi-Krum 可疑 DO: {suspects}")
        else:
            print("[CSP] 未发现可疑 DO（根据当前阈值）。")
        return suspects

    def detect_poison_geomedian(self, beta: float = 1.5, max_iter: int = 50, eps: float = 1e-9) -> List[int]:
        """
        基于几何中位数的异常检测：计算鲁棒中心后，按中位数+IQR 标记远离中心的 DO。
        基于所有的参数，计算得到全局的几何的中位数【离所有的点最近】，然后基于这个全局的中位数，进行距离比较
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
    
    def detect_poison_clustering(self, k: int = 2, max_iter: int = 10, alpha: float = 1.5) -> List[int]:
        """
        简易聚类式防御（余弦相似度版）：使用 spherical k-means（cosine distance）将投影向量聚类，丢弃离群簇。
        - assignment: 最小化 cosine distance = 1 - cos
        - center update: 簇内单位向量求和后再归一化（方向中心）

        修改点（不改变输入输出）：
        1) 初始化中心：由“取前 k 个样本” -> 余弦距离版 k-means++（更稳定）
        2) 可选少量重启（restarts），选择簇内总余弦距离最小的结果（仍保持输出结构一致）
        3) 复用 counts 计算 size，减少重复遍历
        """
        if not self.do_projection_map or k < 1:
            print("[CSP] Cluster 检测跳过：无投影数据或 k 无效。")
            return []

        ids = sorted(self.do_projection_map.keys())
        raw_vectors = [self.do_projection_map[i] for i in ids]
        n = len(raw_vectors)
        if n <= 1:
            return []
        dim = len(raw_vectors[0])
        k = min(k, n)

        def _norm(v: List[float]) -> float:
            return math.sqrt(sum(x * x for x in v))

        def _normalize(v: List[float]) -> List[float]:
            nv = _norm(v)
            if nv == 0.0:
                return [0.0] * len(v)
            inv = 1.0 / nv
            return [x * inv for x in v]

        def _dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def cosine_dist_unit(a_unit: List[float], b_unit: List[float]) -> float:
            # a_unit, b_unit 都应已归一化；cos ∈ [-1, 1]
            return 1.0 - _dot(a_unit, b_unit)

        # 1) 预归一化所有向量（球面 k-means 的基本步骤）
        vectors = [_normalize(v) for v in raw_vectors]

        # 若全是零向量（归一化后仍全 0），余弦距离失效，直接退出（保持安全）
        if all(_norm(v) == 0.0 for v in vectors):
            print("[CSP] Cluster 检测跳过：所有向量为零，余弦聚类无意义。")
            return []

        # -------------------------
        # 2) 余弦版 k-means++ 初始化
        # -------------------------
        def _kmeanspp_init(rng: random.Random) -> List[List[float]]:
            # 选第一个中心：从非零向量里随机选
            nonzero_indices = [i for i, v in enumerate(vectors) if _norm(v) > 0.0]
            if not nonzero_indices:
                # 理论上已被上面的 all-zero 拦住，这里兜底
                return [vectors[0][:] for _ in range(k)]

            first = rng.choice(nonzero_indices)
            centers = [vectors[first][:]]

            # 维护每个点到“最近中心”的距离（余弦距离）
            closest = [cosine_dist_unit(vectors[i], centers[0]) for i in range(n)]

            while len(centers) < k:
                # 概率 ∝ dist^2（经典 k-means++；这里 dist 是余弦距离）
                weights = []
                total = 0.0
                for i in range(n):
                    d = closest[i]
                    w = d * d
                    weights.append(w)
                    total += w

                if total <= 0.0:
                    # 所有点到最近中心距离都为 0：说明基本同向，剩余中心随便补齐
                    for i in range(n):
                        if len(centers) >= k:
                            break
                        centers.append(vectors[i][:])
                    break

                # 轮盘赌采样
                r = rng.random() * total
                acc = 0.0
                chosen = 0
                for i, w in enumerate(weights):
                    acc += w
                    if acc >= r:
                        chosen = i
                        break

                centers.append(vectors[chosen][:])

                # 更新 closest
                new_c = centers[-1]
                for i in range(n):
                    d = cosine_dist_unit(vectors[i], new_c)
                    if d < closest[i]:
                        closest[i] = d

            return centers

        # -------------------------
        # 3) spherical k-means 主循环（单次运行）
        # -------------------------
        def _run_spherical_kmeans(init_centers: List[List[float]]):
            centers = [c[:] for c in init_centers]
            assignments = [-1] * n

            for _ in range(max_iter):
                changed = False

                # assignment
                for idx, vec in enumerate(vectors):
                    # 跳过零向量：它与任何中心的 dot 都是 0，cos dist = 1，分配结果不稳定
                    # 这里采取“照常参与”，但不会让其主导中心（update 时会被加进去）
                    dists = [cosine_dist_unit(vec, c) for c in centers]
                    new_k = min(range(k), key=lambda i: dists[i])
                    if assignments[idx] != new_k:
                        assignments[idx] = new_k
                        changed = True

                if not changed:
                    break

                # update
                counts = [0] * k
                sums = [[0.0] * dim for _ in range(k)]
                for assign, vec in zip(assignments, vectors):
                    if assign < 0:
                        continue
                    counts[assign] += 1
                    for j in range(dim):
                        sums[assign][j] += vec[j]

                for i in range(k):
                    if counts[i] == 0:
                        # fallback：随机挑一个点当中心（比 vectors[i % n] 更不依赖顺序）
                        # 这里不额外引入 rng，保持单次运行内简单；外层重启会提供多样性
                        centers[i] = vectors[i % n][:]
                    else:
                        centers[i] = _normalize(sums[i])

            # 计算簇内总损失（用于重启择优）：sum_i min_c cosine_dist(x_i, c)
            inertia = 0.0
            for idx, vec in enumerate(vectors):
                c = centers[assignments[idx]]
                inertia += cosine_dist_unit(vec, c)

            return centers, assignments, inertia
        # -------------------------
        # 4) 少量重启：择优（不改变对外接口）
        # -------------------------
        # 重启次数取一个小值，避免额外开销；n 小时也不会明显变慢
        restarts = 3 if n >= 6 else 1

        # 用 ids 生成确定性 seed，尽量保证“同一批输入 -> 同一输出”（便于复现实验）
        seed = 146527 + (sum(ids) % 1000003)
        rng = random.Random(seed)

        best_centers = None
        best_assignments = None
        best_inertia = float("inf")

        for _ in range(restarts):
            init_centers = _kmeanspp_init(rng)
            centers, assignments, inertia = _run_spherical_kmeans(init_centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_assignments = assignments

        centers = best_centers
        assignments = best_assignments

        # 3) 全局中心：所有单位向量求和后归一化（方向整体中心）
        overall_sum = [0.0] * dim
        for vec in vectors:
            for j, val in enumerate(vec):
                overall_sum[j] += val
        overall = _normalize(overall_sum)

        # 4) 簇得分：与整体中心的“角距离” × (n/size)
        #    角距离这里用 cosine distance（越大越离群）
        #    同时复用 counts，避免 size 重复遍历
        counts = [0] * k
        for a in assignments:
            if 0 <= a < k:
                counts[a] += 1

        scores = []
        for i in range(k):
            size = counts[i]
            if size == 0:
                scores.append(float("inf"))
                continue
            d = cosine_dist_unit(centers[i], overall)
            score = d * (n / size)
            scores.append(score)

        def _median(vals: List[float]) -> float:
            vals_sorted = sorted(vals)
            m = len(vals_sorted)
            mid = m // 2
            if m % 2:
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
        suspect_clusters = [i for i, sc in enumerate(scores) if sc > threshold]
        suspects = [ids[idx] for idx, c in enumerate(assignments) if c in suspect_clusters]

        print(f"[CSP] Cluster(Cosine) 检测：scores={scores}, 阈值={threshold:.6f}")
        if suspects:
            print(f"[CSP] Cluster(Cosine) 可疑 DO: {suspects}")
        else:
            print("[CSP] Cluster(Cosine) 未发现可疑 DO（根据当前阈值）。")
        return suspects
        
    def detect_poison_lasa_lite(self, angle_threshold: float = 0.0, beta: float = 1.5) -> List[int]:
        """
        LASA-lite风格检测：方向和模长双重检查。
        angle_threshold 为参考方向的余弦相似度阈值（越大越严格）。
        """
        if not self.do_projection_map:
            print("[CSP] LASA-lite 检测跳过：无投影数据。")
            return []

        ids = sorted(self.do_projection_map.keys())
        vectors = [self.do_projection_map[i] for i in ids]
        n = len(vectors)
        if n == 0:
            return []
        dim = len(vectors[0])

        def norm(vec):
            return math.sqrt(sum(x * x for x in vec))

        # 方向标准化
        norm_dirs = []
        lengths = []
        for vec in vectors:
            length = norm(vec)
            lengths.append(length)
            if length == 0:
                norm_dirs.append([0.0] * dim)
            else:
                norm_dirs.append([x / length for x in vec])

        # 参考方向（归一化方向的平均）
        ref = [0.0] * dim
        for dir_vec in norm_dirs:
            for j, val in enumerate(dir_vec):
                ref[j] += val
        ref_norm = norm(ref)
        if ref_norm == 0:
            print("[CSP] LASA-lite 参考方向无效，跳过检测。")
            return []
        ref = [x / ref_norm for x in ref]

        # 余弦相似度
        cos_sims = []
        for dir_vec in norm_dirs:
            cos_sims.append(sum(a * b for a, b in zip(dir_vec, ref)))

        # 模长阈值
        def _median(vals: List[float]) -> float:
            vals_sorted = sorted(vals)
            m = len(vals_sorted)
            mid = m // 2
            if m % 2:
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

        med_len = _median(lengths)
        iqr_len = _iqr(lengths)
        upper = med_len + beta * iqr_len
        lower = med_len - beta * iqr_len

        suspects = []
        for do_id, cos_sim, length in zip(ids, cos_sims, lengths):
            direction_bad = cos_sim < angle_threshold
            magnitude_bad = (length > upper) or (length < max(0.0, lower))
            if direction_bad and magnitude_bad:
                suspects.append(do_id)
        print(f"[CSP] LASA-lite 检测：angle阈值={angle_threshold}, len阈值=[{lower:.3f}, {upper:.3f}]")
        if suspects:
            print(f"[CSP] LASA-lite 可疑 DO: {suspects}")
        else:
            print("[CSP] LASA-lite 未发现可疑 DO（根据当前阈值）。")
        return suspects

    # ============== SafeMul: 1+3轮（PA侧） ==============
    def safe_mul_prepare_payload(self) -> Dict[str, object]:
        """准备安全点积第1轮数据，基于 CSP 的正交向量组发送 (p, alpha, C_all)。"""
        sip = SafeInnerProduct(precision_factor=self.precision)
        # Extend vectors by one dim; CSP side uses trailing 1 to carry DO noise r.
        print("setup safe mul with orthogonal vectors count:", len(self.orthogonal_vectors_for_csp))
        extended_vectors = [list(vec) + [1.0] for vec in self.orthogonal_vectors_for_csp]
        print("extended vector dimension:", len(extended_vectors[0]))
        p, alpha, C_all, s, s_inv = sip.round1_setup_and_encrypt(extended_vectors)
        return {'p': p, 'alpha': alpha, 'C_all': C_all, 's_inv': s_inv}

    def safe_mul_finalize(self, ctx: Dict[str, object], D_sums: List[int], do_part: List[float], do_id: int) -> List[float]:
        """
        执行安全点积第3轮并与 DO 明文部分求和，得到 w·U 的 1×m 向量，并缓存该 DO 的映射结果。
        
        关键设计：DO已通过两个返回值(D_sums和do_part)实现掩码化机制：
        - D_sums decrypts to csp_part that includes +r per projection.
        - do_part already subtracts r per projection on DO side.
        - Adding them cancels r; CSP never sees r directly.
        """
        sip = SafeInnerProduct(precision_factor=self.precision)
        p = ctx['p']
        alpha = ctx['alpha']
        s_inv = ctx['s_inv']
        csp_part = sip.round3_decrypt(D_sums, s_inv, alpha, p)
        # 直接相加，扰动自动消去
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

    print("\n===== Round 2: 模拟 DO2 掉线并恢复 =====")
    global_params = csp.broadcast_params()
    do_list[2] = None  # 模拟掉线

    do_cipher_map = {}
    for do in [d for d in do_list if d is not None]:
        ciphertexts = do.train_and_encrypt(global_params)
        do_cipher_map[do.id] = ciphertexts

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
  
