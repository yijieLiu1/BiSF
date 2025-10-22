# CSP.py
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hashlib
from typing import List, Dict, Optional
from utils.ImprovedPaillier import ImprovedPaillier
from utils.Threshold import recover_secret


class CSP:
    """中心服务器（CSP）"""

    def __init__(self, ta, model_size: int = 5, precision: int = 10 ** 6):
        self.ta = ta
        self.model_size = model_size
        self.precision = precision

        # 全局模型参数管理
        self.global_params: List[float] = [0.0] * self.model_size
        self.global_params_snapshot: List[float] = list(self.global_params)
        self.round_count: int = 0

        # 为掉线恢复缓存
        self.recoveredNiValues: Dict[int, int] = {}

        # 保存TA生成的正交向量组
        self.orthogonal_vectors_for_csp: List[List[float]] = []
        self._load_orthogonal_vectors()

        # 使用 ImprovedPaillier，但用 TA 的参数覆盖，确保全局一致
        self.impaillier = ImprovedPaillier(m=self.ta.num_do, bit_length=512, precision=self.precision)
        self._sync_paillier_with_ta()

    def _load_orthogonal_vectors(self) -> None:
        self.orthogonal_vectors_for_csp = self.ta.get_orthogonal_vectors_for_csp()
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
            
        print(f"[CSP] 已同步Paillier公参，N={N}, g={self.impaillier.g}, h={self.impaillier.h}, y={self.impaillier.y}")
    # ============== 广播 ==============
    def broadcast_params(self) -> List[float]:
        try:
            self.ta.update_keys_for_new_round()
        except Exception as e:
            print(f"[CSP] 密钥更新失败: {e}")
        self.global_params_snapshot = list(self.global_params)
        self.round_count += 1
        print(f"[CSP] 第{self.round_count}轮广播参数: {self.global_params_snapshot}")
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
        print(f"[CSP] 正常解密结果: {results}")
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
            print(f"[CSP] 恢复解密坐标 {i}: 原始密文={aggregated[i]}, 修正密文={modified} -> 明文={results[i]}")
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
        params_bytes = str(self.global_params_snapshot).encode('utf-8')
        params_hash = int.from_bytes(hashlib.sha256(params_bytes).digest(), 'big', signed=False)

        if not missing_ids:
            summed = self._decrypt_vector(aggregated)
        else:
            print(f"\n\n[CSP] 检测到掉线DO，进行密钥恢复\n")
            self.recover_missing_private_keys(missing_ids, online_dos, self.ta.get_threshold())
            summed = self._decrypt_with_recovery(aggregated, params_hash)

        print(f"[CSP] 解密后的聚合结果: {summed}")

        num_online = max(1, len(online_dos))
        next_params = [
            self.global_params_snapshot[i] + (summed[i] / num_online)
            for i in range(self.model_size)
        ]

        self.global_params = next_params
        print(f"[CSP] 更新后的全局参数: {next_params}")

        return next_params
# ===========================
# === 测试代码部分 (CSP.py) ===
# ===========================
if __name__ == "__main__":
    print("===== [TEST] 启动 CSP 测试 =====")
    from TA.TA import TA
    from DO.DO import DO

    # 初始化
    ta = TA(num_do=3, model_size=5, orthogonal_vector_count=3, bit_length=512)
    csp = CSP(ta)
    do_list = [DO(i, ta) for i in range(3)]

    print("\n===== Round 1: 广播参数 =====")
    global_params = csp.broadcast_params()

    print("\n===== Round 1: 收集各 DO 的密文更新 =====")
    do_cipher_map = {}
    for do in do_list:
        ciphertexts = do.train_and_encrypt(global_params)
        do_cipher_map[do.id] = ciphertexts

    print("\n===== Round 1: CSP 聚合 + 解密更新 =====")
    new_params = csp.round_aggregate_and_update(do_list, do_cipher_map)
    print(f"\n>>> Round 1 结束，新全局参数: {new_params}")

    print("\n===== Round 2: 模拟 DO2 掉线并恢复 =====")
    global_params = csp.broadcast_params()
    do_list[2] = None  # 模拟掉线

    do_cipher_map = {}
    for do in [d for d in do_list if d is not None]:
        ciphertexts = do.train_and_encrypt(global_params)
        do_cipher_map[do.id] = ciphertexts

    new_params = csp.round_aggregate_and_update(do_list, do_cipher_map)
    print(f"\n>>> Round 2 结束（含恢复），新全局参数: {new_params}")

    print("\n===== 测试完成 =====")
