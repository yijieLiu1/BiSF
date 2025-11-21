# TA.py
# TA是可信机构，主要负责：
# 1.基于ImprovedPaillier构建impaillier对象，基于这个对象生成密钥参数。并把基础密钥分发给各自的DO
# 2.每一轮联邦结束后，都对密钥进行更新。
# 3.生成正交向量组，然后把正交向量组分成两部分，用于给DO和CSP。CSP只有一个，DO有多个，每个DO的正交向量组都一样。
# -------------------------------
# Trusted Authority (TA) 实现（Python版本）
# -------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from typing import Dict, List, Tuple, Optional
import secrets
import random
import math

# 可选 NumPy 加速（大规模向量时建议安装）
try:
    import numpy as np
    _NP_AVAILABLE = True
except Exception:
    _NP_AVAILABLE = False

# 导入项目中的秘密分享函数
try:
    from utils.Threshold import split_secret, recover_secret
except Exception:
    def split_secret(*args, **kwargs):
        raise ImportError("threshold.split_secret 未找到，请确保 threshold.py 在 PYTHONPATH 中")
    def recover_secret(*args, **kwargs):
        raise ImportError("threshold.recover_secret 未找到，请确保 threshold.py 在 PYTHONPATH 中")

# 使用 ImprovedPaillier 处理 Paillier 相关密钥/任务密钥拆分
try:
    from utils.ImprovedPaillier import ImprovedPaillier
except Exception:
    raise ImportError("ImprovedPaillier 未找到，请确保 utils/ImprovedPaillier.py 存在且包含 ImprovedPaillier 类")


class TA:
    """联邦学习协议中的可信第三方 (TA)"""

    def __init__(self, num_do: int, model_size: int = 10000, orthogonal_vector_count: int = 1024,
                 bit_length: int = 1024, precision: int = 10**6, k: int = 1 << 48):
        """
        初始化TA
        Args:
            num_do: DO的数量
            model_size: 模型大小(几万维,默认50000)
            orthogonal_vector_count: 正交向量数量(默认2048,用于投影到2048维)
            bit_length: 密钥长度
            precision: 浮点精度
            k: 参数k
        """
        self.num_do = num_do
        self.bit_length = bit_length
        self.precision = precision
        self.k = k
        self.threshold = max(1, (num_do * 2) // 3)  # 阈值门限

        # 加密参数（从ImprovedPaillier获取）
        self.N: int = 0
        self.g: int = 0
        self.h: int = 0
        self.lambda_val: int = 0
        self.mu: int = 0
        self.gamma: int = 0

        # 每个DO的私钥和秘密分享
        self.do_private_keys: Dict[int, int] = {}
        self.do_key_shares: Dict[int, Dict] = {}  # {'shares': {do_id: share}, 'prime': prime_used}
        self.n_i: List[int] = [0] * num_do

        # R_t（任务密钥）
        self.R_t: int = 0

        # ImprovedPaillier 实例
        self.impaillier: Optional[ImprovedPaillier] = None

        # 正交向量
        self.MODEL_SIZE = model_size
        self.ORTHOGONAL_VECTOR_COUNT = orthogonal_vector_count
        self.orthogonal_vectors: List[List[float]] = []
        self.orthogonal_vectors_for_csp: List[List[float]] = []
        self.orthogonal_vectors_for_do: List[List[float]] = []
        self.orthogonal_sumvectors_for_csp: List[float] = []

        # 轮次管理
        self.current_round: int = 0
        self.key_history: List[Dict] = []  # 存储历史密钥信息

        # 初始化正交向量和密钥
        self._generate_orthogonal_vectors()
        self._key_generation()

    # ---------- 工具函数 ----------
    def _lcm(self, a: int, b: int) -> int:
        return a // math.gcd(a, b) * b

    # ---------- 正交向量 ----------
    def _generate_orthogonal_vectors(self) -> None:
        print(f"TA开始生成正交向量组{self.MODEL_SIZE}*{self.ORTHOGONAL_VECTOR_COUNT}")
        t1=time.time()
        # 以轮次和 R_t 混合的安全随机数作为种子，确保每轮不同
        try:
            mix = (self.current_round + 1) ^ (self.R_t if isinstance(self.R_t, int) else 0)
        except Exception:
            mix = (self.current_round + 1)
        seed = int.from_bytes(secrets.token_bytes(16), 'big') ^ mix
        rng = random.Random(seed)

        # 大规模场景优先采用 NumPy QR 分解生成正交基（更快更稳）
        use_numpy_qr = _NP_AVAILABLE and (self.MODEL_SIZE >= 2000 or self.ORTHOGONAL_VECTOR_COUNT >= 64)
        if use_numpy_qr:
            # 生成尺寸为 (MODEL_SIZE, ORTHOGONAL_VECTOR_COUNT) 的高斯随机矩阵
            # 对其做 QR 分解，得到列正交的 Q（每列长度为 MODEL_SIZE，共 ORTHOGONAL_VECTOR_COUNT 列）
            rnd = np.random.default_rng(seed)
            A = rnd.standard_normal(size=(self.MODEL_SIZE, self.ORTHOGONAL_VECTOR_COUNT))
            # 使用经济 QR 分解
            Q, _ = np.linalg.qr(A, mode='reduced')
            self.orthogonal_vectors = [Q[:, i].astype(float).tolist() for i in range(Q.shape[1])]
        else:
            # 纯 Python 的 Gram-Schmidt（小规模时可接受）
            V = [[rng.gauss(0, 1) for _ in range(self.MODEL_SIZE)]
                 for _ in range(self.ORTHOGONAL_VECTOR_COUNT)]

            U: List[List[float]] = []
            for v in V:
                w = v.copy()
                for u in U:
                    dot = sum(wi * ui for wi, ui in zip(w, u))
                    norm_sq = sum(ui * ui for ui in u)
                    if norm_sq != 0:
                        coef = dot / norm_sq
                        for idx in range(self.MODEL_SIZE):
                            w[idx] -= coef * u[idx]
                norm = math.sqrt(sum(x * x for x in w))
                if norm > 0:
                    w = [x / norm for x in w]
                else:
                    w = [(xi + 1e-8) for xi in w]
                    norm = math.sqrt(sum(x * x for x in w))
                    w = [x / norm for x in w]
                U.append(w)
            self.orthogonal_vectors = U

        print(f"生成的正交向量组：共{len(self.orthogonal_vectors)}个向量，每个向量{self.MODEL_SIZE}维")
        # 只打印前3个向量的前5维，避免输出过多
        for i in range(min(3, len(self.orthogonal_vectors))):
            vec_preview = self.orthogonal_vectors[i][:5]
            print(f"向量{i}前5维: {vec_preview}...")

        # self._check_orthogonality()

        # 初始化 CSP 和 DO 分配向量
        if _NP_AVAILABLE and use_numpy_qr:
            ratios = np.random.default_rng(seed ^ 0xABCDEF).random(size=(self.ORTHOGONAL_VECTOR_COUNT, self.MODEL_SIZE))
            U_np = np.array(self.orthogonal_vectors)
            csp_np = U_np * ratios
            do_np = U_np * (1.0 - ratios)
            self.orthogonal_vectors_for_csp = [row.tolist() for row in csp_np]
            self.orthogonal_vectors_for_do = [row.tolist() for row in do_np]
        else:
            self.orthogonal_vectors_for_csp = [[0.0]*self.MODEL_SIZE for _ in range(self.ORTHOGONAL_VECTOR_COUNT)]
            self.orthogonal_vectors_for_do = [[0.0]*self.MODEL_SIZE for _ in range(self.ORTHOGONAL_VECTOR_COUNT)]
            for i in range(self.ORTHOGONAL_VECTOR_COUNT):
                for j in range(self.MODEL_SIZE):
                    ratio = rng.random()
                    val = self.orthogonal_vectors[i][j]
                    self.orthogonal_vectors_for_csp[i][j] = val * ratio
                    self.orthogonal_vectors_for_do[i][j] = val * (1 - ratio)
        # 生成 CSP 使用的求和向量（按列求和，得到长度为 MODEL_SIZE 的列向量）
        if self.orthogonal_vectors_for_csp:
            self.orthogonal_sumvectors_for_csp = [
                float(sum(col)) for col in zip(*self.orthogonal_vectors)
            ]
        else:
            self.orthogonal_sumvectors_for_csp = [0.0] * self.MODEL_SIZE
        t2=time.time()
        # print(f"生成的分配给CSP的向量组{self.orthogonal_sumvectors_for_csp}，长度({len(self.orthogonal_sumvectors_for_csp)})")
        print(f"TA生成正交向量组{self.MODEL_SIZE}*{self.ORTHOGONAL_VECTOR_COUNT}共用时{t2-t1}s")

    def _check_orthogonality(self) -> None:
        # 对于大维度向量，只检查前几对向量的正交性，避免计算量过大
        check_count = min(10, self.ORTHOGONAL_VECTOR_COUNT)
        for i in range(check_count):
            for j in range(i + 1, min(i + 2, check_count)):
                dot = sum(self.orthogonal_vectors[i][k] * self.orthogonal_vectors[j][k]
                          for k in range(self.MODEL_SIZE))
                if abs(dot) > 1e-6:
                    print(f"警告: 向量 {i} 和 向量 {j} 的点积: {dot} (可能不严格正交)")

    # ---------- 密钥生成（使用 ImprovedPaillier） ----------
    def _key_generation(self) -> None:
        """
        使用 ImprovedPaillier 实例生成密钥，并同步密钥参数到 TA。
        """
        # 1. 创建 ImprovedPaillier 实例
        self.impaillier = ImprovedPaillier(
            m=self.num_do,
            bit_length=self.bit_length,
            precision=self.precision
        )

        # 2. 与 impaillier 同步必要参数

        self.N = self.impaillier.getN()

        self.g = self.impaillier.g

        self.h = self.impaillier.h

        self.lambda_val = self.impaillier.lambda_

        self.u = self.impaillier.u

        self.gamma = self.impaillier.y  # 同步 y 参数到 gamma

        self.R_t = self.impaillier.R_t

        self.n_i = self.impaillier.n_i

        self.do_private_keys = {i: sk for i, sk in enumerate(self.impaillier.SK_DO)}



        # 3. 根据当前私钥生成门限秘密分享

        self._generate_key_shares()



        # 4. 保存密钥快照
        self._save_key_info()

    def _save_key_info(self) -> None:
        """保存当前轮次的密钥信息到历史记录"""
        key_info = {
            'round': self.current_round,
            'R_t': self.R_t,
            'n_i': self.n_i.copy(),
            'do_private_keys': self.do_private_keys.copy(),
            'timestamp': secrets.randbits(64)  # 简单的时间戳
        }
        self.key_history.append(key_info)

    def _generate_key_shares(self) -> None:
        """根据当前 DO 私钥生成门限秘密分享"""
        self.do_key_shares = {}
        for i in range(self.num_do):
            try:
                shares, prime_used = split_secret(self.do_private_keys[i], self.threshold, self.num_do)
            except Exception as e:
                raise RuntimeError(f"DO {i} 的秘密分享失败: {e}")

            distributed = {j: shares[j + 1] for j in range(self.num_do) if j != i}
            self.do_key_shares[i] = {'shares': distributed, 'prime': prime_used}

    def get_aggregated_base_key(self, do_ids: List[int]) -> int:
        """
        返回指定 DO 集合的基础私钥乘积（mod N^2），用于补偿缺失 DO。
        空集合返回 1。
        """
        N_sq = self.N * self.N
        result = 1
        for do_id in do_ids:
            sk = self.do_private_keys.get(do_id)
            if sk is None:
                continue
            result = (result * sk) % N_sq
        return result

    # ---------- 密钥更新功能 ----------
    def update_keys_for_new_round(self) -> None:
        """
        每轮联邦学习结束后的密钥更新功能
        1. 更新R_t
        2. 重新生成DO私钥
        3. 更新轮次计数
        """
        print(f"开始第 {self.current_round + 1} 轮密钥更新...")
        
        # 更新轮次
        self.current_round += 1
        
        # 生成新的R_t
        self._generate_new_R_t()
        
        # 重新计算所有DO的私钥
        self._update_do_private_keys()
        # 并基于新私钥刷新门限秘密分享
        self._generate_key_shares()

        # 每轮刷新正交向量组，并重新拆分给 CSP 与 DO
        # 防止各轮使用同一组向量影响投毒检测的鲁棒性
        self._generate_orthogonal_vectors()
        
        # 保存新的密钥信息
        self._save_key_info()
        
        print(f"第 {self.current_round} 轮密钥更新完成")

    def _generate_new_R_t(self) -> None:
        """生成新的R_t，确保与N互质"""
        while True:
            candidate = secrets.randbits(self.bit_length)
            if candidate > 1 and math.gcd(candidate, self.N) == 1:
                self.R_t = candidate
                break

    def _update_do_private_keys(self) -> None:
        """根据新的R_t更新所有DO的私钥"""
        N_sq = self.N * self.N
        for i in range(self.num_do):
            self.do_private_keys[i] = pow(self.R_t, self.n_i[i], N_sq)

    # ---------- 密钥恢复功能 ----------
    def recover_do_key(self, do_id: int, available_do_ids: List[int]) -> Optional[int]:
        """
        当某个DO掉线时，使用其他DO的shares恢复其密钥
        Args:
            do_id: 需要恢复密钥的DO ID
            available_do_ids: 可用的DO ID列表
        Returns:
            恢复的密钥，如果失败返回None
        """
        if do_id not in self.do_key_shares:
            return None
        
        shares_info = self.do_key_shares[do_id]
        available_shares = {i+1: shares_info['shares'][i] for i in available_do_ids if i in shares_info['shares']}
        
        if len(available_shares) < self.threshold:
            print(f"可用shares数量不足，需要{self.threshold}个，只有{len(available_shares)}个")
            return None
        
        try:
            # 选择足够的shares进行恢复
            selected_shares = dict(list(available_shares.items())[:self.threshold])
            recovered_key = recover_secret(selected_shares, shares_info['prime'])
            
            # 验证恢复的私钥是否正确
            if recovered_key == self.do_private_keys.get(do_id):
                return recovered_key
            else:
                print(f"恢复的DO {do_id} 私钥不正确")
                return None
        except Exception as e:
            print(f"恢复DO {do_id}的密钥失败: {e}")
            return None

    # ---------- getter ----------
    def get_N(self) -> int: return self.N
    def get_base_key(self, do_id: int) -> int: return self.do_private_keys.get(do_id)
    def get_g(self) -> int: return self.g
    def get_h(self) -> int: return self.h
    def get_lambda(self) -> int: return self.lambda_val
    def get_mu(self) -> int: return self.mu
    def get_gamma(self) -> int: return self.gamma
    def get_threshold(self) -> int: return self.threshold
    def get_R_t(self) -> int: return self.R_t
    def get_ni(self, do_id: int) -> int: return self.n_i[do_id]
    def get_orthogonal_vectors(self) -> List[List[float]]: return self.orthogonal_vectors
    def get_orthogonal_vectors_for_csp(self) -> List[List[float]]: return self.orthogonal_vectors_for_csp
    def get_orthogonal_vectors_for_do(self) -> List[List[float]]: return self.orthogonal_vectors_for_do
    def get_orthogonal_sumvectors_for_csp(self) -> List[float]: return self.orthogonal_sumvectors_for_csp
    def get_model_size(self) -> int: return self.MODEL_SIZE
    def get_current_round(self) -> int: return self.current_round
    def get_key_history(self) -> List[Dict]: return self.key_history.copy()

    # ---------- 兼容性方法（保持向后兼容） ----------
    def get_u(self) -> int: return self.mu  # 兼容旧接口
    def get_y(self) -> int: return self.gamma  # 兼容旧接口

    # ---------- 每轮刷新 R_t 与基础密钥（保持向后兼容） ----------
    def rotate_R_t(self) -> None:
        """每轮更新 R_t，并据此重算每个 DO 的基础密钥 sk_i = R_t^{n_i} mod N^2"""
        self.update_keys_for_new_round()


# ----------------- 测试 -----------------
if __name__ == "__main__":
    print("初始化TA（5个DO）...")
    ta = TA(num_do=3, model_size=10000, orthogonal_vector_count=1024, bit_length=512)

    print("\n=== 全局参数 ===")
    print("N:", ta.get_N())
    print("threshold:", ta.get_threshold())
    print("R_t:", ta.get_R_t())
    print("当前轮次:", ta.get_current_round())

    print("\n=== DO私钥及校验 ===")
    correct_count = 0
    for do_id in range(ta.num_do):
        sk = ta.get_base_key(do_id)
        ni = ta.get_ni(do_id)
        expected_sk = pow(ta.get_R_t(), ni, ta.get_N()**2)
        is_correct = sk == expected_sk
        if is_correct: correct_count += 1
        print(f"DO {do_id} -> sk: {sk}, n_i: {ni}, 校验: {'正确' if is_correct else '错误'}")
    print(f"正确私钥数量: {correct_count}/{ta.num_do}")

    print("\n=== 正交向量组 ===")
    # print("原始正交向量:")
    # for i, vec in enumerate(ta.get_orthogonal_vectors()):
    #     print(f"向量{i}: {vec}")
    # print("CSP向量组:")
    # for i, vec in enumerate(ta.get_orthogonal_vectors_for_csp()):
    #     print(f"向量{i}: {vec}")
    # print("DO向量组:")
    # for i, vec in enumerate(ta.get_orthogonal_vectors_for_do()):
    #     print(f"向量{i}: {vec}")

    print("\n=== 模拟门限恢复 n_i ===")
    missing_do = 0
    available_do_ids = [i for i in range(ta.num_do) if i != missing_do]
    threshold_needed = ta.get_threshold()
    shares_info = ta.do_key_shares[missing_do]
    used_shares = {i+1: shares_info['shares'][i] for i in available_do_ids[:threshold_needed]}
    #使用生成 shares 时用的那个同一个质数 p
    prime_used = shares_info['prime']
    print(f"DO {missing_do} 掉线，使用 {threshold_needed} 个 shares 恢复: {used_shares}")

    recovered_n = recover_secret(used_shares, prime_used)
    print(f"恢复出的 n_{missing_do}: {recovered_n}")
    print(f"原始 n_{missing_do}: {ta.get_base_key(missing_do)}, 恢复是否正确: {recovered_n ==ta.get_base_key(missing_do)}")

    print("\n=== 测试密钥更新功能 ===")
    print("第0轮密钥信息:")
    print(f"R_t: {ta.get_R_t()}")
    print(f"DO0私钥: {ta.get_base_key(0)}")
    
    # 模拟联邦学习结束，更新密钥
    ta.update_keys_for_new_round()
    print("\n第1轮密钥信息:")
    print(f"R_t: {ta.get_R_t()}")
    print(f"DO0私钥: {ta.get_base_key(0)}")
    
    # 再次更新
    ta.update_keys_for_new_round()
    print("\n第2轮密钥信息:")
    print(f"R_t: {ta.get_R_t()}")
    print(f"DO0私钥: {ta.get_base_key(0)}")
    
    print(f"\n密钥历史记录数量: {len(ta.get_key_history())}")
    for i, key_info in enumerate(ta.get_key_history()):
        print(f"轮次 {i}: R_t={key_info['R_t']}, DO0私钥={key_info['do_private_keys'][0]}")

    print("\n=== 测试密钥恢复功能 ===")
    # 模拟DO 0掉线，使用其他DO恢复其密钥
    available_do_ids = [1, 2, 3, 4]  # 假设DO 0掉线
    recovered_key = ta.recover_do_key(0, available_do_ids)
    if recovered_key:
        print(f"成功恢复DO 0的密钥: {recovered_key}")
        print(f"与当前密钥一致: {recovered_key == ta.get_base_key(0)}")
    else:
        print("恢复DO 0的密钥失败")

    # 测试加密解密功能
    print("\n=== 测试与ImprovedPaillier的集成（含聚合） ===")
    test_values = [1.5, -2.3, 3.75, 0.5, -1.25][:ta.num_do]  # 每个DO一个浮点数
    print(f"原始明文向量: {test_values}")

    # 各 DO 加密自己的值
    encrypted_values = [
        ta.impaillier.encrypt(val, ta.get_base_key(i))
        for i, val in enumerate(test_values)
    ]

    # 聚合同态加密结果（连乘 mod N²）
    aggregated_ciphertext = ta.impaillier.aggregate(encrypted_values)
    print(f"聚合后的密文: {aggregated_ciphertext}")

    # 解密聚合密文
    decrypted_sum = ta.impaillier.decrypt(aggregated_ciphertext)
    expected_sum = sum(test_values)

    print(f"解密得到的求和值: {decrypted_sum}")
    print(f"理论正确求和: {expected_sum}")
    print(f"误差: {abs(decrypted_sum - expected_sum)}")

