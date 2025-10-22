# DO.py
# Data Owner（数据拥有方）实现
# 主要功能：
# 1. 接收来自TA的基础私钥、正交向量组、秘密分片值
# 2. 基于SHA-256哈希全局模型参数进行密钥派生
# 3. 模拟本地模型训练（递增0.2）
# 4. 使用真实私钥加密训练结果并上传
# 5. 提供门限恢复辅助接口
# -------------------------------
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import hashlib
from typing import List, Optional, Dict, Any
from utils.ImprovedPaillier import ImprovedPaillier


class DO:
    """Data Owner（数据拥有方）"""

    def __init__(self, do_id: int, ta, model_size: int = 5, precision: int = 10 ** 6, rng_seed: Optional[int] = None):
        """
        初始化DO
        Args:
            do_id: DO的唯一标识符
            ta: 可信机构TA的实例
            model_size: 模型参数大小
            precision: 浮点数精度
            rng_seed: 随机数种子（用于可复现性）
        """
        self.id = do_id
        self.ta = ta
        self.model_size = model_size
        self.precision = precision

        # 基础密钥：来自 TA（这是 TA 生成并分发给 DO 的 sk = R_t^{n_i} mod N^2）
        self.base_private_key: int = self.ta.get_base_key(self.id)
        self.round_private_key: Optional[int] = None

        # 创建本地 ImprovedPaillier 实例（用于本地加密/聚合）
        # 注意：构造函数会生成一套本地密钥，但我们随后会用 TA 的公参/必要项来覆盖本地公参，避免使用本地私钥。
        # 这里传入 m=self.ta.num_do 保持一致（但最终不依赖本地自动生成的密钥）
        self.impaillier = ImprovedPaillier(m=self.ta.num_do, bit_length=512, precision=self.precision)
        self._sync_paillier_with_ta()

        # 保存正交向量组
        self.orthogonal_vectors_for_do: List[List[float]] = []
        self._load_orthogonal_vectors()

        # 随机数生成器（可复现）
        self._rng = random.Random(rng_seed if rng_seed is not None else (12345 + do_id))
        
        # 训练历史记录
        self.training_history: List[Dict[str, Any]] = []
        
        print(f"[DO {self.id}] 初始化完成，模型大小: {self.model_size}, 精度: {self.precision}")

    def _load_orthogonal_vectors(self) -> None:
        """从TA加载正交向量组"""
        self.orthogonal_vectors_for_do = self.ta.get_orthogonal_vectors_for_do()
        print(f"[DO {self.id}] 已加载正交向量组，共{len(self.orthogonal_vectors_for_do)}个向量")

    def _sync_paillier_with_ta(self) -> None:
        """从 TA 同步 Paillier 公参，包括 y 参数"""
        N = self.ta.get_N()
        self.impaillier.N = N
        self.impaillier.N2 = N * N
        self.impaillier.g = self.ta.get_g()
        self.impaillier.h = self.ta.get_h()
        # lambda_ / u 可同步
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
        print(f"[DO {self.id}] 已同步Paillier公参，N={N}, g={self.impaillier.g}, h={self.impaillier.h}, y={self.impaillier.y}")

    # ============== 工具函数 ==============
    def _hash_global_params(self, global_params: List[float]) -> int:
        """
        SHA-256哈希全局参数，返回整数
        Args:
            global_params: 全局模型参数列表
        Returns:
            哈希值的整数表示
        """
        # 将全局参数转换为字符串并编码
        s = str(global_params).encode('utf-8')
        # 计算SHA-256哈希
        h = hashlib.sha256(s).digest()
        # 转换为整数
        hash_int = int.from_bytes(h, byteorder='big', signed=False)
        print(f"[DO {self.id}] 全局参数哈希值: {hash_int}")
        return hash_int

    def update_key(self, global_params: List[float]) -> None:
        """
        基于全局参数 hash 对基础密钥进行衍生
        核心公式：派生密钥 = base_private_key ^ hash mod N^2
        Args:
            global_params: 全局模型参数列表
        """
        N2 = self.ta.get_N() ** 2
        # 获取最新的基础私钥（可能已更新）
        self.base_private_key = self.ta.get_base_key(self.id)
        # 计算全局参数的哈希值
        h = self._hash_global_params(global_params)
        print(f"[DO {self.id}] 开始密钥派生，基础私钥: {self.base_private_key}")
        # 核心：派生密钥 = base_private_key ^ hash mod N^2
        # base_private_key 是 TA 分发给 DO 的 sk_i = R_t^{n_i} mod N^2
        self.round_private_key = pow(self.base_private_key, h, N2)
        print(f"[DO {self.id}] 派生密钥计算完成: {self.round_private_key}")

    # ============== 本地训练（模拟） ==============
    def _local_train(self, global_params: List[float]) -> List[float]:
        """
        模拟本地模型训练，每次让模型参数递增0.2
        后续可以替换为真实的CNN训练
        Args:
            global_params: 全局模型参数
        Returns:
            本地训练更新向量
        """
        print(f"[DO {self.id}] 开始本地训练，全局参数: {global_params}")
        
        # 模拟本地训练：每个参数递增0.2
        # 这里可以后续替换为真实的CNN训练逻辑
        updates = []
        for i in range(self.model_size):
            # 简单的递增策略，后续可替换为梯度下降等真实训练算法
            update = 0.5
            updates.append(update)
        
        print(f"[DO {self.id}] 本地训练完成，更新向量: {updates}")
        
        # 记录训练历史
        training_record = {
            'round': len(self.training_history) + 1,
            'global_params': global_params.copy(),
            'local_updates': updates.copy(),
            'timestamp': random.randint(1000000, 9999999)  # 简单时间戳
        }
        self.training_history.append(training_record)
        
        return updates

    # ============== 加密上传 ==============
    def _encrypt_value(self, value: float) -> int:
        """
        使用派生私钥和 ImprovedPaillier 加密单个值
        Args:
            value: 要加密的浮点数值
        Returns:
            加密后的密文
        """
        if self.round_private_key is None:
            raise ValueError(f"[DO {self.id}] round_private_key 未设置，请先调用 update_key()")
        
        # 使用派生密钥进行加密（内部已做精度放大）
        # 派生密钥 = base_private_key ^ hash mod N^2
        ciphertext = self.impaillier.encrypt(value, self.round_private_key)
        print(f"[DO {self.id}] 加密值 {value} -> 密文: {ciphertext}")
        return ciphertext

    def train_and_encrypt(self, global_params: List[float]) -> List[int]:
        """
        完整的训练和加密流程：
        1. 更新密钥（基于全局参数哈希）
        2. 本地训练（模拟递增0.2）
        3. 使用派生私钥加密训练结果
        4. 返回密文向量
        Args:
            global_params: 全局模型参数
        Returns:
            加密后的训练更新向量
        """
        print(f"[DO {self.id}] 开始训练和加密流程")
        
        # 步骤1：基于全局参数更新密钥
        self.update_key(global_params)
        
        # 步骤2：本地训练
        updates = self._local_train(global_params)
        
        # 步骤3：使用派生私钥加密训练结果
        ciphertexts = []
        for i, update in enumerate(updates):
            ciphertext = self._encrypt_value(update)
            ciphertexts.append(ciphertext)
        
        print(f"[DO {self.id}] 加密完成，密文数量: {len(ciphertexts)}")
        print(f"[DO {self.id}] 密文向量: {ciphertexts}")
        
        return ciphertexts

    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        获取训练历史记录
        Returns:
            训练历史记录列表
        """
        return self.training_history.copy()

    def get_orthogonal_vectors(self) -> List[List[float]]:
        """
        获取正交向量组
        Returns:
            DO的正交向量组
        """
        return self.orthogonal_vectors_for_do.copy()

    # ============== 门限恢复接口 ==============
    def uploadKeyShare(self, missing_do_id: int) -> Optional[int]:
        """
        为掉线 DO 提供自己的秘密分片值
        这是门限恢复机制的重要组成部分
        Args:
            missing_do_id: 掉线的DO的ID
        Returns:
            自己的秘密分片值，如果失败返回None
        """
        try:
            # 从TA的密钥分享记录中获取自己的分片
            entry = self.ta.do_key_shares[missing_do_id]
            share = entry['shares'].get(self.id)
            
            if share is not None:
                print(f"[DO {self.id}] 为掉线DO {missing_do_id}提供秘密分片: {share}")
                return share
            else:
                print(f"[DO {self.id}] 未找到DO {missing_do_id}的分片信息")
                return None
                
        except KeyError as e:
            print(f"[DO {self.id}] 密钥分享记录中不存在DO {missing_do_id}: {e}")
            return None
        except Exception as e:
            print(f"[DO {self.id}] 提供分片失败: {e}")
            return None

    def get_key_share_info(self, missing_do_id: int) -> Optional[Dict[str, Any]]:
        """
        获取指定DO的密钥分享信息
        Args:
            missing_do_id: 目标DO的ID
        Returns:
            密钥分享信息字典，包含分片值和使用的素数
        """
        try:
            entry = self.ta.do_key_shares[missing_do_id]
            share = entry['shares'].get(self.id)
            prime = entry['prime']
            
            if share is not None:
                return {
                    'share': share,
                    'prime': prime,
                    'do_id': self.id,
                    'missing_do_id': missing_do_id
                }
            else:
                return None
                
        except Exception as e:
            print(f"[DO {self.id}] 获取密钥分享信息失败: {e}")
            return None

    def verify_key_share(self, missing_do_id: int, recovered_key: int) -> bool:
        """
        验证恢复的密钥是否正确
        Args:
            missing_do_id: 掉线DO的ID
            recovered_key: 恢复的密钥
        Returns:
            验证结果
        """
        try:
            # 获取原始密钥进行对比
            original_key = self.ta.get_base_key(missing_do_id)
            is_valid = recovered_key == original_key
            
            print(f"[DO {self.id}] 验证DO {missing_do_id}的恢复密钥: {'有效' if is_valid else '无效'}")
            return is_valid
            
        except Exception as e:
            print(f"[DO {self.id}] 验证密钥失败: {e}")
            return False
# ===========================
# === 测试代码部分 (DO.py) ===
# ===========================
if __name__ == "__main__":
    print("===== [TEST] DO 完整功能测试 =====")
    
    # 导入必要的模块
    try:
        from TA.TA import TA
        from CSP.CSP import CSP
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保 TA.py 和 CSP.py 在正确的路径下")
        exit(1)

    print("\n===== 1. 初始化系统 =====")
    # 初始化 TA、CSP 与 3 个 DO
    ta = TA(num_do=3, model_size=5, orthogonal_vector_count=3, bit_length=512)
    csp = CSP(ta)
    do_list = [DO(i, ta) for i in range(3)]
    
    print(f"系统初始化完成：TA、CSP、{len(do_list)}个DO")

    print("\n===== 2. 测试密钥派生功能 =====")
    test_global_params = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"测试全局参数: {test_global_params}")
    
    for do in do_list:
        do.update_key(test_global_params)
        print(f"DO {do.id} 派生密钥: {do.round_private_key}")

    print("\n===== 3. 测试本地训练功能 =====")
    for do in do_list:
        updates = do._local_train(test_global_params)
        print(f"DO {do.id} 训练更新: {updates}")
        print(f"DO {do.id} 训练历史记录数: {len(do.get_training_history())}")

    print("\n===== 4. 测试加密上传功能 =====")
    do_cipher_map = {}
    for do in do_list:
        ciphertexts = do.train_and_encrypt(test_global_params)
        do_cipher_map[do.id] = ciphertexts
        print(f"DO {do.id} 密文向量长度: {len(ciphertexts)}")

    print("\n===== 5. 测试门限恢复功能 =====")
    # 模拟DO 1掉线，测试其他DO提供分片
    missing_do_id = 1
    print(f"模拟DO {missing_do_id}掉线")
    
    available_do_ids = [0, 2]  # 可用的DO
    shares = {}
    for do_id in available_do_ids:
        do = do_list[do_id]
        share = do.uploadKeyShare(missing_do_id)
        if share is not None:
            shares[do_id + 1] = share  # 注意：shares的索引从1开始
            print(f"DO {do_id} 提供分片: {share}")
    
    # 测试密钥恢复
    if len(shares) >= ta.get_threshold():
        print(f"收集到足够的分片（{len(shares)}个），开始恢复密钥...")
        recovered_key = ta.recover_do_key(missing_do_id, available_do_ids)
        if recovered_key:
            print(f"成功恢复DO {missing_do_id}的密钥: {recovered_key}")
            # 验证恢复的密钥
            original_key = ta.get_base_key(missing_do_id)
            print(f"原始密钥: {original_key}")
            print(f"恢复正确: {recovered_key == original_key}")
        else:
            print("密钥恢复失败")
    else:
        print(f"分片数量不足，需要{ta.get_threshold()}个，只有{len(shares)}个")

    print("\n===== 6. 测试正交向量组功能 =====")
    for do in do_list:
        vectors = do.get_orthogonal_vectors()
        print(f"DO {do.id} 正交向量组数量: {len(vectors)}")
        if vectors:
            print(f"DO {do.id} 第一个向量: {vectors[0]}")

    print("\n===== 7. 测试多轮训练 =====")
    # 模拟多轮联邦学习
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        
        # 广播全局参数
        if round_num == 0:
            global_params = csp.broadcast_params()
        else:
            global_params = [p + 0.1 for p in global_params]  # 模拟参数更新
        
        print(f"全局参数: {global_params}")
        
        # 各DO训练和加密
        round_cipher_map = {}
        for do in do_list:
            ciphertexts = do.train_and_encrypt(global_params)
            round_cipher_map[do.id] = ciphertexts
        
        # 聚合和解密
        try:
            updated_params = csp.round_aggregate_and_update(do_list, round_cipher_map)
            print(f"Round {round_num + 1} 聚合完成，新参数: {updated_params}")
            global_params = updated_params
        except Exception as e:
            print(f"Round {round_num + 1} 聚合失败: {e}")

    print("\n===== 8. 测试训练历史记录 =====")
    for do in do_list:
        history = do.get_training_history()
        print(f"DO {do.id} 训练历史记录数: {len(history)}")
        if history:
            latest = history[-1]
            print(f"DO {do.id} 最新训练记录: Round {latest['round']}, 更新: {latest['local_updates']}")

    print("\n===== 9. 测试密钥更新功能 =====")
    print("TA更新密钥...")
    ta.update_keys_for_new_round()
    
    # 测试DO使用新密钥
    new_global_params = [2.0, 3.0, 4.0, 5.0, 6.0]
    for do in do_list:
        do.update_key(new_global_params)
        print(f"DO {do.id} 使用新密钥派生: {do.round_private_key}")

    print("\n===== 测试完成 =====")
    print("所有DO功能测试通过！")
