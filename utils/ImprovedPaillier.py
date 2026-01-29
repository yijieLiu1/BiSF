import math
import secrets
from sympy import nextprime

class ImprovedPaillier:
    # ======================== 初始化 ======================== #
    def __init__(self, m, bit_length=2048, precision=1_000_000):
        """
        Improved Paillier 算法（支持整数和小数，纯Python实现）
        :param m: 数据拥有方数量
        :param bit_length: 密钥长度
        :param precision: 小数放大倍数（默认 1e6）
        """
        self.bit_length = bit_length
        self.m = m
        self.PRECISION = precision  # 放大倍数
        self.key_generation()

    # ======================== 工具函数 ======================== #
    def lcm(self, a, b):
        """最小公倍数"""
        return abs(a * b) // math.gcd(a, b)

    def get_prime(self, bits):
        """生成指定位长的大素数"""
        n = secrets.randbits(bits)
        if n % 2 == 0:
            n += 1
        return nextprime(n)

    def get_random_coprime(self, N, bits=None):
        """生成与 N 互质的随机数"""
        if bits is None:
            bits = self.bit_length
        while True:
            r = secrets.randbits(bits)
            if math.gcd(r, N) == 1 and r != 0:
                return r

    # ======================== 密钥生成 ======================== #
    def key_generation(self):
        p = self.get_prime(self.bit_length // 2)
        q = self.get_prime(self.bit_length // 2)
        self.N = p * q
        self.N2 = self.N * self.N

        self.k = secrets.randbits(self.bit_length)
        self.g = self.N + 1

        while True:
            self.y = secrets.randbits(self.bit_length // 3)
            if math.gcd(self.k, self.y) == 1 and self.y != 0:
                break

        self.h = pow(self.g, self.y, self.N2)
        self.lambda_ = self.lcm(p - 1, q - 1)
        L = (pow(self.g, self.lambda_, self.N2) - 1) // self.N
        self.u = pow(L, -1, self.N)

        self.n_i = []
        total = 0
        for _ in range(self.m - 1):
            val = secrets.randbits(self.bit_length // 2)
            self.n_i.append(val)
            total += val
        self.n_i.append(self.N - total)

        self.R_t = self.get_random_coprime(self.N)
        self.SK_DO = [pow(self.R_t, n_i, self.N2) for n_i in self.n_i]
        self.SK_CSP = self.lambda_

    # ======================== 核心加解密函数 ======================== #
    def encrypt(self, x, SK_DO_i):
        """加密，自动支持小数"""
        # 先放大小数
        x_int = int(round(x * self.PRECISION))
        r = secrets.randbits(self.bit_length // 10)
        c = (pow(self.g, x_int, self.N2) * pow(self.h, r, self.N2) * SK_DO_i) % self.N2
        return c

    def aggregate(self, encrypted_data):
        """同态加法聚合"""
        result = 1
        for c in encrypted_data:
            result = (result * c) % self.N2
        return result


    # def decrypt(self, aggregated_data):
    #     """解密，自动缩小小数到原始精度"""
    #     L = (pow(aggregated_data, self.lambda_, self.N2) * self.u - 1) // self.N
    #     x = L % self.N % self.y
    #     if x > self.y // 2:
    #         x = x - self.y
    #     # 缩小到原始精度
    #     return x / self.PRECISION

    def decrypt(self, aggregated_data):
        t = pow(aggregated_data, self.lambda_, self.N2)

    # 自检 1：Paillier L 函数条件（必须成立）
        if (t - 1) % self.N != 0:
            print("[!!] L 条件失败: (c^λ - 1) mod N != 0")
            print("    (t-1)%N =", (t - 1) % self.N)

        L = (t - 1) // self.N          # 先做 L
        m = (L * self.u) % self.N      # 再乘 u（这是标准公式）

        x = m % self.y
        if x > self.y // 2:
            x -= self.y
        return x / self.PRECISION


    # ---------- Getter for TA ----------
    def get_SK_DO(self):
        return self.SK_DO

    def get_R_t(self):
        return self.R_t

    def get_lambda(self):
        return self.lambda_

    def getN(self):
        return self.N




# ======================== 测试案例 ======================== #
if __name__ == "__main__":
    num_DO = 2
    paillier = ImprovedPaillier(m=num_DO, bit_length=512, precision=1_000_000)

    test_cases = [
        (5.123456, -4.654321),    # 支持小数
        (5, -9),                   # 支持整数
        (0.5, 0.25),               # 小数示例
    ]

    print("====== Improved Paillier 算法测试（支持整数/小数） ======\n")

    for case in test_cases:
        encrypted = [paillier.encrypt(val, SK) for val, SK in zip(case, paillier.SK_DO)]
        aggregated = paillier.aggregate(encrypted)
        decrypted = paillier.decrypt(aggregated)

        print(f"原始数据: {case}")
        print(f"明文求和: {sum(case)}")
        print(f"解密结果: {decrypted}")
        print(f"y - 解密结果: {paillier.y / paillier.PRECISION - decrypted}")
        print("--------------------------------------------------")
