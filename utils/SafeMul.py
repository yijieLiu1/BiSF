# ===================== FILE: SafeInnerProduct.py =====================
"""
SafeInnerProduct.py

A Python implementation of a secure inner product protocol inspired by the Java version.

Protocol has 3 steps:
  1. round1_setup_and_encrypt(): PA side - setup parameters and encrypt vector group A
  2. round2_client_process(): PB side - process vector B with received data
  3. round3_decrypt(): PA side - decrypt aggregated results


"""

import time
import random
from typing import List, Tuple
from secrets import randbits
from sympy import nextprime


class SafeInnerProduct:
    def __init__(self, precision_factor: int = 10**6):
        self.precision_factor = precision_factor

    # ========== Round 1: PA side ==========
    def round1_setup_and_encrypt(
        self,
        a_vectors: List[List[float]],
        k1: int = 1024,
        k2: int = 128,
        k3: int = 64
    ) -> Tuple[int, int, List[List[int]], int, int]:
        """
        PA generates parameters and encrypts its vector group.

        Args:
            a_vectors: list of float vectors
            k1: bit length of prime p
            k2: bit length of alpha
            k3: bit length of random c_i

        Returns:
            (p, alpha, C, s, s_inv)
        """
        rnd = random.SystemRandom()
        # generate large prime p
        p = nextprime(randbits(k1))
        alpha = nextprime(randbits(k2))
        s = rnd.randrange(1, p - 1)
        s_inv = pow(s, -1, p)

        C_all = []
        for vec in a_vectors:
            n = len(vec)
            a_ext = list(vec) + [0, 0]  # extend vector by 2
            C = []
            for i in range(n + 2):
                c_i = rnd.randrange(1 << (k3 - 1), 1 << k3)
                if i < n:
                    scaled_val = int(round(a_ext[i] * self.precision_factor))
                    val = (s * (scaled_val * alpha + c_i)) % p
                else:
                    val = (s * c_i) % p
                C.append(val)
            C_all.append(C)

        return p, alpha, C_all, s, s_inv

    # ========== Round 2: PB side ==========
    def round2_client_process(
        self,
        b_vector: List[float],
        C_all: List[List[int]],
        alpha: int,
        p: int,
        k4: int = 64
    ) -> List[int]:
        """
        PB processes received C_all with its vector b.

        Args:
            b_vector: list of floats
            C_all: encrypted vectors from PA
            alpha, p: parameters from PA
            k4: bit length of random r_i

        Returns:
            list of D_sums for each vector
        """
        rnd = random.SystemRandom()
        n = len(b_vector)
        b_ext = list(b_vector) + [0, 0]

        D_sums = []
        for C in C_all:
            D = []
            for i in range(n + 2):
                if i < n:
                    scaled_val = int(round(b_ext[i] * self.precision_factor))
                    val = (scaled_val * alpha * C[i]) % p
                else:
                    r = rnd.randrange(1 << (k4 - 1), 1 << k4)
                    val = (r * C[i]) % p
                D.append(val)
            D_sum = sum(D) % p
            D_sums.append(D_sum)
        return D_sums

    # ========== Round 3: PA side ==========
    def round3_decrypt(
        self,
        D_sums: List[int],
        s_inv: int,
        alpha: int,
        p: int
    ) -> List[float]:
        """
        PA decrypts final inner products.

        Args:
            D_sums: list of aggregated sums from PB
            s_inv: modular inverse of s
            alpha, p: parameters

        Returns:
            list of float inner products
        """
        results = []
        alpha2 = alpha * alpha
        for D_sum in D_sums:
            E = (s_inv * D_sum) % p
            # handle negative numbers
            if E > p // 2:
                print("溢出修正: E > p/2, subtracting p")
                E -= p
            inner = (E - (E % alpha2)) // alpha2
            val = inner / (self.precision_factor * self.precision_factor)
            results.append(val)
        return results


# ===================== TEST =====================
if __name__ == "__main__":
    start_time = time.time()
    sip = SafeInnerProduct(precision_factor=10**6)

    n = 1000       # vector length
    num_vec = 10   # number of vectors

    # generate random vectors for PA and PB
    rnd = random.Random(1)
    a_vectors = [[-100 + 200 * rnd.random() for _ in range(n)] for _ in range(num_vec)]
    b_vector = [-100 + 200 * rnd.random() for _ in range(n)]

    # Round1 (PA)
    p, alpha, C_all, s, s_inv = sip.round1_setup_and_encrypt(a_vectors)
    # Round2 (PB)
    D_sums = sip.round2_client_process(b_vector, C_all, alpha, p)
    # Round3 (PA)
    results = sip.round3_decrypt(D_sums, s_inv, alpha, p)

    # verify correctness
    for idx in range(num_vec):
        plain_inner = sum(a_vectors[idx][j] * b_vector[j] for j in range(n))
        print(f"\nVector {idx+1}:")
        print("安全点积结果:", results[idx])
        print("明文点积结果:", plain_inner)
        rel_err = abs((results[idx] - plain_inner) / (plain_inner if plain_inner != 0 else 1))
        print("相对误差:", rel_err)

    print("\n程序运行时间：", time.time() - start_time, "秒")
