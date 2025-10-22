# Project: paillier_threshold_safemul_tools (three separate files)
# --------------------------------------------------
# This canvas contains three separate Python modules. Save each section below
# into its own file with the indicated filename.
#
# Files (copy each section into a separate .py file):
#   1) threshold.py
#   2) improved_paillier.py
#   3) safe_mul_test.py
#
# After saving, run the demo with:
#   python safe_mul_test.py
#
# --------------------------------------------------

# ===================== FILE: threshold.py =====================
"""
threshold.py

A compact, educational Shamir secret-sharing implementation.
Functions:
  - split_secret(secret, threshold, n_shares, prime=None) -> (shares_dict, prime)
  - reconstruct(shares_dict, prime) -> secret

Notes:
  - This is for testing / educational use only (not production-grade crypto).
"""
from typing import Dict, Tuple
import random
import secrets


def _egcd(a: int, b: int):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _modinv(a: int, m: int) -> int:
    g, x, y = _egcd(a % m, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


def _eval_polynomial(coeffs: list, x: int, p: int) -> int:
    result = 0
    for power, coef in enumerate(coeffs):
        result = (result + coef * pow(x, power, p)) % p
    return result


def _is_prime(n: int, k: int = 12) -> bool:
    if n <= 3:
        return n == 2 or n == 3
    if n % 2 == 0:
        return False
    # write n-1 as d*2^s
    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def _next_prime(start: int) -> int:
    if start <= 2:
        return 2
    candidate = start if start % 2 == 1 else start + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate

#接收秘密、门限值、分片值、一个素数。返回分片的结果，以及当前使用的素数prime
def split_secret(secret: int, threshold: int, n_shares: int, prime: int = None) -> Tuple[Dict[int, int], int]:
    """
    Split `secret` into `n_shares` shares with reconstruction threshold `threshold`.
    Returns (shares_dict, prime). `shares_dict` maps x->y (1-based x values).

    If `prime` is None a prime > max(secret, n_shares) will be chosen.
    """
    if threshold < 1 or n_shares < 1 or threshold > n_shares:
        raise ValueError("invalid threshold/n_shares")

    if prime is None:
        prime = _next_prime(max(secret, n_shares) + 1)
    else:
        if prime <= max(secret, n_shares):
            raise ValueError("prime must be larger than secret and n_shares")

    coeffs = [secret] + [secrets.randbelow(prime) for _ in range(threshold - 1)]
    shares: Dict[int, int] = {}
    for i in range(1, n_shares + 1):
        shares[i] = _eval_polynomial(coeffs, i, prime)
    return shares, prime


def recover_secret(shares: Dict[int, int], prime: int) -> int:
    """
    Reconstruct secret from `shares` (mapping x->y) using Lagrange interpolation at x=0.
    """
    if len(shares) == 0:
        raise ValueError("no shares provided")

    x_s = list(shares.keys())
    y_s = [shares[x] for x in x_s]

    secret = 0
    for i, xi in enumerate(x_s):
        yi = y_s[i]
        num = 1
        den = 1
        for j, xj in enumerate(x_s):
            if j == i:
                continue
            num = (num * (-xj)) % prime
            den = (den * (xi - xj)) % prime
        inv_den = _modinv(den, prime)
        lagrange_coeff = (num * inv_den) % prime
        secret = (secret + yi * lagrange_coeff) % prime
    return secret


# Simple module self-test
if __name__ == '__main__':
    SECRET = 123456789
    THRESHOLD = 3
    N_SHARES = 5
    shares, prime = split_secret(SECRET, THRESHOLD, N_SHARES)
    print(f"prime = {prime}")
    print("shares:")
    for k, v in shares.items():
        print(f"  {k} -> {v}")

    # pick a random subset of threshold shares
    picked = dict(random.sample(list(shares.items()), THRESHOLD))
    rec = recover_secret(picked, prime)
    print("reconstructed:", rec)



