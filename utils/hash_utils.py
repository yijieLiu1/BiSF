from typing import List
import hashlib


def hash_global_params(global_params: List[float]) -> int:
    """
    SHA-256哈希全局参数，返回整数。
    对于大向量，使用采样策略提高效率。
    Args:
        global_params: 全局模型参数列表
    Returns:
        哈希值的整数表示
    """
    if len(global_params) > 10000:
        sample_size = 3000
        step = len(global_params) // sample_size
        sampled = global_params[::max(1, step)][:sample_size]
        sampled = (
            global_params[:1000]
            + global_params[len(global_params) // 2:len(global_params) // 2 + 1000]
            + global_params[-1000:]
        )
        s = str(sampled).encode("utf-8")
    else:
        s = str(global_params).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h, byteorder="big", signed=False)


def hash_text_prefix(text: str, length: int = 8) -> str:
    """对文本做 SHA-256，返回指定长度的十六进制前缀。"""
    if length <= 0:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]
