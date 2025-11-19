from pathlib import Path
import re
path = Path('CSP/CSP.py')
text = path.read_text(encoding='utf-8')
pattern = r"# ============== 正交求和向量相关 ==============.*?# ============== SafeMul: 1\+3轮（PA侧） =============="
replacement = '''# ============== 正交求和向量相关 ==============
    def compute_sum_with_orthogonal_vector(self, params: List[float]) -> float:
        """
        将聚合解密后的 1×n 模型参数与 orthogonal_sumvectors_for_csp 做点积，得到单个标量。
        """
        if not self.orthogonal_sumvectors_for_csp:
            raise ValueError("[CSP] 未加载 orthogonal_sumvectors_for_csp")
        if len(params) != len(self.orthogonal_sumvectors_for_csp):
            raise ValueError(f"[CSP] 参数长度不匹配: params={len(params)}, sum_vec={len(self.orthogonal_sumvectors_for_csp)}")
        return float(sum(p * s for p, s in zip(params, self.orthogonal_sumvectors_for_csp)))

    def compare_consistency(self, *_args, **_kwargs):
        """
        占位接口：用于后续将 compute_sum_with_orthogonal_vector 的结果与其他结果做一致性对比。
        """
        pass

    # ============== SafeMul: 1+3轮（PA侧） =============='''
new_text, count = re.subn(pattern, replacement, text, flags=re.S)
if count != 1:
    raise SystemExit(f"replacement failed, count={count}")
path.write_text(new_text, encoding='utf-8')
