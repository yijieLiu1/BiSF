# Bi-SF：双向验证的联邦学习原型

本仓库实现了基于 Improved Paillier、SafeMul 与门限恢复的联邦学习原型，并支持正交向量投影的一致性审计与投毒检测。

## 目录结构
- `utils/`：密码学与工具类（`ImprovedPaillier.py`、`SafeMul.py`、`Threshold.py`、`hash_utils.py`）
- `TA/`：可信机构（密钥生成、R_t 轮换、分片）
- `DO/`：数据拥有方（本地训练、加密上传、SafeMul 第2轮、掉线分片提供）
- `CSP/`：中心服务器（聚合解密、审计、SafeMul 第1/3轮）
- `ASP/`：审计服务器（用于密文审计辅助）
- `Test.py`：端到端联邦仿真（含审计与 SafeMul）
- `Train.py`：明文训练/投毒检测基线
- `Train2.py`：另一套明文训练脚本（保持原样）

## 核心机制概览
1) **密钥与轮次派生**
- TA 生成 Paillier 参数与每个 DO 的基础私钥 `sk_i = R_t^{n_i} mod N^2`
- DO 使用全局参数哈希派生轮次私钥：`sk_i^hash mod N^2`
- 哈希逻辑统一在 `utils/hash_utils.py`

2) **掉线与门限恢复**
- TA 对每个 `sk_i` 做 Shamir 分片
- DO 掉线时，在线 DO 提供分片，CSP 进行恢复

3) **SafeMul 安全向量内积**
- CSP 第1轮加密正交向量组并发送
- DO 第2轮计算 `D_sums` + 明文 `do_part`
- CSP 第3轮解密并合并得到投影向量
- Test/Train 已对 SafeMul 第1轮做缓存以节省时间

4) **正交投影一致性审计**
- CSP 对聚合结果与投影求和进行一致性判断
- 不一致时对 DO 逐一审计（可配置阈值）

## 快速开始
### 1. 创建环境（可选）
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

### 2. 运行联邦仿真（推荐）
```bash
python Test.py
```

### 3. 运行明文训练基线
```bash
python Train.py
```

## 常用参数（Test.py / Train.py）
### 模型与数据
- `--model-name`：`lenet` / `resnet20` / `cnn`
- `--dataset-name`：`mnist` / `cifar10` / `cifar100`
- `--model-size`：模型参数维度
- `--train-batch-size` / `--train-max-batches`

### 正交向量
- `--orthogonal-vector-count`
- `--refresh-orthogonal-each-round`：每轮刷新正交向量组

### SafeMul 分块
- `--safe-mul-block-size`：`<=0` 表示不分块

### 审计与投毒检测
- `--enable-all-detection`：是否开启检测
- `--detection-methods`：选择方案  
  支持 `all` 或 `multi,geo,cluster`（逗号分隔）
- `--audit-round` / `--audit-do-id`
- `--audit-simulate-dropout` / `--audit-simulate-mismatch`

### 典型示例
只用 Multi-Krum：
```bash
python Test.py --detection-methods multi
```

只用 GeoMedian：
```bash
python Train.py --detection-methods geo
```

全部检测：
```bash
python Test.py --detection-methods all
```

## 调试建议
- `cipher_sum` 异常且所有 DO 相同：优先检查 `params_hash`、`sum_weights`、`rt_pow` 是否一致
- 审计中负权重求逆失败：检查密文是否可逆（`gcd(c, N^2) == 1`）
- DO/CSP/ASP hash 不一致：确保都使用 `utils/hash_utils.py`

## 说明
本项目偏研究原型，默认参数较大（如正交向量数量、模型维度），运行时间较长。  
可通过降低 `--orthogonal-vector-count`、`--train-max-batches`、`--safe-mul-block-size` 进行加速。
