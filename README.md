

### 双向验证的FL

##### 1.实现ImprovedPaillier、SafeMul、Threshold工具类。<u>*<!--完成-->*</u>

##### 2.基于工具类完成TA、CSP、DO、Test代码的实现。<!--完成-->

##### 3.Test模拟FL实现的基础上，支持DO掉线、增加SafeMul的使用。*<!--完成-->*

> [!NOTE]
>
> ##### ####完成了基本框架####
>
> ##### ###仔细验证代码，检查功能的正确使用，同时考虑几个问题：###
>
> ##### ###1）模型参数/正交向量组的替换是否可以顺利实现？###
>
> ##### ###2）更换使用真实的训练场景###

##### 4.使用真正的神经网络训练

##### 5.正交向量投毒检测

​	--1）正交向量乘积检验通过、投毒检测通过

​	--2）正交向量乘积检验通过、投毒检测不通过——》投毒检测定位恶意DO

​	--3）正交向量乘积检验不通过、投毒检测通过——》二分查找恶意DO

######################################################################

######################################################################

> [!NOTE]
>
> 二分查找恶意DO？：直接从头开始二分，即，CSP一次向一半的DO发起整个请求，重新构建一半DO的密文、一半DO的安全点积，然后验证。
>
> 
>
> CSP如何验证正交向量乘积正确？
>
> --CSP此时拥有聚合的模型参数、拥有自己的正交向量组
>
> --DO拥有自己的模型参数、自己的正交向量组
>
> 1.CSP把自己的正交向量组【一轮】发给DO，
>
> 2.DO乘自己的模型参数然后发过去密文结果【二轮】和自己的正交向量组*自己的模型参数【明文算】
>
> 3.CSP接收后【三轮】解密，加上另一部分，得到结果。
>
> ##此时，CSP得到了每个DO的模型参数与整个正交向量组的点积结果
>
> 4.CSP发送聚合解密后的明文模型参数，DO乘完发给CSP，CSP解密再与自己的相乘。最后相加。？

![image-20251118155624924](C:\Users\leg\AppData\Roaming\Typora\typora-user-images\image-20251118155624924.png)

```python
# ============== 真实密钥构建 ==============
def _hash_global_params(self, global_params: List[float]) -> int:
    s = str(global_params).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h, "big", signed=False)

def update_key(self, global_params: List[float]) -> None:
    N2 = self.ta.get_N() ** 2
    self.base_private_key = self.ta.get_base_key(self.id)
    h = self._hash_global_params(global_params)
    self.round_private_key = pow(self.base_private_key, h, N2)

# ============== 掉线DO的密钥恢复 ==============
def upload_key_share(self, missing_do_id: int) -> Optional[int]:
    entry = self.ta.do_key_shares[missing_do_id]
    share = entry["shares"].get(self.id)
    return share

# ============== 安全向量内积协议 ==============
CSP：payload = {
    "p": ...,        # 大素数
    "alpha": ...,    # 随机掩码
    "C_all": ...,    # 第一轮中得到的密文集合
}
DO：拿自己的模型向量 b_vector+r 参与第二轮计算D_sums。减去随机数，计算do_part：
def safe_mul_round2_process(self, payload: Dict[str, Any], b_vector: List[float]) -> Dict[str, Any]:
    sip = SafeInnerProduct(precision_factor=self.precision)
    p, alpha, C_all = payload['p'], payload['alpha'], payload['C_all']

    # 生成参数随机掩码 r，并构造带掩码的 b_masked
    r_vec = [self._rng.uniform(-1.0, 1.0) for _ in range(len(b_vector))]
    b_masked = [b + r for b, r in zip(b_vector, r_vec)]

    # 密文态计算（使用 b_masked）
    D_sums = sip.round2_client_process(b_masked, C_all, alpha, p)

    # 明文态计算真实 <b, u_k>（从 <b_masked, u_k> 中减去 <r, u_k>）
    do_part = []
    for u in self.orthogonal_vectors_for_do:
        masked_dot = sum((bi + ri) * ui for bi, ri, ui in zip(b_vector, r_vec, u))
        mask_bias = sum(ri * ui for ri, ui in zip(r_vec, u))
        do_part.append(masked_dot - mask_bias)

    return {'D_sums': D_sums, 'do_part': do_part}

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    
# 1. 准备数据
meta = _get_dataset_meta(dataset_name)
train_dataset = meta["dataset_cls"](
    root=data_root,
    train=True,
    download=True,
    transform=meta["transform"]
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 构建模型
model = SimpleCNN(
    in_channels=meta["in_channels"],
    input_size=meta["input_size"],
    num_classes=meta["num_classes"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 3. 训练循环
model.train()
for data, target in train_loader:
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()

# 4. 导出参数向量
params = []
for p in model.parameters():
    params.append(p.view(-1))
flat_params = torch.cat(params).cpu().numpy().tolist()


# 1) 初始化：生成 Paillier 参数 + R_t + 每个 DO 的 sk
impaillier = ImprovedPaillier(m=num_do, bit_length=bit_length, precision=precision)
N = impaillier.getN()
R_t = impaillier.R_t
n_i = list(impaillier.n_i)            # 每个 DO 的指数
SK_DO = list(impaillier.SK_DO)        # 每个 DO 的基础私钥 sk_i = R_t^{n_i} mod N^2
do_private_keys = {i: SK_DO[i] for i in range(num_do)}
# 注：TA 把 do_private_keys 存好并通过 get_base_key() 提供给 DO

# 2) 对每个 DO 的 sk 进行门限分片（Shamir）
for i in range(num_do):
    shares, prime = split_secret(do_private_keys[i], threshold, num_do)
    # 存储供恢复时使用：do_key_shares[i] = {'shares': {donor: share}, 'prime': prime}
    do_key_shares[i] = {'shares': {j: shares[j+1] for j in range(num_do) if j!=i}, 'prime': prime}
# 注：每个 DO 保留其它 DO 的分片，用于门限恢复

# 3) 每轮结束时旋转 R_t，重算 sk 并重新分片
def rotate_R_t():
    # 生成新的 R_t（与 N 互质）
    while True:
        cand = secrets.randbits(bit_length)
        if cand>1 and math.gcd(cand, N)==1:
            R_t = cand
            break
    # 重新计算 sk_i 并刷新分片
    N2 = N*N
    for i in range(num_do):
        do_private_keys[i] = pow(R_t, n_i[i], N2)
    regenerate_shares()  # 重新 split_secret 并覆盖 do_key_shares
# 注：确保每轮密钥、分片都更新，防止长期重用

# 4) 生成全局正交向量并拆分给 CSP/DO（NumPy 版本）
A = rng.standard_normal((MODEL_SIZE, ORTH_COUNT))   # d × k
Q, _ = np.linalg.qr(A, mode='reduced')              # 列正交（d × k）
U = [Q[:, i].tolist() for i in range(ORTH_COUNT)]   # 全局正交基 U

ratios = rng.random((ORTH_COUNT, MODEL_SIZE))       # 每元素的拆分比例
U_np = np.array(U)                                  # (k, d)
U_csp = (U_np * ratios).tolist()                    # 分给 CSP 的部分
U_do  = (U_np * (1.0 - ratios)).tolist()            # 分给 DO 的部分
# 注：U = U_csp + U_do（按元素相加近似恢复 U），单方不可还原完整 U

# Round 1 (PA)
generate p, α, s; compute s⁻¹ mod p
for each A: C ← s · (α·A + random_c) mod p

# Round 2 (PB)
for each C: D ← Σ (α·B·C + random_r·C_extra) mod p

# Round 3 (PA)
for each D: E ← s⁻¹·D mod p
output inner_product ≈ floor(E / α²)


```



### 投毒检测

防御方案：

**FedAvg**

- 普通联邦平均，无任何防御，作为**不鲁棒基线**。

**Trimmed Mean（TrMean）**

- 典型 **coordinate-wise 鲁棒聚合**。
- 每一维上丢掉若干个最大值和最小值，再对剩下的取平均，目的是过滤掉特别极端的恶意值。

**Geometric median（GeoMed）**

- **model-wise 鲁棒聚合**。
- 找到一个点，让它到所有客户端更新向量的距离和最小（几何中位数），这个点对极端 outlier 不敏感。

**Multi-Krum**

- **距离型、model-wise 鲁棒方法**。
- 计算每个客户端更新与其它更新的距离，把“离大家都不远”的那些更新选出来做平均，试图过滤掉孤立的恶意更新。

**Bulyan**

- 在 Krum/Multi-Krum 的基础上，再做一层坐标-截断，简单说就是：
  1. 用 Krum 反复选出一批“相对可信”的客户端；
  2. 再对这些客户端的各坐标做 trimmed mean。
- 兼具 model-wise + coordinate-wise 的特性。

**DnC（Divide-and-Conquer）** [arXiv](https://arxiv.org/pdf/2409.01435)

- 将客户端更新做聚类，利用“多数诚实、少数恶意”的假设，丢掉异常簇，再在剩下的簇中聚合。

**SignGuard** [arXiv+1](https://arxiv.org/pdf/2409.01435)

- 这篇文章引用的一个 SOTA 防御。
- 核心思想：
  - 使用**方向（sign）+ 模长（magnitude）** 的信息进行聚类和过滤；
  - 尝试识别那些在符号模式上偏离大多数诚实客户端的更新，再结合模长做过滤。

**SparseFed** [arXiv](https://arxiv.org/pdf/2409.01435)

- 典型的 **sparsification-based 防御**。
- 在服务器端对聚合后的模型做剪枝（去掉小参数），结合 clipping 与误差反馈，希望弱化恶意参数的影响。

**LASA（作者提出的方法）** [arXiv](https://arxiv.org/pdf/2409.01435)

- 防御策略可以理解为两步：
  1. **Pre-aggregation sparsification**：对每个客户端更新单独做剪枝，仅保留最重要的参数（每个客户端有各自的掩码）；
  2. **Layer-wise adaptive filtering**：在每一层上，用“方向 + 模长”的指标选择看起来是**诚实的层**来参与聚合，过滤掉可疑层。



# 投毒方案：

### 3.1 Naive attacks（三种）

这些是比较“粗暴”的攻击方式，用来测防御在简单场景下的表现：

1. **Random attack**
   - 恶意客户端不管本地数据，直接上传随机向量当作模型更新（比如高斯随机向量）。
   - 非常容易被任何稍微鲁棒一点的聚合方法识别为 outlier。
2. **Noise attack**
   - 在本地正常更新的基础上，加上很大的噪声（比如高方差高斯噪声），使得上传的更新被严重扰动。
   - 依然比较容易被基于距离或截断的方法滤掉。
3. **Sign-flip attack**
   - 把正常更新的**符号取反并且放大**（例如乘上一个负的大系数），相当于沿着“正确方向”的反方向用力推模型。
   - 比 Random/Noise 稍微“聪明”一点，但形状仍然明显异常。

### 3.2 SOTA attacks（五种）

这五种都是在前人工作中被证明对鲁棒聚合非常强的 **Byzantine 攻击**：[arXiv+1](https://arxiv.org/pdf/2409.01435)

1. **TailoredTrMean（AGR-tailored Trimmed-mean attack）**
   - 出自 [45]，专门“对着 Trimmed Mean 这种聚合器来设计的攻击”。
   - 攻击者精心选择更新值，让自己大部分维度落在“不会被截断”的区间内，但整体方向仍朝着破坏模型的方向偏移，从而骗过 TrMean。
2. **Min-Max attack**
   - 同样来自 [45]。
   - 利用“**最大化/最小化距离**”的思路构造恶意更新，让恶意集体在“被聚合器选中的那一侧”，同时又足够集中，难以被看成 outlier。
   - 对 Multi-Krum、距离型防御特别难搞。
3. **Min-Sum attack**
   - 也是 [45]。
   - 和 Min-Max 类似，目标是最小化到诚实更新的距离和（或者在某些度量下显得“很像诚实更新”），从而在聚合时占大权重，但方向又是有害的。
4. **Lie attack**（来自 [5]）
   - 攻击者先“估计”诚实客户端更新的大致均值方向，再在这个方向上 **适度夸大或扭曲** 自己的更新。
   - 这样既能在几何上看起来像多数诚实客户端，又能慢慢把整体梯度推向错误的方向。
5. **ByzMean attack**（来自 SignGuard 的论文 [58]）
   - 利用聚合器的结构（特别是基于 sign/clustering 的防御）设计的攻击。
   - 通过让恶意更新和诚实更新在符号模式上高度相似，但在模长或局部维度上有策略性的偏移，来逃避基于 sign 的过滤。
