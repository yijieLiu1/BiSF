# Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive Sparsified Model Aggregation

# **1. 联邦学习运行设置**

### **1.1 全局训练轮次**

论文在多处实验中使用 **100 轮训练**。
 如图 1（页 7）显示纵轴是 Accuracy / TPR / FPR，横轴是 **Training Rounds 0–100**，表明实验均跑了 100 轮。
 LASA投毒方案

### **1.2 客户端（DO）数量**

论文未固定写死 n，但实验部分明确使用了常见的配置：

- **FMNIST / CIFAR10 / CIFAR100**
   基于 FedAvg 的典型配置通常使用 **n = 100 客户端**，每轮随机选取一部分参与。
   从 ByzMean 攻击表格中观察可知，每轮抽样的 malicious 数量与 25% 攻击比例匹配。
- **FEMNIST、Shakespeare**
   这些数据集本身是**天然划分的 Non-IID 多客户端数据集**（Leaf benchmark）
   论文明确引用了 LEAF 设置（页 6）。
   LASA投毒方案

### **1.3 恶意客户端比例**

论文默认攻击比例为：

- **25% 恶意客户端**（页 6）
   “The default attack ratio is set to 25%.”
   LASA投毒方案

并测试了：

- **5%, 10%, 15%, 20%, 25%, 30%**
   如图 2（页 8）



# 2.**每个 DO 的神经网络结构**

论文使用了 3 类模型（页 6–7）：
 LASA投毒方案

------

## **2.1 CNN（用于 FMNIST）**

结构来自表 1（FMNIST CNN）：

- Conv2d → ReLU → MaxPool
- Conv2d → ReLU → MaxPool
- Flatten
- FC → ReLU
- FC（输出 10 类）

⚠ 与 LeNet-5 结构非常接近。

------

## **2.2 ResNet-18（用于 CIFAR-10 与 CIFAR-100）**

论文明确引用 ResNet18（He et al. 2016）作为 backbone（表 1）。
 结构包括：

- conv1
- 4 个 residual block 组
- global average pooling
- fc 输出层（10 类或 100 类）

------

## **2.3 RNN（Shakespeare）**

使用 FedAvg 经典配置（由 LEAF 提供）：

- 2 层 LSTM（内含 256 隐层）
- 后接全连接层输出下一个字符预测

（模型来自 LEAF 数据集 benchmark，论文引用 LEAF 文献 33）。

# **3. 数据集**

论文使用 5 个数据集：
 见页 6 实验设置表述：
 LASA投毒方案

| 数据集          | 性质                          | 用途               |
| --------------- | ----------------------------- | ------------------ |
| **FMNIST**      | 图像分类（10 类，黑白）       | IID & Non-IID 实验 |
| **FEMNIST**     | 字母/数字手写体，天然 Non-IID | Non-IID            |
| **CIFAR-10**    | 彩色图像，10 类               | IID & Non-IID      |
| **CIFAR-100**   | 彩色图像，100 类              | IID & Non-IID      |
| **Shakespeare** | 文本生成，天然 Non-IID        | NLP 实验           |

Non-IID 采用 Dirichlet(α) 分布：

- **α = 0.1, 0.2, 0.3, 0.4, 0.5, 1.0**
   如表 2，页 7



# **4. 攻击方式**

论文研究了 8 类攻击（页 6），包括 naive + SOTA：
 LASA投毒方案

------

## **4.1 Naive Attacks（简单攻击）**

1. **Random Attack**
    上传随机权重向量。
2. **Noise Attack**
    上传高噪声权重。
3. **Sign-Flip Attack**
    将本地梯度方向取反（即 Δ → –Δ）。

------

## **4.2 SOTA Byzantine Attacks（高级拜占庭攻击）**

论文明确列出 5 种（页 6–7）：

1. **Min-Max Attack**
    使恶意更新与良性更新距离最大化，以破坏聚合。
2. **Min-Sum Attack**
    使恶意模型与少数正常模型接近，躲避检测。
3. **Tailored Trimmed-Mean Attack（AGR-tailored）**
    针对 Trimmed-Mean 聚合器的对抗式攻击。
4. **Lie Attack**（来自 Baruch et al. 2019）
    恶意客户端模拟“平均 benign 梯度 ± kσ”的方向，逼近阈值边界。
5. **ByzMean Attack（SignGuard 论文提出）**
    基于方向扰动，使 malicious update 与 benign cluster 的方向错位。

攻击实现细节可在附录 7.3 中找到（页 12–13）。

# **5. 防御方法（对比方案）**

论文共对比 8 种防御方法（页 6）：
 LASA投毒方案

------

## **5.1 baseline：FedAvg**

- 普通平均，无防御。

------

## **5.2 经典鲁棒聚合**

1. **Trimmed Mean（TrMean）**
    对每维坐标裁剪极端值。
2. **GeoMed（Geometric Median）**
    对所有更新求几何中位数。
3. **Multi-Krum**
    选择与其他客户端距离最近的更新。
4. **Bulyan**
    先 Krum 再 Trimmed Mean。

------

## **5.3 现代对抗防御**

1. **Divide-and-Conquer（DnC）**
    层级聚类剔除异常模型。
2. **SignGuard**
    最新 SOTA，基于方向+幅度的多阶段过滤。

## **5.4 论文提出的方法：LASA**

LASA 由三个核心模块：

1. **pre-aggregation sparsification（预稀疏）**
    使用 Top-k 保留大参数，减少攻击面。
2. **magnitude-based layer filtering**
    层级 L2 norm 过滤。
3. **direction-based filtering（PDP方向纯度）**
    使用 PDP 方向度量过滤恶意层。

其整体流程见 Algorithm 1（页 3–4）。

#  **Table 1（IID 设置）实验的全部 FL 参数设置** 

## **1. 客户端数量（DO 数量）**

- **MNIST / FMNIST**：6000 个客户端（IID）【【1:1†LASA投毒方案.pdf†L18-L19】】
- **CIFAR-10 / CIFAR-100**：100 个客户端（IID）【【1:1†LASA投毒方案.pdf†L29-L31】】
- **实际每轮参与训练的客户端数 h：**
  - 对所有数据集：**h = 100**
  - 但 **CIFAR-10/100 特殊：h = 25**【【4:4†LASA投毒方案.pdf†L24-L25】】

➡ **因此 Table 1（FMNIST / CIFAR-10/100 任务）中：
 每轮分别只有 100 或 25 个客户端参与更新。**

# **2. 联邦学习训练的本地更新设置**

论文没有直接给出 **本地 epoch 数量**，但给出了本地优化器超参数：

- **优化器：SGD + Momentum**
  - Momentum：0.9（非 Shakespeare）【【4:4†LASA投毒方案.pdf†L26-L29】】
- **学习率 η：0.1（非 Shakespeare）**【【4:4†LASA投毒方案.pdf†L30-L31】】

论文未写明 local epoch 与 batch size，但按 Federated Averaging 标准实验，通常：

### 👉 默认设置（在类似论文中通用）

- **Local Epoch：1（标准 FedAvg）**
- **Batch Size：32 或 64（论文未说明）**

➡ 因为文中没有任何地方给出 batch size / local epoch，因此你可以在复述实验时注明：

> *“Batch size 与 local epoch 未在论文中明确给出，按 FL 文献惯例可能为 epoch=1，batch=32/64。”*