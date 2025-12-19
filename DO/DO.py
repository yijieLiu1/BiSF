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
import time
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import hashlib
from typing import List, Optional, Dict, Any
from utils.ImprovedPaillier import ImprovedPaillier
from utils.SafeMul import SafeInnerProduct

# 可选 NumPy 加速（大规模参数时建议安装）
try:
    import numpy as np
    _NP_AVAILABLE = True
except Exception:
    _NP_AVAILABLE = False

# PyTorch + torchvision，用于真实 CNN 训练
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms, models
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

class DO:
    """Data Owner（数据拥有方）"""
    _DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
    _DEFAULT_BATCH_SIZE = 64
    _DEFAULT_MAX_BATCHES = 100  # 每轮训练最多使用的 batch 数，避免过慢
    _data_loader_cache: Dict[str, DataLoader] = {}
    # 训练数据按 DO 均分后的索引缓存：key -> List[List[int]]（长度=num_do，每个元素是一份 shard）
    _sharded_indices_cache: Dict[str, List[List[int]]] = {}

    def __init__(
        self,
        do_id: int,
        ta,
        model_size: int = 10000,
        precision: int = 10 ** 6,
        rng_seed: Optional[int] = None,
        model_name: str = "lenet",
        dataset_name: str = "cifar10",
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
        attack_config: Optional[Dict[str, Any]] = None,
        partition_mode: str = "iid",
    ):
        """
        初始化DO
        Args:
            do_id: DO的唯一标识符
            ta: 可信机构TA的实例
            model_size: 模型参数大小(几万维,默认50000)
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

        # 训练配置
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size if batch_size is not None else self._DEFAULT_BATCH_SIZE
        self.max_batches = max_batches if max_batches is not None else self._DEFAULT_MAX_BATCHES
        self.data_root = self._DATA_ROOT
        self.attack_config: Dict[str, Any] = attack_config or {}
        meta_info = self._get_dataset_meta(self.dataset_name)
        self.num_classes: int = meta_info.get("num_classes", 10)
        # 用于“翻转到不存在标签”时在交叉熵中忽略这些样本
        self._ignore_label: int = max(255, self.num_classes)
        # 数据划分模式：iid / mild / extreme
        mode = (partition_mode or "iid").lower()
        if mode not in ("iid", "mild", "extreme"):
            mode = "iid"
        self.partition_mode: str = mode

        # 持久化模型/优化器（仅在 resnet 系列下复用，避免 BN 统计重置）
        self._persist_model = None
        self._persist_optimizer = None
        self._persist_scheduler = None
        # 每个 DO 自己的 DataLoader（使用各自固定的数据子集）
        self._train_loader: Optional[DataLoader] = None

        # 初始化 CNN 相关组件
        self._check_torch_available()
        self.device = torch.device("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            print(f"[DO {self.id}] 使用 GPU: {torch.cuda.get_device_name(self.device)}")
        poison_seed = self.attack_config.get("poison_seed", rng_seed if rng_seed is not None else (12345 + do_id))
        self._torch_rng = torch.Generator(device=self.device)
        self._torch_rng.manual_seed(int(poison_seed) & 0xFFFFFFFF)
        print(f"[DO {self.id}] 初始化完成，模型大小: {self.model_size}, 精度: {self.precision}")

        # 对 ResNet 系列，在设备初始化之后再创建持久化模型/优化器，避免 device 未就绪
        if self.model_name.startswith("resnet") and self._persist_model is None:
            self._init_persistent_model()

    def _check_torch_available(self) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "[DO] 需要安装 PyTorch 与 torchvision 才能使用 MNIST CNN 训练，请执行 `pip install torch torchvision`"
            )

    @classmethod
    def _get_dataset_meta(cls, dataset_name: str) -> Dict[str, Any]:
        """根据数据集名称返回元信息与 transform"""
        name = dataset_name.lower()
        if name == "mnist":
            return {
                "name": "mnist",
                "dataset_cls": datasets.MNIST,
                # 标准 LeNet-5 使用 1×32×32 输入，这里通过 Pad(2) 将 28×28 扩展为 32×32
                "in_channels": 1,
                "input_size": 32,
                "num_classes": 10,
                "transform": transforms.Compose([
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
            }
        if name == "cifar10":
            return {
                "name": "cifar10",
                "dataset_cls": datasets.CIFAR10,
                "in_channels": 3,
                "input_size": 32,
                "num_classes": 10,
                "transform": transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ]),
            }
        if name == "cifar100":
            return {
                "name": "cifar100",
                "dataset_cls": datasets.CIFAR100,
                "in_channels": 3,
                "input_size": 32,
                "num_classes": 100,
                "transform": transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ]),
            }
        raise ValueError(f"不支持的数据集: {dataset_name}")

    @classmethod
    def _get_data_loader(cls, dataset_name: str, batch_size: int, data_root: str) -> DataLoader:
        """返回缓存的数据加载器（使用官方二进制数据 + 80% 训练索引）"""
        name = dataset_name.lower()
        cache_key = f"{name}_bs{batch_size}_root{data_root}"
        if cache_key in cls._data_loader_cache:
            return cls._data_loader_cache[cache_key]
        meta = cls._get_dataset_meta(dataset_name)
        os.makedirs(data_root, exist_ok=True)
        dataset = meta["dataset_cls"](
            root=data_root,
            train=True,
            download=True,
            transform=meta["transform"]
        )
        # 加载拆分索引
        split_dir = os.path.join(data_root, "splits")
        train_idx_path = os.path.join(split_dir, f"{name}_train_idx.json")
        if not os.path.exists(train_idx_path):
            raise FileNotFoundError(f"[DO] 未找到训练索引 {train_idx_path}，请先运行 split_datasets.py 生成 80/20 拆分索引")
        import json
        with open(train_idx_path, "r", encoding="utf-8") as f:
            train_idx = json.load(f)
        subset = Subset(dataset, train_idx)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        cls._data_loader_cache[cache_key] = loader
        return loader

    @classmethod
    def _build_label_buckets(
        cls,
        dataset,
        full_idx_shuffled: List[int],
        num_classes: int,
    ) -> List[List[int]]:
        """根据标签将索引分桶，用于 non-IID 划分。"""
        # 获取标签序列（兼容 MNIST/CIFAR 的常用属性）
        if hasattr(dataset, "targets"):
            labels_seq = list(dataset.targets)
        elif hasattr(dataset, "train_labels"):
            labels_seq = list(dataset.train_labels)
        else:
            labels_seq = []
            for i in range(len(dataset)):
                _, y = dataset[i]
                labels_seq.append(int(y))
        buckets: List[List[int]] = [[] for _ in range(num_classes)]
        for idx in full_idx_shuffled:
            y = int(labels_seq[idx])
            if 0 <= y < num_classes:
                buckets[y].append(idx)
        return buckets

    @classmethod
    def _build_shards_iid(
        cls,
        full_idx: List[int],
        num_do: int,
        seed: int = 2025,
    ) -> List[List[int]]:
        """IID 划分：整体打乱后按 DO 等分。"""
        rnd = random.Random(seed)
        full_idx_shuffled = list(full_idx)
        rnd.shuffle(full_idx_shuffled)
        total = len(full_idx_shuffled)
        base = total // num_do
        rem = total % num_do
        shards: List[List[int]] = []
        start = 0
        for i in range(num_do):
            extra = 1 if i < rem else 0
            end = start + base + extra
            shards.append(full_idx_shuffled[start:end])
            start = end
        return shards

    @classmethod
    def _build_shards_mild_non_iid(
        cls,
        full_idx: List[int],
        dataset,
        num_do: int,
        num_classes: int,
        alpha: float = 0.75,
        seed: int = 2025,
    ) -> List[List[int]]:
        """
        轻度 non-IID：每个 DO 偏好 2 个标签（样本约 75% 来自偏好标签，25% 来自其他标签），
        同时不同 DO 的偏好标签不同。
        """
        rnd = random.Random(seed)
        full_idx_shuffled = list(full_idx)
        rnd.shuffle(full_idx_shuffled)
        buckets = cls._build_label_buckets(dataset, full_idx_shuffled, num_classes)
        # 每个 bucket 用指针避免重复使用样本
        bucket_ptr = [0] * num_classes

        total = len(full_idx_shuffled)
        base = total // num_do
        rem = total % num_do
        shards: List[List[int]] = []

        for i in range(num_do):
            target_size = base + (1 if i < rem else 0)
            if target_size <= 0:
                shards.append([])
                continue
            # 偏好两个标签
            primary_labels = [i % num_classes, (i + 1) % num_classes]
            primary_quota = int(alpha * target_size)
            assigned: List[int] = []

            # 先从偏好标签中取
            for lab in primary_labels:
                while primary_quota > 0 and bucket_ptr[lab] < len(buckets[lab]):
                    assigned.append(buckets[lab][bucket_ptr[lab]])
                    bucket_ptr[lab] += 1
                    primary_quota -= 1
                if primary_quota == 0:
                    break

            # 若偏好桶不够，再从所有标签中补足
            remaining = target_size - len(assigned)
            if remaining > 0:
                for lab in range(num_classes):
                    while remaining > 0 and bucket_ptr[lab] < len(buckets[lab]):
                        assigned.append(buckets[lab][bucket_ptr[lab]])
                        bucket_ptr[lab] += 1
                        remaining -= 1
                    if remaining == 0:
                        break

            shards.append(assigned)

        return shards

    @classmethod
    def _build_shards_extreme_non_iid(
        cls,
        full_idx: List[int],
        dataset,
        num_do: int,
        num_classes: int,
        seed: int = 2025,
    ) -> List[List[int]]:
        """
        极端 non-IID：每个 DO 只拥有少数几个标签，其余标签完全缺失。
        实现方式：
        - 先为每个标签分桶
        - 再将标签轮流分配给 DO（标签 -> DO 的映射）
        - 每个 DO 只从其负责的标签桶中取样本
        """
        rnd = random.Random(seed)
        full_idx_shuffled = list(full_idx)
        rnd.shuffle(full_idx_shuffled)
        buckets = cls._build_label_buckets(dataset, full_idx_shuffled, num_classes)
        bucket_ptr = [0] * num_classes

        # 标签 -> DO 的分配表（轮流分配）
        labels_for_do: List[List[int]] = [[] for _ in range(num_do)]
        cur = 0
        for lab in range(num_classes):
            labels_for_do[cur].append(lab)
            cur = (cur + 1) % num_do

        total = len(full_idx_shuffled)
        base = total // num_do
        rem = total % num_do
        shards: List[List[int]] = []

        for i in range(num_do):
            target_size = base + (1 if i < rem else 0)
            if target_size <= 0:
                shards.append([])
                continue
            allowed_labels = labels_for_do[i] if labels_for_do[i] else list(range(num_classes))
            assigned: List[int] = []
            remaining = target_size

            # 仅从自己负责的标签桶中取
            while remaining > 0:
                progress = False
                for lab in allowed_labels:
                    if remaining <= 0:
                        break
                    if bucket_ptr[lab] < len(buckets[lab]):
                        assigned.append(buckets[lab][bucket_ptr[lab]])
                        bucket_ptr[lab] += 1
                        remaining -= 1
                        progress = True
                if not progress:
                    # 自己负责的标签不够用了，停止（少量样本缺失可以接受）
                    break

            shards.append(assigned)

        return shards

    def _get_data_loader_for_do(self, dataset_name: str, batch_size: int, data_root: str) -> DataLoader:
        """
        为当前 DO 返回“固定子数据集”的 DataLoader：
        - 先根据全局 train_idx 做一次随机打乱
        - 再按 TA.num_do 均分成 num_do 份
        - 第 i 个 DO（self.id=i）永远只使用自己这一份索引
        这样可以模拟典型的联邦场景：每个 DO 拿到一个互不重叠、近似 IID 的数据子集。
        """
        name = dataset_name.lower()
        meta = self._get_dataset_meta(dataset_name)
        os.makedirs(data_root, exist_ok=True)

        # 1) 构建完整训练集（官方二进制 + 统一 transform）
        dataset = meta["dataset_cls"](
            root=data_root,
            train=True,
            download=True,
            transform=meta["transform"],
        )

        # 2) 读取 80% 训练索引（全局）
        split_dir = os.path.join(data_root, "splits")
        train_idx_path = os.path.join(split_dir, f"{name}_train_idx.json")
        if not os.path.exists(train_idx_path):
            raise FileNotFoundError(
                f"[DO] 未找到训练索引 {train_idx_path}，请先运行 split_datasets.py 生成 80/20 拆分索引"
            )
        with open(train_idx_path, "r", encoding="utf-8") as f:
            full_idx = json.load(f)

        # 3) 按 DO 数量划分索引（IID / mild-non-IID / extreme-non-IID，带缓存）
        num_do = getattr(self.ta, "num_do", 1)
        mode = getattr(self, "partition_mode", "iid")
        shard_cache_key = f"{name}_mode{mode}_n{num_do}_root{data_root}"
        if shard_cache_key not in DO._sharded_indices_cache:
            num_classes = meta["num_classes"]
            if mode == "mild":
                shards = DO._build_shards_mild_non_iid(
                    full_idx, dataset, num_do=num_do, num_classes=num_classes, alpha=0.75, seed=2025
                )
            elif mode == "extreme":
                shards = DO._build_shards_extreme_non_iid(
                    full_idx, dataset, num_do=num_do, num_classes=num_classes, seed=2025
                )
            else:
                shards = DO._build_shards_iid(full_idx, num_do=num_do, seed=2025)
            DO._sharded_indices_cache[shard_cache_key] = shards

        shards = DO._sharded_indices_cache[shard_cache_key]
        do_id = int(self.id) if hasattr(self, "id") else 0
        if do_id < 0 or do_id >= len(shards):
            # 兜底：若 id 超界，回退到整集合
            do_indices = full_idx
        else:
            do_indices = shards[do_id]

        subset = Subset(dataset, do_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,   # 仅在自己这份数据内部打乱
            num_workers=0,
            drop_last=True,
        )
        return loader

    class _SimpleMNISTCNN(nn.Module):
        def __init__(self, in_channels: int = 1, input_size: int = 28, num_classes: int = 10):
            super().__init__()
            # 中等规模 CNN，参数量约 5.7 万（会随输入尺寸略有变化）
            self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)    # 1*3*3*16 + 16 ≈ 160
            self.conv2 = nn.Conv2d(16, 32, 3, 1)             # 16*3*3*32 + 32 ≈ 4.6k
            self.pool = nn.MaxPool2d(2)
            flat_dim = self._calc_flat_dim(in_channels, input_size)
            self.fc1 = nn.Linear(flat_dim, 64)              # 800*64 + 64 ≈ 51.2k（MNIST 输入时）
            self.fc2 = nn.Linear(64, num_classes)           # 64*10 + 10 ≈ 650

        def _calc_flat_dim(self, in_channels: int, input_size: int) -> int:
            """根据输入尺寸计算全连接层输入维度"""
            with torch.no_grad():
                x = torch.zeros(1, in_channels, input_size, input_size)
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                return int(x.numel())

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class _LeNet(nn.Module):
        """
        经典 LeNet-5 结构（C1-S2-C3-S4-C5-F6 + 输出层）的简化实现：
        - C1: conv 5x5, out_ch=6（无 padding）
        - S2/S4: avg pool 2x2
        - C3: conv 5x5, out_ch=16
        - C5/F6: 全连接 120 -> 84 -> num_classes
        激活函数使用原版的 Tanh。
        """
        def __init__(self, in_channels: int = 1, input_size: int = 32, num_classes: int = 10):
            super().__init__()
            # C1: 5x5 conv, 无 padding（32x32 -> 28x28，28x28 -> 24x24 对于 28x28 输入）
            self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=0)
            # S2/S4: average pooling
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            # C3: 5x5 conv
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
            # 激活函数（需要在 _calc_flat_dim 中使用，因此提前定义）
            self.act = nn.Tanh()
            # 动态计算展平维度，兼容 28x28 / 32x32 输入
            flat_dim = self._calc_flat_dim(in_channels, input_size)
            self.fc1 = nn.Linear(flat_dim, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def _calc_flat_dim(self, in_channels: int, input_size: int) -> int:
            with torch.no_grad():
                x = torch.zeros(1, in_channels, input_size, input_size)
                x = self.pool(self.act(self.conv1(x)))
                x = self.pool(self.act(self.conv2(x)))
                return int(x.numel())

        def forward(self, x):
            x = self.pool(self.act(self.conv1(x)))
            x = self.pool(self.act(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.fc3(x)
            return x

    class _ResNet18(nn.Module):
        """适配小尺寸输入的 ResNet18（移除初始最大池化，调整首层卷积）"""
        def __init__(self, in_channels: int = 3, num_classes: int = 10):
            super().__init__()
            self.model = models.resnet18(weights=None, num_classes=num_classes)
            # 调整首层，适配小输入与可变通道数
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()

        def forward(self, x):
            return self.model(x)

    class _SmallBasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=True)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = F.relu(out, inplace=True)
            return out

    class _ResNetSmall(nn.Module):
        """
        轻量级 ResNet（约 11 万参数，3 通道，10 类），层配置：C=[12,24,48,64]，每组 1 个 BasicBlock。
        """
        def __init__(self, in_channels: int = 3, num_classes: int = 10):
            super().__init__()
            channels = [12, 24, 48, 64]
            self.in_planes = channels[0]
            self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.layer1 = self._make_layer(channels[0], blocks=1, stride=1)
            self.layer2 = self._make_layer(channels[1], blocks=1, stride=2)
            self.layer3 = self._make_layer(channels[2], blocks=1, stride=2)
            self.layer4 = self._make_layer(channels[3], blocks=1, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(channels[3], num_classes)

        def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
            downsample = None
            if stride != 1 or self.in_planes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            layers = [DO._SmallBasicBlock(self.in_planes, planes, stride, downsample)]
            self.in_planes = planes * DO._SmallBasicBlock.expansion
            for _ in range(1, blocks):
                layers.append(DO._SmallBasicBlock(self.in_planes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x, inplace=True)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    class _ResNet20(nn.Module):
        """标准 CIFAR ResNet20（3x3 卷积堆叠，3-3-3 BasicBlock 配置）"""
        def __init__(self, in_channels: int = 3, num_classes: int = 10):
            super().__init__()
            self.in_planes = 16
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(16, blocks=3, stride=1)
            self.layer2 = self._make_layer(32, blocks=3, stride=2)
            self.layer3 = self._make_layer(64, blocks=3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, num_classes)

        def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
            downsample = None
            if stride != 1 or self.in_planes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            layers = [DO._SmallBasicBlock(self.in_planes, planes, stride, downsample)]
            self.in_planes = planes * DO._SmallBasicBlock.expansion
            for _ in range(1, blocks):
                layers.append(DO._SmallBasicBlock(self.in_planes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x, inplace=True)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    def _build_model(self, dataset_name: str) -> nn.Module:
        """
        构建本地模型：
        - 目前仅保留两类网络：
          * LeNet-5（标准 CNN，用于 MNIST / CIFAR 的轻量实验）
          * ResNet20（标准 CIFAR ResNet-20）
        其他结构暂不启用，避免配置过多带来困惑。
        """
        meta = self._get_dataset_meta(dataset_name)
        in_channels = meta["in_channels"]
        input_size = meta["input_size"]
        num_classes = meta["num_classes"]
        name = self.model_name

        # 将 "cnn"/"mnist_cnn"/"simple_cnn" 统一映射到标准 LeNet-5
        if name in ("lenet", "lenet5", "lenet_cifar", "cnn", "mnist_cnn", "simple_cnn"):
            return self._LeNet(in_channels=in_channels, input_size=input_size, num_classes=num_classes)

        # 统一的 ResNet 入口：仅保留 ResNet20
        if name in ("resnet20", "resnet20_cifar", "resnet"):
            return self._ResNet20(in_channels=in_channels, num_classes=num_classes)

        raise ValueError(f"不支持的模型: {self.model_name}")
    def _init_persistent_model(self) -> None:
        model = self._build_model(self.dataset_name).to(self.device)
        model_seed = 12345 + self.id
        self._reset_model_parameters(model, model_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        # 不用 scheduler，或者后面按轮更新
        scheduler = None
        self._persist_model = model
        self._persist_optimizer = optimizer
        self._persist_scheduler = scheduler


    def _reset_model_parameters(self, model: nn.Module, seed: int) -> None:
        torch.manual_seed(seed)
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _flatten_model_parameters(self, model: nn.Module) -> List[float]:
        with torch.no_grad():
            params = []
            for param in model.parameters():
                params.append(param.detach().view(-1))
            flat = torch.cat(params, dim=0).cpu().numpy().astype(float)
        return flat.tolist()

    def _assign_flat_to_model(self, model: nn.Module, flat: List[float]) -> None:
        """
        将给定的一维参数向量写回到模型参数张量中（按 PyTorch 参数迭代顺序）。
        仅覆盖前 len(flat) 个元素，多余的模型参数保持不变。
        ResNet 系列：BN 的 weight/bias 保持本地，不随全局覆盖（FedBN 思路）。
        """
        with torch.no_grad():
            offset = 0
            for param in model.parameters():
                if self.model_name.startswith("resnet") and isinstance(param, nn.BatchNorm2d):
                    offset += param.numel()
                    continue
                numel = param.numel()
                if offset >= len(flat):
                    break
                take = min(numel, len(flat) - offset)
                new_vals = torch.tensor(flat[offset:offset + take], dtype=param.dtype, device=param.device)
                # 保持其余未覆盖部分的现有初始化（防止全零对称性问题）
                param.view(-1)[:take].copy_(new_vals)
                offset += take

    def _project_params_to_model_size(self, params: List[float]) -> List[float]:
        if len(params) == self.model_size:
            return params
        if len(params) > self.model_size:
            return params[:self.model_size]
        return params + [0.0] * (self.model_size - len(params))

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
        # print(f"[DO {self.id}] 已同步Paillier公参，N={N}, g={self.impaillier.g}, h={self.impaillier.h}, y={self.impaillier.y}")

    # ============== 工具函数 ==============
    def _hash_global_params(self, global_params: List[float]) -> int:
        """
        SHA-256哈希全局参数，返回整数
        对于大向量，使用采样策略提高效率
        Args:
            global_params: 全局模型参数列表
        Returns:
            哈希值的整数表示
        """
        # 对于大向量，使用采样策略：每隔一定间隔采样，确保哈希的稳定性
        if len(global_params) > 10000:
            # 采样策略：取前1000个、中间1000个、后1000个，以及每隔一定间隔的样本
            sample_size = 3000
            step = len(global_params) // sample_size
            sampled = global_params[::max(1, step)][:sample_size]
            # 添加首尾和中间部分确保覆盖
            sampled = global_params[:1000] + global_params[len(global_params)//2:len(global_params)//2+1000] + global_params[-1000:]
            s = str(sampled).encode('utf-8')
        else:
            # 小向量直接转换
            s = str(global_params).encode('utf-8')
        # 计算SHA-256哈希
        h = hashlib.sha256(s).digest()
        # 转换为整数
        hash_int = int.from_bytes(h, byteorder='big', signed=False)
        # print(f"[DO {self.id}] 全局参数哈希值(参数维度{len(global_params)}): {hash_int}")
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

    def _should_label_flip(self, round_idx: int) -> bool:
        cfg = self.attack_config
        if not cfg or cfg.get("attack_type") != "label_flip":
            return False
        attacker = cfg.get("attacker_do_id")
        if attacker is not None and attacker != self.id:
            return False
        rounds = cfg.get("attack_rounds", "all")
        if rounds in ("all", None):
            return True
        if isinstance(rounds, (list, tuple, set)):
            return round_idx in rounds
        return round_idx == rounds

    def _apply_label_flip(self, target: "torch.Tensor") -> int:
        cfg = self.attack_config
        if cfg.get("attack_type") != "label_flip":
            return 0
        if "source_label" not in cfg or "target_label" not in cfg:
            return 0
        source_label = int(cfg.get("source_label"))
        target_label = int(cfg.get("target_label"))
        num_classes = getattr(self, "num_classes", 0)
        ratio = float(cfg.get("poison_ratio", 0.3))
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio <= 0.0 or source_label == target_label:
            return 0
        # 若目标标签超出类别范围：映射到一个随机有效标签（避免直接忽略，制造“混乱”梯度）
        if target_label < 0 or target_label >= num_classes:
            if self._torch_rng is None:
                self._torch_rng = torch.Generator(device=target.device)
                self._torch_rng.manual_seed(12345 + self.id)
            target_label = torch.randint(
                low=0,
                high=max(1, num_classes),
                size=(1,),
                device=target.device,
                generator=self._torch_rng,
            ).item()
            if num_classes > 1 and target_label == source_label:
                target_label = (target_label + 1) % num_classes
        src_indices = (target == source_label).nonzero(as_tuple=False).view(-1)
        if src_indices.numel() == 0:
            return 0
        num_flip = max(1, math.ceil(src_indices.numel() * ratio))
        if self._torch_rng is None:
            self._torch_rng = torch.Generator(device=target.device)
            self._torch_rng.manual_seed(12345 + self.id)
        chosen = src_indices[torch.randperm(src_indices.numel(), device=target.device, generator=self._torch_rng)[:num_flip]]
        target[chosen] = target_label
        return int(chosen.numel())

    # ============== 本地训练（模拟CNN） ==============
    def _local_train(self, global_params: List[float]) -> List[float]:
        """
        模拟CNN神经网络本地训练：
        1. 基于全局参数初始化本地模型
        2. 模拟梯度下降更新（添加随机噪声模拟真实训练）
        3. 返回训练后的完整参数向量（几万维）
        Args:
            global_params: 全局模型参数
        Returns:
            本地训练后的"完整参数"向量（几万维）
        """
        

        self._check_torch_available()
        dataset_name = self.dataset_name
        model_name = self.model_name
        print(f"[DO {self.id}] 开始CNN本地训练，模型大小: {self.model_size}维，数据集 {dataset_name}，模型 {model_name}")

        # 使用全局参数哈希作为随机种子，确保每轮初始化与全局状态关联
        seed_source = int(hashlib.sha256(str(global_params).encode('utf-8')).hexdigest(), 16)
        model_seed = (seed_source + self.id) % (2 ** 32)

        use_persistent = self.model_name.startswith("resnet")
        if use_persistent:
            # 持久化模式：复用同一模型/优化器/调度器，保持 BN running 统计
            model = self._persist_model
            optimizer = self._persist_optimizer
            scheduler = self._persist_scheduler
        else:
            model = self._build_model(dataset_name).to(self.device)
            self._reset_model_parameters(model, model_seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
            step_size = max(1, self.max_batches // 2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        # 将当前收到的全局参数覆盖到模型扁平参数的前K项，确保每轮从全局权重出发（不触碰 BN running 统计）
        try:
            if any(abs(v) > 1e-12 for v in global_params):
                self._assign_flat_to_model(model, list(global_params))
            else:
                print(f"[DO {self.id}] 全局参数近似全零，跳过覆盖，采用良好初始化以避免零梯度。")
        except Exception as e:
            print(f"[DO {self.id}] 应用全局参数初始化失败，将继续使用默认初始化: {e}")
        loss_fn = nn.CrossEntropyLoss(ignore_index=self._ignore_label)

        # 为当前 DO 获取固定子数据集的 DataLoader（首轮构建后复用）
        if self._train_loader is None:
            self._train_loader = self._get_data_loader_for_do(dataset_name, self.batch_size, self.data_root)
        train_loader = self._train_loader
        model.train()
        batches_processed = 0
        current_round = len(self.training_history) + 1
        is_label_flip_round = self._should_label_flip(current_round)

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= self.max_batches:
                break
            data = data.to(self.device)
            target = target.to(self.device)
            if is_label_flip_round:
                target = target.clone()
                flipped = self._apply_label_flip(target)
                if flipped > 0 and batch_idx == 0:
                    print(f"[DO {self.id}] Round {current_round} label_flip 生效，batch0 换标签 {flipped} 条")

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batches_processed += 1
            if batches_processed % 10 == 0:
                print(f"[DO {self.id}] 训练批次: {batches_processed}")

        print(f"[DO {self.id}] CNN本地训练完成，使用批次数: {batches_processed}")

        # Flatten parameters并映射到指定维度
        print(f"[DO {self.id}] 提取并扁平化模型参数...，原始参数大小为 {sum(p.numel() for p in model.parameters())}维")
        local_params_raw = self._flatten_model_parameters(model)
        local_params = self._project_params_to_model_size(local_params_raw)

        print(f"[DO {self.id}] 参数范围: [{min(local_params):.6f}, {max(local_params):.6f}]")
        print(f"[DO {self.id}] 参数前10维: {local_params[:10]}")

        # 记录训练历史
        training_record = {
            'round': len(self.training_history) + 1,
            'global_params': global_params.copy(),
            'local_updates': local_params.copy(),
            'timestamp': time.time()  # 简单时间戳
        }
        self.training_history.append(training_record)
        
        return local_params

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
        # print(f"[DO {self.id}] 加密值 {value} -> 密文: {ciphertext}")
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
        # 同步最新一轮的正交向量（用于随后 SafeMul）
        self._load_orthogonal_vectors()
        
        # 步骤2：本地训练
        updates = self._local_train(global_params)

        
        print(f"DO开始执行模型参数加密...")
        time1=time.time()
        # 步骤3：使用派生私钥加密训练结果（加密的是完整参数而非增量）
        ciphertexts = []
        for i, update in enumerate(updates):
            ciphertext = self._encrypt_value(update)
            ciphertexts.append(ciphertext)
        
        print(f"[DO {self.id}] 加密完成，密文数量: {len(ciphertexts)}")
        # 对于大向量，只打印前几个密文
        if len(ciphertexts) > 5:
            print(f"[DO {self.id}] 密文向量前5个: {ciphertexts[:5]}...")
        else:
            print(f"[DO {self.id}] 密文向量: {ciphertexts}")
        
        time2=time.time()
        print(f"加密总用时{time2-time1}秒")
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

    # ============== SafeMul 第二轮（PB侧） ==============
    def get_last_updates(self) -> List[float]:
        """返回最近一次本地训练的更新向量（用于安全点积）"""
        if not self.training_history:
            return []
        return list(self.training_history[-1].get('local_updates', []))

    def safe_mul_round2_process(self, payload: Dict[str, Any], b_vector: List[float]) -> Dict[str, Any]:
        """
        执行安全点积协议第2轮：
        - 使用 CSP 发送的 (p, alpha, C_all) 与本地模型向量 b_vector 计算 D_sums
        - 同时计算本地部分正交向量组与模型向量的点积（明文），作为 do_part
        返回 { 'D_sums': List[int], 'do_part': List[float] }
        """
        sip = SafeInnerProduct(precision_factor=self.precision)
        p = payload['p']
        alpha = payload['alpha']
        C_all = payload['C_all']

        # Round2: 计算 D_sums
        D_sums = sip.round2_client_process(b_vector, C_all, alpha, p)

        # 本地 DO 部分的明文点积
        do_part: List[float] = []
        for vec in self.orthogonal_vectors_for_do:
            s = 0.0
            for x, y in zip(b_vector, vec):
                s += x * y
            do_part.append(s)

        return {'D_sums': D_sums, 'do_part': do_part}

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
    print("===== [TEST] DO 功能验证（精简输出） =====")

    try:
        from TA.TA import TA
        from CSP.CSP import CSP
    except ImportError as e:
        raise SystemExit(f"导入失败: {e}")

    # 与 DO 的模型规模保持一致：5万维，正交向量组 2048
    ta = TA(num_do=3, model_size=83126, orthogonal_vector_count=2048, bit_length=512)
    csp = CSP(ta)
    do_list = [DO(i, ta,model_size=83126) for i in range(3)]

    baseline_params = [1.0, 2.0, 3.0, 4.0, 5.0]
    for do in do_list:
        do.update_key(baseline_params)
    print("[验证] 密钥派生完成")

    # 单轮训练 + 加密
    # do_cipher_map = {}
    # for do in do_list:
    #     do_cipher_map[do.id] = do.train_and_encrypt(baseline_params)
    #     assert len(do_cipher_map[do.id]) == do.model_size
    # print("[验证] 本地训练 + 加密通过")

    # 门限恢复
    # missing_do_id = 1
    # available = [0, 2]
    # shares = {do_id + 1: do_list[do_id].uploadKeyShare(missing_do_id) for do_id in available}
    # shares = {k: v for k, v in shares.items() if v is not None}
    # recovered_key = ta.recover_do_key(missing_do_id, available)
    # if(recovered_key==ta.get_base_key(missing_do_id)):
    #     print("[验证] 门限恢复通过")

    # 多轮流程：进行 5 轮联邦学习；每轮每个 DO 训练 10 个 batch
    print("\n===== 5. 多轮联邦训练（5轮，每轮10个batch） =====")
    rounds = 2
    gp_history: List[List[float]] = []
    for r in range(1, rounds + 1):
        print(f"\n----- Round {r} -----")
        global_params = csp.broadcast_params()
        gp_history.append(list(global_params))

        round_cipher_map = {do.id: do.train_and_encrypt(global_params) for do in do_list}
        updated_params = csp.round_aggregate_and_update(do_list, round_cipher_map)
        assert len(updated_params) == csp.model_size
    print("[验证] 多轮聚合/解密更新通过")

    # 简单收敛性评估：统计相邻两轮全局参数的 L2 差异与最大绝对差
    print("\n===== 6. 收敛性评估 =====")
    diffs_l2 = []
    diffs_inf = []
    gp_history.append(list(csp.global_params))
    for i in range(1, len(gp_history)):
        a = gp_history[i-1]
        b = gp_history[i]
        sq = 0.0
        m_abs = 0.0
        for x, y in zip(a, b):
            d = y - x
            sq += d * d
            ad = abs(d)
            if ad > m_abs:
                m_abs = ad
        diffs_l2.append(math.sqrt(sq))
        diffs_inf.append(m_abs)
        print(f"Round {i-1} -> {i}: L2差={diffs_l2[-1]:.6f}, 无穷范数差={diffs_inf[-1]:.6f}")

    if len(diffs_l2) >= 4:
        mid = len(diffs_l2) // 2
        front = sum(diffs_l2[:mid]) / max(1, mid)
        back = sum(diffs_l2[mid:]) / max(1, len(diffs_l2) - mid)
        print(f"平均L2差 前半:{front:.6f} 后半:{back:.6f}")
        if back <= front * 0.9:
            print("[评估] 全局参数呈收敛趋势（后半段差距更小）。")
        else:
            print("[评估] 收敛趋势不明显，可增加轮次或调整学习率/批次数。")

    print("\n===== 7. 测试投影计算功能 =====")
    starttime=time.time()
    # 直接计算 DO 模型参数在 TA 正交向量组上的投影（w·U -> 2048维）
    # 选择 DO 0 进行演示
    target_do = do_list[0]
    w = target_do.get_last_updates()
    U = ta.get_orthogonal_vectors()  # List[List[float]]，形状: [orthogonal_count][model_size]
    assert len(w) == ta.MODEL_SIZE
    assert len(U) == ta.ORTHOGONAL_VECTOR_COUNT

    try:
        import numpy as _np
        w_np = _np.array(w, dtype=float)              # (model_size,)
        U_np = _np.array(U, dtype=float)              # (orthogonal_count, model_size)
        proj_np = U_np.dot(w_np)                      # (orthogonal_count,)
        projection = proj_np.astype(float).tolist()
    except Exception:
        # 纯 Python 回退实现
        projection = []
        for u_vec in U:
            s = 0.0
            for x, y in zip(w, u_vec):
                s += x * y
            projection.append(s)
    print(projection)
    assert len(projection) == ta.ORTHOGONAL_VECTOR_COUNT
    endtime=time.time()
    print(f"投影计算时间: {endtime-starttime}秒")

    print(">>> 全部 DO 功能验证通过")
