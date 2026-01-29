# Bi-SF 参数说明（CorrectnessTest / EfficiencyTest）

本仓库已将原 `Train.py` / `Test.py` 调整为：
- `CorrectnessTest.py`：明文训练与检测（正确性/可解释性）
- `EfficiencyTest.py`：加密流程联邦仿真（效率/审计/安全向量内积）

本文只介绍 **命令行参数（parser）** 的使用方式。

---

## 1. EfficiencyTest.py（加密流程联邦仿真）

### 基本参数
- `--num-rounds`：联邦轮次
- `--num-do`：在线 DO 数
- `--model-size`：模型参数维度
- `--model-name`：模型名称（cnn / lenet / resnet20）
- `--dataset-name`：数据集（mnist / cifar10 / cifar100）
- `--partition-mode`：数据划分模式（iid / mild / extreme）
- `--initial-params-path`：初始全局参数文件

### 训练与评估
- `--train-batch-size`：训练 batch
- `--train-max-batches`：每轮最大 batch 数
- `--eval-each-round`：每轮是否评估
- `--eval-batch-size`：评估 batch
- `--eval-batches`：评估 batch 上限
- `--bn-calib-batches`：BN 校准批数（ResNet）

### 正交向量与 SafeMul
- `--orthogonal-vector-count`：正交向量数量
- `--refresh-orthogonal-each-round`：是否每轮刷新正交向量
- `--safe-mul-block-size`：SafeMul 分块大小（<=0 不分块）

### 投毒与攻击
- `--attack-type`：untarget 攻击类型（stealth/random/signflip/lie_stat）
- `--attack-round-untarget`：untarget 攻击轮次（all / 单轮 / 多轮）
- `--attack-lambda`：放大系数
- `--attack-sigma`：random 攻击噪声
- `--attack-do-id`：攻击 DO id（逗号分隔）
- `--attack-rounds`：label flip 轮次（all / 逗号）
- `--source-label` / `--target-label` / `--poison-ratio`
- `--bd-enable` / `--bd-target-label` / `--bd-ratio` / `--bd-trigger-size` / `--bd-trigger-value`

### 投毒检测
- `--enable-all-detection`：是否启用投毒检测
- `--enable-compressed-detect`：启用压缩向量检测（ResNet20）
- `--detection-methods`：检测方法选择  
  取值：`all` / `multi` / `geo` / `cluster` / 组合如 `multi,geo`

### 审计模拟
- `--audit-round`：审计轮次
- `--audit-do-id`：审计 DO id（逗号分隔）
- `--audit-simulate-dropout`：模拟掉线
- `--audit-simulate-mismatch`：SafeMul 与加密不一致

---

## 2. CorrectnessTest.py（明文训练与检测）

### 基本参数
- `--rounds`：训练轮次
- `--num-do`：DO 数
- `--model-name` / `--dataset-name`
- `--batch-size` / `--max-batches`
- `--partition-mode`：iid / mild / extreme

### 投毒与攻击
- `--poison-do-id`：攻击 DO 列表（逗号）
- `--source-label` / `--target-label` / `--poison-ratio`
- `--attack-type`：untarget 攻击类型（stealth/random/signflip/lie_stat）
- `--attack-round-untarget`：触发轮次（all / 单轮 / 多轮）
- `--attack-lambda` / `--attack-sigma`
- `--bd-enable` / `--bd-target-label` / `--bd-ratio` / `--bd-trigger-size` / `--bd-trigger-value`

### 投毒检测
- `--enable-all-detection`
- `--enable-compressed-detect`
- `--proj-block-size`：正交投影分块
- `--refresh-orthogonal-each-round`
- `--detection-methods`：检测方法选择（同上）

### 评估与输出
- `--save-path`：保存目录
- `--initial-params-path`
- `--eval-batch-size` / `--eval-batches`
- `--bn-calib-batches`

---

## 3. 示例

仅启用 Multi-Krum：
```bash
python EfficiencyTest.py --detection-methods multi
```

仅启用 GeoMedian：
```bash
python CorrectnessTest.py --detection-methods geo
```

启用全部检测：
```bash
python EfficiencyTest.py --detection-methods all
```
