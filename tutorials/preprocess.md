# BatteryLife 数据集特征提取区间确认报告

> 本文档记录正式编写特征提取代码前，对 BatteryLife 各子数据集充电协议和特征提取区间的分析结论。目标是复现 PINN4SOH（Wang et al., *Nature Communications* 2024）的 16 个充电特征。

---

## 1. PINN4SOH 原始特征定义

PINN4SOH 使用了 4 个数据集，均为 BatteryLife 的子集：

| PINN4SOH 名称 | BatteryLife 对应名称 | 说明 |
|:---|:---|:---|
| XJTU | **XJTU** | 完全一致 |
| TJU | **Tongji** | 完全一致 |
| HUST | **HUST** | 完全一致 |
| MIT | **MATR** | MIT 电池数据集在 BatteryLife 中归入 MATR |

论文特征提取策略：

> "从电池**充满前的一小段数据**中提取统计特征。"

原始 16 个特征：

| # | 特征 | 阶段 |
|:-:|------|------|
| 1 | `voltage_mean` | CC |
| 2 | `voltage_std` | CC |
| 3 | `voltage_kurtosis` | CC |
| 4 | `voltage_skewness` | CC |
| 5 | `CC_Q` | CC |
| 6 | `CC_charge_time` | CC |
| 7 | `voltage_slope` | CC |
| 8 | `voltage_entropy` | CC |
| 9 | `current_mean` | CV |
| 10 | `current_std` | CV |
| 11 | `current_kurtosis` | CV |
| 12 | `current_skewness` | CV |
| 13 | `CV_Q` | CV |
| 14 | `CV_charge_time` | CV |
| 15 | `current_slope` | CV |
| 16 | `current_entropy` | CV |

**关键发现**：PINN4SOH 对不同数据集使用了**不同的电压/电流区间**，并非固定值。具体区间需查阅其预处理代码库 [`Battery-dataset-preprocessing-code-library`](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)。

---

## 2. PINN4SOH 子集的精确区间

以下区间直接摘自 PINN4SOH 预处理代码，**可直接沿用**。

### 2.1 XJTU

代码：`XJTUBatteryClass.py`

| 阶段 | 提取逻辑 | 区间/条件 |
|------|----------|-----------|
| CC | `get_CC_value` 默认 | `voltage <= 4.199V` |
| CC 电压子窗口 | IC 曲线等场景 | `[3.6V, 4.19V]` |
| CV | `get_CV_value` 默认 | `voltage >= 4.199V` |
| SOC 过滤 | 无 | 不限制 |
| CV current 过滤 | 无 | 不限制 |

### 2.2 MIT（MATR）

代码：`MITBatteryClass.py`

| 阶段 | 提取逻辑 | 区间/条件 |
|------|----------|-----------|
| CCCV 粗过滤 | `charge Q >= 0.79 * max(Q)` | **仅取 SOC > 80% 的数据** |
| CC 电压窗口 | `voltage_range=[3.4, 3.595]` | `(3.4V, 3.595V)` |
| CV 分界 | `voltage > 3.595V` | 电压阈值 |
| CV current 窗口 | `current_range=[0.5, 0.1]` | `(0.1A, 0.5A)` |

> MIT 有**前置过滤**——只取 SOC > 80% 的充电数据。这与 XJTU 直接从完整充电过程提取不同。

### 2.3 HUST

代码：`HUSTBatteryClass.py`

**关键事实**：BatteryLife 中的 HUST 数据**不含 `Status` 字段**，无法沿用 PINN4SOH 源码中的阶段标签过滤，必须改用基于电流/电压阈值的自动检测。

**实测数据结构**（以 `HUST_6-4.pkl` 为例）：

| 阶段 | 电流范围 | 电压范围 | 点数 | 说明 |
|------|----------|----------|------|------|
| Fast CC (5C) | `5.497 ~ 5.500` A | `2.68 ~ 3.60` V | ~98–114 | 第一段恒流快充 |
| Slow CC (1C) | `1.099 ~ 1.100` A | `3.36 ~ 3.599` V | ~100–150 | 第二段恒流慢充 |
| CV | `0.055 ~ 0.50` A | `≈ 3.599` V | ~59–78 | 恒压阶段，电流衰减 |

**替代检测逻辑**：

```
Step 1: 取 current > 0 的点（充电段）
Step 2: 过滤 I > 1.5A 的点（排除 5C 快充段）
Step 3: 对剩余数据划分 CC 与 CV
    - CC: I > 0.5A（1C 恒流段，I ≈ 1.1A）
    - CV: V >= 3.595V 且 I <= 0.5A
```

| 阶段 | 提取逻辑 | 区间/条件 |
|------|----------|-----------|
| 前置过滤 | `current <= 1.5A` | 排除 5C 快充段 |
| CC | `current > 0.5A` | 1C 恒流段 |
| CC 电压窗口 | `[3.4, 3.595]` | 与 MIT/MATR 一致 |
| CV | `voltage >= 3.595V` 且 `current <= 0.5A` | 恒压阶段 |
| CV current 窗口 | `[0.05, 0.5]` A | 电流衰减区间 |

> 1. `I > 1.5A` 过滤是对 PINN4SOH 源码中 `Status` 过滤的等效替代。5C 段（I≈5.5A）与 1C+CV 段（I<1.5A）在电流上差异显著，该过滤足够可靠。
> 2. HUST 属于 LTO（钛酸锂）低压体系，截止电压约 3.6V，与 MIT/MATR 的 3.6V 体系一致，因此 CC 电压窗口可沿用 `[3.4, 3.595]`。

### 2.4 TJU（Tongji）

代码：`TongJiBatteryClass.py`

| 阶段 | 提取逻辑 | 区间/条件 |
|------|----------|-----------|
| CC | `control/mA > 0` | BMS 电流控制模式 |
| CV | `control/V > 0` | BMS 电压控制模式 |
| CC 电压窗口 | `[4.0V, 4.2V]` | 硬编码 |
| CV current 窗口 | `[2000, 1000]` mA = `[2.0A, 1.0A]` | 硬编码 |

> Tongji 的 CC/CV 划分基于 **BMS 控制模式字段**（`control/mA` vs `control/V`），而非电压阈值。CV 阶段只取电流在 `[1.0A, 2.0A]` 子区间内的数据。

---

## 3. PINN4SOH 区间设计的核心规律

通用范式：

```
Step 1: 定位充电末端（SOC > 80% 或阶段标签过滤）
    ↓
Step 2: 划分 CC 与 CV（电压阈值或控制模式字段）
    ↓
Step 3: 在 CC 内取电压子窗口（如 [3.4, 3.595]）
    ↓
Step 4: 在 CV 内取电流子窗口（如 [0.1, 0.5]）
    ↓
Step 5: 在子窗口内计算 mean/std/kurtosis/skewness/entropy/slope
```

各数据集参数差异：

| 数据集 | 前置过滤 | CC/CV 划分 | CC 电压窗口 | CV 电流窗口 |
|--------|----------|------------|-------------|-------------|
| **XJTU** | 无 | `voltage <= 4.199` | `[3.6, 4.19]` | 无（全 CV） |
| **MATR (MIT)** | `SOC > 80%` | `voltage > 3.595` | `[3.4, 3.595]` | `[0.1, 0.5]` A |
| **HUST** | `I > 1.5A` 过滤 | `voltage >= 3.595` | `[3.4, 3.595]` | `[0.05, 0.5]` A |
| **Tongji (TJU)** | 无 | `control/mA` vs `control/V` | `[4.0, 4.2]` | `[1.0, 2.0]` A |

---

## 4. BatteryLife 其他数据集的区间推断

对非 PINN4SOH 子集的 14 个数据集，沿用上述通用范式，结合各数据集充电协议推断。

### 4.1 分类总览

| 类别 | 数据集 | 化学体系 | 截止电压 | 充电协议 | CV |
|------|--------|----------|----------|----------|----|
| A. 标准 CC-CV (4.2V) | CALCE, ISU_ILCC, MICH, MICH_EXP, SDU, UL_PUR | NCM/NCA/LCO | ~4.2V | CC-CV | 有 |
| B. 高压 CC-CV (4.3V+) | CALB, HNEI, Stanford, Stanford_2 | 高压 NCM/NMC | 4.3V~4.4V | CC-CV | 有 |
| C. LFP CC-CV (3.6V) | SNL | LFP | 3.6V | CC-CV | 有 |
| D. 多阶段/特殊充电 | RWTH | NMC | 3.9V | CC-CV | 有 |
| E. 纯恒流（无 CV） | NA-ion, ZN-coin | Na-ion / Zn | 4.0V / 1.8V | 纯 CC | **无** |

### 4.2 各类别详细分析

#### A. 标准 CC-CV（4.2V 截止）

**数据集**：CALCE, ISU_ILCC, MICH, MICH_EXP, SDU, UL_PUR

**共性**：截止电压 4.20V ± 0.01V，与 XJTU/Tongji 同属 NCM/NCA 体系。有明显的 CC 阶段和 CV 阶段。

**区间推荐**：
- 前置过滤：无
- CC/CV 分界：`voltage >= 4.199V`
- CC 电压窗口：参考 XJTU，使用 **`[3.6V, 4.19V]`**
- CV 电流窗口：不限制（取全 CV 阶段）

**特殊注意**：
- **CALCE**：0.5C 充电，CV 阶段很短（约 44 个点），current 统计特征可能因样本点少而噪声大。
- **ISU_ILCC**：SOC 区间为 `[0.46, 1.0]`，充电起始电压较高。建议采用**动态下限**：`max(3.6V, 该循环实际起始电压 + 0.1V)`，避免窗口内有效数据点过少。

#### B. 高压 CC-CV（4.3V ~ 4.4V 截止）

**数据集**：CALB (4.35V), HNEI (4.30V), Stanford / Stanford_2 (4.40V)

**共性**：高压 NCM/NMC 体系。参考 MIT 的映射逻辑（`V_cutoff - 0.2` 到 `V_cutoff - 0.005`），对高压体系做等比例映射。

**区间推荐**：
- CC/CV 分界：`voltage >= V_cutoff - 0.005V`
- CC 电压窗口：**`[V_cutoff - 0.5V, V_cutoff - 0.005V]`**
- CV 电流窗口：不限制

| 数据集 | 截止电压 | CC 电压窗口 | CV 分界 |
|--------|----------|-------------|---------|
| CALB | 4.35V | `[3.85V, 4.345V]` | `>= 4.345V` |
| HNEI | 4.30V | `[3.80V, 4.295V]` | `>= 4.295V` |
| Stanford / Stanford_2 | 4.40V | `[3.90V, 4.395V]` | `>= 4.395V` |

#### C. LFP CC-CV（3.6V 截止）

**数据集**：SNL

LFP 充电平台电压约 3.2V~3.6V。直接沿用 MIT 的精确区间：

- CC 电压窗口：**`[3.4V, 3.595V]`**
- CV 分界：`>= 3.595V`
- CV 电流窗口：不限制

> MATR 也属 3.6V 体系，但 PINN4SOH 对其使用了 `SOC > 80%` 前置过滤。SNL 没有该过滤的代码依据，因此不添加。

#### D. 多阶段/特殊充电

**RWTH**：2C CC 到 3.9V，然后 CV 3.9V。SOC 区间 `[0.2, 0.8]`。

- CC 电压窗口：`[V_cutoff - 0.5, V_cutoff - 0.005]` = **`[3.4V, 3.895V]`**
- CV 分界：`>= 3.895V`

#### E. 纯恒流充电（无 CV 阶段）

**数据集**：NA-ion, ZN-coin

电流在整个充电过程中几乎完全恒定（std < 0.001A），没有 CV 阶段。

- **8 个 CV 相关特征统一置 `0.0`**：current 统计特征（6 个）+ `CV_Q` + `CV_charge_time`（2 个）
- 仅提取 **8 个 CC 特征**
- 保持 16 维特征向量不变，CV 特征位恒为 `0.0`。模型训练时可通过权重学习自动忽略这些常量维度。

| 数据集 | 截止电压 | CC 电压窗口 | CV 特征处理 |
|--------|----------|-------------|-------------|
| NA-ion | 4.0V | `[V_start + 0.1V, 3.99V]` | 全部 8 个 CV 特征置 `0.0` |
| ZN-coin | 1.8V | `[V_start + 0.1V, 1.79V]` | 全部 8 个 CV 特征置 `0.0` |

---

## 5. 各数据集特征提取区间汇总表

| 数据集 | PINN4SOH 子集 | 前置过滤 | CC 电压窗口 | CV 分界 | CV 电流窗口 | 备注 |
|--------|:-------------:|----------|-------------|---------|-------------|------|
| CALB | ❌ | 无 | `[3.85, 4.345]` | `>= 4.345` | 无限制 | 高压 NCM |
| CALCE | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | CV 极短 |
| HNEI | ❌ | 无 | `[3.80, 4.295]` | `>= 4.295` | 无限制 | 高压 NMC |
| HUST | ✅ | `I > 1.5A` 过滤 | `[3.4, 3.595]` | `>= 3.595` | `[0.05, 0.5]` A | LTO 体系，三段式充电 |
| ISU_ILCC | ❌ | 无 | `[max(3.6, V_start+0.1), 4.19]` | `>= 4.199` | 无限制 | SOC 区间 [0.46, 1.0] |
| MATR | ✅ (MIT) | `SOC > 80%` | `[3.4, 3.595]` | `> 3.595` | `[0.1, 0.5]` A | 直接沿用 MIT 精确值 |
| MICH | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 NMC |
| MICH_EXP | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 NMC |
| NA-ion | ❌ | 无 | `[V_start+0.1, 3.99]` | N/A | N/A | 纯恒流，8 个 CV 特征置 0 |
| RWTH | ❌ | 无 | `[3.4, 3.895]` | `>= 3.895` | 无限制 | SOC 区间 [0.2, 0.8] |
| SDU | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 CC-CV |
| SNL | ❌ | 无 | `[3.4, 3.595]` | `>= 3.595` | 无限制 | LFP，沿用 MIT 窗口 |
| Stanford | ❌ | 无 | `[3.90, 4.395]` | `>= 4.395` | 无限制 | 高压 NCM |
| Stanford_2 | ❌ | 无 | `[3.90, 4.395]` | `>= 4.395` | 无限制 | 高压 NCM |
| Tongji | ✅ (TJU) | 无 | `[4.0, 4.2]` | `control/V > 0` | `[1.0, 2.0]` A | 直接沿用 TJU 精确值 |
| UL_PUR | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 NCA |
| XJTU | ✅ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 直接沿用 XJTU 精确值 |
| ZN-coin | ❌ | 无 | `[V_start+0.1, 1.79]` | N/A | N/A | 纯恒流，8 个 CV 特征置 0 |

---

## 6. 关键工程决策

### 6.1 PINN4SOH 子集的实现优先级

4 个 PINN4SOH 子集**必须严格遵循原始代码中的精确区间和前置过滤逻辑**：

1. **MATR (MIT)**：必须实现 `SOC > 80%` 前置过滤。
2. **HUST**：BatteryLife 中无 `Status` 字段，改用 `I > 1.5A` 过滤排除 5C 快充段。
3. **Tongji (TJU)**：必须使用 `control/mA` 和 `control/V` 字段划分 CC/CV。
4. **XJTU**：直接使用电压阈值 `4.199V` 划分，无前置过滤。

### 6.2 CC/CV 自动划分兜底策略

对非 PINN4SOH 子集且缺乏阶段标签的数据集，采用基于数据的自动划分：

```python
def split_cc_cv(voltage, current, v_cutoff):
    """
    基于电压和电流曲线自动划分 CC 和 CV 阶段。
    逻辑：
    1. 找到电压接近截止电压的区域（v > 0.98 * v_cutoff）
    2. 在该区域内，如果电流显著下降，则判定为 CV
    3. 否则视为纯 CC（无 CV）
    """
    near_max = voltage > 0.98 * v_cutoff
    if sum(near_max) < 5:
        return "no_cv"
    i_near_max = current[near_max]
    i_before_max = current[~near_max]
    if len(i_before_max) > 0 and mean(i_near_max) < 0.3 * mean(i_before_max):
        return "has_cv", cc_indices, cv_indices
    else:
        return "no_cv"
```

### 6.3 无 CV 数据集的处理

对 NA-ion 和 ZN-coin：
- 8 个 CV 相关特征统一置 `0.0`。
- 仅提取 8 个 CC 特征。
- 16 维特征向量保持不变，CV 特征位恒为 `0.0`。模型训练时通过权重学习自动忽略。
- 不采用"从全充电过程提取 current stats"的替代方案，以避免引入与 PINN4SOH 不等价的特征语义。

---

## 7. 待验证事项

正式编写代码前，建议对以下 3 个数据集抽取样本做**可视化确认**：

1. **HUST**：确认 `I > 1.5A` 过滤能否干净排除 5C 段，保留的 1C + CV 段与 PINN4SOH 原始结果是否一致。
2. **CALCE**：确认 CV 阶段的真实长度（此前分析显示 CV 仅约 44 个点）。
3. **ISU_ILCC**：确认充电起始电压，验证动态下限 `max(3.6, V_start+0.1)` 是否会导致窗口过窄。
