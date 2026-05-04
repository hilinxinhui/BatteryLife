# BatteryLife 数据集特征提取区间确认报告

> 本文件用于记录在正式编写特征提取代码之前，对 BatteryLife 数据集中各子数据集充电协议和特征提取区间的分析结论。
>
> 目标：复现 PINN4SOH（Wang et al., Nature Communications 2024）中的 16 个充电特征。

---

## 1. PINN4SOH 原始特征定义

PINN4SOH 论文中使用了 **4 个数据集**，且这 4 个数据集均为 BatteryLife 的子集：

| PINN4SOH 名称 | BatteryLife 对应名称 | 说明 |
|:---|:---|:---|
| XJTU | **XJTU** | 完全一致 |
| TJU | **Tongji** | 完全一致 |
| HUST | **HUST** | 完全一致 |
| MIT | **MATR** | MIT 电池数据集在 BatteryLife 中归入 MATR |

论文明确指出特征提取策略：

> "A general feature extraction method is designed to extract statistical features from a **short period of data before the battery is fully charged**."

原始 16 个特征列表如下：

| 序号 | 特征名称 | 所属阶段 |
|:---|:---|:---|
| 1 | `voltage mean` | CC 阶段 |
| 2 | `voltage std` | CC 阶段 |
| 3 | `voltage kurtosis` | CC 阶段 |
| 4 | `voltage skewness` | CC 阶段 |
| 5 | `CC Q` | CC 阶段 |
| 6 | `CC charge time` | CC 阶段 |
| 7 | `voltage slope` | CC 阶段 |
| 8 | `voltage entropy` | CC 阶段 |
| 9 | `current mean` | CV 阶段 |
| 10 | `current std` | CV 阶段 |
| 11 | `current kurtosis` | CV 阶段 |
| 12 | `current skewness` | CV 阶段 |
| 13 | `CV Q` | CV 阶段 |
| 14 | `CV charge time` | CV 阶段 |
| 15 | `current slope` | CV 阶段 |
| 16 | `current entropy` | CV 阶段 |

**关键发现**：PINN4SOH 对不同数据集使用了**不同的电压/电流区间**，并非固定值。具体区间需查阅其预处理代码库 [`Battery-dataset-preprocessing-code-library`](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)。

---

## 2. PINN4SOH 子集的精确区间（基于原始代码）

以下区间直接摘自 PINN4SOH 的预处理代码库，**可直接沿用**。

### 2.1 XJTU 数据集

代码文件：`XJTUBatteryClass.py`

| 阶段 | 提取逻辑 | 区间/条件 |
|:---|:---|:---|
| **CC 阶段** | `get_CC_value` 默认 | `voltage <= 4.199V` |
| **CC 电压子窗口** | IC 曲线等场景 | `[3.6V, 4.19V]` |
| **CV 阶段** | `get_CV_value` 默认 | `voltage >= 4.199V` |
| **SOC 过滤** | 无 | 不限制 |
| **CV current 过滤** | 无 | 不限制 |

### 2.2 MIT 数据集（BatteryLife 中对应 **MATR**）

代码文件：`MITBatteryClass.py`

| 阶段 | 提取逻辑 | 区间/条件 |
|:---|:---|:---|
| **CCCV 粗过滤** | `charge Q >= 0.79 * max(Q)` | **仅取充电后 20%（SOC > 80%）的数据** |
| **CC 电压窗口** | `voltage_range=[3.4, 3.595]` | `(3.4V, 3.595V)` |
| **CV 分界** | `voltage > 3.595V` | 基于电压阈值 |
| **CV current 窗口** | `current_range=[0.5, 0.1]` | `(0.1A, 0.5A)` |

**重要说明**：MIT 的特征提取有一个**前置过滤步骤**——只取 SOC > 80% 的充电数据（即充电末期）。这与 XJTU 直接从完整充电过程提取不同。

### 2.3 HUST 数据集

代码文件：`HUSTBatteryClass.py`

**关键事实**：BatteryLife 中的 HUST 数据**不包含 `Status` 字段**，因此无法直接沿用 PINN4SOH 源码中的阶段标签过滤逻辑，必须改用基于电流/电压阈值的自动检测。

**实测数据结构**（以 `HUST_6-4.pkl` 为例）：

| 阶段 | 电流范围 | 电压范围 | 点数 | 说明 |
|:---|:---|:---|:---|:---|
| **Fast CC (5C)** | `5.497 ~ 5.500` A | `2.68 ~ 3.60` V | ~98–114 | 第一段恒流快充 |
| **Slow CC (1C)** | `1.099 ~ 1.100` A | `3.36 ~ 3.599` V | ~100–150 | 第二段恒流慢充 |
| **CV** | `0.055 ~ 0.50` A | `≈ 3.599` V | ~59–78 | 恒压阶段，电流衰减 |

**替代检测逻辑**：

```
Step 1: 取所有 current > 0 的点（充电段）
Step 2: 过滤掉 I > 1.5A 的点（排除 5C 快充段）
Step 3: 对剩余数据按电压/电流划分 CC 与 CV
    - CC 段: I > 0.5A（即 1C 恒流段，I ≈ 1.1A）
    - CV 段: V >= 3.595V 且 I <= 0.5A
```

| 阶段 | 提取逻辑 | 区间/条件 |
|:---|:---|:---|
| **前置过滤** | `current <= 1.5A` | 排除 5C 快充段 |
| **CC 段** | `current > 0.5A` | 1C 恒流段，I ≈ 1.1A |
| **CC 电压窗口** | `[3.4, 3.595]` | 与 MIT/MATR 一致（同属 ~3.6V 体系） |
| **CV 段** | `voltage >= 3.595V` 且 `current <= 0.5A` | 恒压阶段 |
| **CV current 窗口** | `[0.05, 0.5]` A | 电流衰减区间 |

**重要说明**：
1. 上述逻辑是对 PINN4SOH 源码中 `Status` 过滤的**等效替代**。原始代码通过实验日志标签直接定位 CCCV 段；BatteryLife 数据中无此标签，但 5C 段（I≈5.5A）与 1C+CV 段（I<1.5A）在电流上差异显著，因此 `I > 1.5A` 过滤足够可靠。
2. HUST 的电池化学体系属于 **LTO（钛酸锂）** 或类似低压体系，截止电压约 3.6V，与 MIT/MATR 的 3.6V 体系一致，因此 CC 电压窗口可直接沿用 `[3.4, 3.595]`。

### 2.4 TJU 数据集（BatteryLife 中对应 **Tongji**）

代码文件：`TongJiBatteryClass.py`

| 阶段 | 提取逻辑 | 区间/条件 |
|:---|:---|:---|
| **CC 阶段** | `control/mA > 0` | 基于 BMS 控制模式（电流控制） |
| **CV 阶段** | `control/V > 0` | 基于 BMS 控制模式（电压控制） |
| **CC 电压窗口** | `[4.0V, 4.2V]` | 硬编码在 `plot_one_cycle_CCCV` 中 |
| **CV current 窗口** | `[2000, 1000]` mA = `[2.0A, 1.0A]` | 硬编码在 `plot_one_cycle_CCCV` 中 |

**重要说明**：Tongji 的 CC/CV 划分基于 **BMS 控制模式字段**（`control/mA` vs `control/V`），而非电压阈值。且 CV 阶段也只取电流在 `[1.0A, 2.0A]` 子区间内的数据。

---

## 3. PINN4SOH 区间设计的核心规律

通过对比 4 个子集，可以总结出 PINN4SOH 特征提取的通用范式：

```
Step 1: 定位充电末端（SOC > 80% 或 阶段标签过滤）
    ↓
Step 2: 划分 CC 与 CV（电压阈值 或 控制模式字段）
    ↓
Step 3: 在 CC 内取电压子窗口（如 [3.4, 3.595]）
    ↓
Step 4: 在 CV 内取电流子窗口（如 [0.1, 0.5]）
    ↓
Step 5: 在子窗口内计算 mean/std/kurtosis/skewness/entropy/slope
```

**各数据集参数差异**：

| 数据集 | 前置过滤 | CC/CV 划分方式 | CC 电压窗口 | CV 电流窗口 |
|:---|:---|:---|:---|:---|
| **XJTU** | 无 | `voltage <= 4.199` | `[3.6, 4.19]` | 无（全 CV） |
| **MIT (MATR)** | `SOC > 80%` | `voltage > 3.595` | `[3.4, 3.595]` | `[0.1, 0.5]` A |
| **HUST** | `Status` 标签 | `voltage >= 3.595` | `[3.4, 3.595]` (示例) | `[1.0, 1.1]` A (示例) |
| **TJU (Tongji)** | 无 | `control/mA` vs `control/V` | `[4.0, 4.2]` | `[1.0, 2.0]` A |

---

## 4. BatteryLife 其他数据集的区间推断

对于非 PINN4SOH 子集的 14 个数据集，沿用 PINN4SOH 的通用范式，结合各数据集的充电协议做推断。

### 4.1 分类总览

| 类别 | 数据集 | 化学体系 | 截止电压 | 充电协议 | CV 存在性 |
|:---|:---|:---|:---|:---|:---|
| **A. 标准 CC-CV (4.2V)** | CALCE, ISU_ILCC, MICH, MICH_EXP, SDU, UL_PUR | NCM/NCA/LCO | ~4.2V | CC-CV | 有 |
| **B. 高压 CC-CV (4.3V+)** | CALB, HNEI, Stanford, Stanford_2 | 高压 NCM/NMC | 4.3V~4.4V | CC-CV | 有 |
| **C. LFP CC-CV (3.6V)** | SNL | LFP | 3.6V | CC-CV | 有 |
| **D. 多阶段/特殊充电** | RWTH | NMC | 3.9V | CC-CV | 有 |
| **E. 纯恒流（无 CV）** | NA-ion, ZN-coin | Na-ion / Zn | 4.0V / 1.8V | 纯 CC | **无** |

### 4.2 各类别详细分析

#### A. 标准 CC-CV（4.2V 截止）

**包含数据集**：CALCE, ISU_ILCC, MICH, MICH_EXP, SDU, UL_PUR

**共性**：
- 截止电压稳定在 4.20V ± 0.01V，与 XJTU/Tongji 同属 NCM/NCA 体系。
- 有明显的 CC 阶段（电流近似恒定）和 CV 阶段（电压恒定、电流衰减）。

**区间推荐**：
- 前置过滤：无（与 XJTU 一致）。
- CC/CV 分界：`voltage >= 4.199V`。
- CC 电压窗口：参考 XJTU，使用 **`[3.6V, 4.19V]`**。
- CV 电流窗口：参考 XJTU，不限制（取全 CV 阶段）。

**特殊注意——CALCE**：
- 0.5C 充电，CV 阶段很短（约 44 个数据点），电流从 0.55A 衰减到 0.05A。
- current 统计特征（std/kurtosis/skewness/entropy）可能因样本点少而噪声大，但仍可计算。

**特殊注意——ISU_ILCC**：
- SOC 区间为 `[0.46, 1.0]`，充电起始电压较高。
- 建议采用**动态下限**：`max(3.6V, 该循环实际起始电压 + 0.1V)`，避免窗口内有效数据点过少。

#### B. 高压 CC-CV（4.3V ~ 4.4V 截止）

**包含数据集**：CALB (4.35V), HNEI (4.30V), Stanford / Stanford_2 (4.40V)

**共性**：
- 属于高压 NCM/NMC 体系，截止电压高于常规 4.2V。
- 参考 MIT 的逻辑（3.6V 体系使用 `[3.4, 3.595]`，即 `V_cutoff - 0.2` 到 `V_cutoff - 0.005`），对高压体系做等比例映射。

**区间推荐**：
- 前置过滤：无。
- CC/CV 分界：`voltage >= V_cutoff - 0.005V`。
- CC 电压窗口：**`[V_cutoff - 0.5V, V_cutoff - 0.005V]`**。
- CV 电流窗口：不限制（取全 CV 阶段）。

| 数据集 | 截止电压 | CC 电压窗口 | CV 分界 |
|:---|:---|:---|:---|
| CALB | 4.35V | `[3.85V, 4.345V]` | `>= 4.345V` |
| HNEI | 4.30V | `[3.80V, 4.295V]` | `>= 4.295V` |
| Stanford / Stanford_2 | 4.40V | `[3.90V, 4.395V]` | `>= 4.395V` |

#### C. LFP CC-CV（3.6V 截止）

**包含数据集**：SNL

**共性**：
- LFP（磷酸铁锂）电池的充电平台电压约为 3.2V~3.6V，截止电压 3.6V。
- 参考 MIT 的精确区间 **`[3.4, 3.595]`**（MIT 即 MATR，同属 3.6V 体系）。

**区间推荐**：
- 前置过滤：无。
- CC/CV 分界：`voltage >= 3.595V`。
- CC 电压窗口：**`[3.4V, 3.595V]`**（直接沿用 MIT/MATR 的精确值）。
- CV 电流窗口：不限制。

> **注意**：MATR 虽然在 BatteryLife 中也属 3.6V 体系，但 PINN4SOH 对其使用了 `SOC > 80%` 前置过滤。SNL 没有该过滤的代码依据，因此不添加此过滤。

#### D. 多阶段/特殊充电

**D1. RWTH**

- 充电协议：2C CC 到 3.9V，然后 CV 3.9V。
- SOC 区间 `[0.2, 0.8]`，充放电深度受限。
- 参考 MIT 的映射逻辑：`V_cutoff = 3.9V`，则 CC 窗口为 `[V_cutoff - 0.5, V_cutoff - 0.005]` = **`[3.4V, 3.895V]`**。
- CV 分界：`>= 3.895V`。

#### E. 纯恒流充电（无 CV 阶段）

**包含数据集**：NA-ion, ZN-coin

**共性**：
- 电流在整个充电过程中**几乎完全恒定**（std < 0.001A），没有 CV 阶段。
- **全部 8 个 CV 相关特征统一置 `0.0`**：包括 `current mean/std/kurtosis/skewness/slope/entropy`（6 个）以及 `CV Q`、`CV charge time`（2 个）。
- 仅提取 **8 个 CC 特征**：`voltage mean/std/kurtosis/skewness`、`CC Q`、`CC charge time`、`voltage slope`、`voltage entropy`。

> **工程决策**：保持 16 维特征向量不变，CV 特征位恒为 `0.0`。模型训练时可通过权重学习自动忽略这些常量维度。避免为纯 CC 数据集单独设计替代特征，以保持与 PINN4SOH 特征空间的一致性。

**区间推荐**：

| 数据集 | 截止电压 | CC 电压窗口 | CV 特征处理 |
|:---|:---|:---|:---|
| NA-ion | 4.0V | `[V_start + 0.1V, 3.99V]` | **全部 8 个 CV 特征置 `0.0`** |
| ZN-coin | 1.8V | `[V_start + 0.1V, 1.79V]` | **全部 8 个 CV 特征置 `0.0`** |

---

## 5. 各数据集特征提取区间汇总表

| 数据集 | 是否 PINN4SOH 子集 | 前置过滤 | CC 电压窗口 | CV 分界 | CV 电流窗口 | 备注 |
|:---|:---|:---|:---|:---|:---|:---|
| **CALB** | ❌ | 无 | `[3.85, 4.345]` | `>= 4.345` | 无限制 | 高压 NCM |
| **CALCE** | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | CV 极短 |
| **HNEI** | ❌ | 无 | `[3.80, 4.295]` | `>= 4.295` | 无限制 | 高压 NMC |
| **HUST** | ✅ | `I > 1.5A` 过滤（排除 5C） | `[3.4, 3.595]` | `>= 3.595` | `[0.05, 0.5]` A | LTO 体系，三段式充电 |
| **ISU_ILCC** | ❌ | 无 | `[max(3.6, V_start+0.1), 4.19]` | `>= 4.199` | 无限制 | SOC 区间 [0.46, 1.0] |
| **MATR** | ✅ (MIT) | `SOC > 80%` | `[3.4, 3.595]` | `> 3.595` | `[0.1, 0.5]` A | 直接沿用 MIT 精确值 |
| **MICH** | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 NMC |
| **MICH_EXP** | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 NMC |
| **NA-ion** | ❌ | 无 | `[V_start+0.1, 3.99]` | N/A | N/A | 纯恒流，**8 个 CV 特征置 0** |
| **RWTH** | ❌ | 无 | `[3.4, 3.895]` | `>= 3.895` | 无限制 | SOC 区间 [0.2, 0.8] |
| **SDU** | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 CC-CV |
| **SNL** | ❌ | 无 | `[3.4, 3.595]` | `>= 3.595` | 无限制 | LFP，沿用 MIT 窗口 |
| **Stanford** | ❌ | 无 | `[3.90, 4.395]` | `>= 4.395` | 无限制 | 高压 NCM |
| **Stanford_2** | ❌ | 无 | `[3.90, 4.395]` | `>= 4.395` | 无限制 | 高压 NCM |
| **Tongji** | ✅ (TJU) | 无 | `[4.0, 4.2]` | `control/V > 0` | `[1.0, 2.0]` A | 直接沿用 TJU 精确值 |
| **UL_PUR** | ❌ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 标准 NCA |
| **XJTU** | ✅ | 无 | `[3.6, 4.19]` | `>= 4.199` | 无限制 | 直接沿用 XJTU 精确值 |
| **ZN-coin** | ❌ | 无 | `[V_start+0.1, 1.79]` | N/A | N/A | 纯恒流，**8 个 CV 特征置 0** |

---

## 6. 关键工程决策

### 6.1 PINN4SOH 子集的实现优先级

对于 4 个 PINN4SOH 子集，**必须严格遵循原始代码中的精确区间和前置过滤逻辑**：

1. **MATR (MIT)**：必须实现 `SOC > 80%` 前置过滤，否则特征空间与 PINN4SOH 不一致。
2. **HUST**：BatteryLife 中无 `Status` 字段，改用 `I > 1.5A` 过滤排除 5C 快充段，再对剩余数据划分 CC/CV。
3. **Tongji (TJU)**：必须使用 `control/mA` 和 `control/V` 字段划分 CC/CV，而非电压阈值。
4. **XJTU**：直接使用电压阈值 `4.199V` 划分，无前置过滤。

### 6.2 CC/CV 自动划分兜底策略

对于非 PINN4SOH 子集且缺乏阶段标签的数据集（如 SDU 的 `charge_protocol` 为空），采用基于数据的自动划分：

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

对于 NA-ion 和 ZN-coin：
- **全部 8 个 CV 相关特征统一置 `0.0`**。
- 仅提取 8 个 CC 特征（电压统计、CC Q/time、slope、entropy）。
- 16 维特征向量保持不变，CV 特征位恒为 `0.0`。模型训练时通过权重学习自动忽略这些常量维度。
- 不采用"从全充电过程提取 current stats"的替代方案，以避免引入与 PINN4SOH 不等价的特征语义。

---

## 7. 待验证事项

在正式编写代码前，建议对以下 3 个数据集抽取样本做**可视化确认**：

1. **HUST**：确认 `I > 1.5A` 过滤是否能干净地排除 5C 段，保留的 1C + CV 段与 PINN4SOH 原始结果的一致性。
2. **CALCE**：确认 CV 阶段的真实长度（此前分析显示 CV 仅约 44 个点）。
3. **ISU_ILCC**：确认充电起始电压，验证动态下限 `max(3.6, V_start+0.1)` 是否会导致窗口过窄。
