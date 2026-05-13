# BatteryLife Benchmark 任务定义

本文档定义 BatteryLife 数据集上的 18 个 benchmark 任务，供后续实验与评估使用。

## 设计原则

1. **物理电池为粒度**：任务面向物理电池实体，而非数据集名称。
2. **去重**：同一批物理电池的 seed 变体（如 `CALB`/`CALB42`/`CALB2024`）只保留一个默认版本。
3. **全覆盖**：所有物理数据源均定义独立任务；尚未接入的数据集（`SDU`、`Stanford_2`）也纳入评估。
4. **Stanford 系列取并集**：采用 `Stanford_2` 完整版，并补充 `Stanford` 独有的 3 块参考电池。

## 任务列表（18 个）

```bash
BATTERYLIFE_RUNNABLE_TASKS=(
  "CALB" "CALCE" "HNEI" "HUST" "ISU_ILCC" "MATR" "MICH"
  "MICH_EXP" "NA-ion" "RWTH" "SDU" "SNL" "Stanford_2"
  "Tongji" "UL_PUR" "XJTU" "ZN-coin" "MIX_large"
)
```

## 各任务物理电池数量

统计基准：v11 版本 `.pkl` 文件。

| # | 任务 | 全量 pkl | 实际使用 | 丢弃 | 说明 |
|---|------|---------:|---------:|-----:|------|
| 1 | CALB | 27 | 27 | 0 | 默认划分（seed 2021） |
| 2 | CALCE | 13 | 13 | 0 | |
| 3 | HNEI | 14 | 14 | 0 | |
| 4 | HUST | 77 | 77 | 0 | |
| 5 | ISU_ILCC | 240 | 240 | 0 | |
| 6 | MATR | 169 | 169 | 0 | |
| 7 | MICH | 40 | 40 | 0 | |
| 8 | MICH_EXP | 18 | 12 | 6 | 6 块 50-100% SOC 电池无寿命标签 |
| 9 | NA-ion | 64 | 34 | 30 | 默认划分（seed 2021） |
| 10 | RWTH | 48 | 48 | 0 | |
| 11 | SDU | 86 | 70 | 16 | 16 块无寿命标签 |
| 12 | SNL | 61 | 52 | 9 | 9 块无寿命标签 |
| 13 | Stanford_2 | **184** | **184** | 0 | 181（Stanford_2/）+ 3（Stanford/ 独有 Ref） |
| 14 | Tongji | 130 | 108 | 22 | 22 块无寿命标签 |
| 15 | UL_PUR | 10 | 2 | 8 | 8 块无寿命标签 |
| 16 | XJTU | 23 | 23 | 0 | |
| 17 | ZN-coin | 140 | 121 | 19 | 默认划分（seed 2021） |
| 18 | MIX_large | **1,344** | **1,234** | **110** | 17 个独立数据源的去重并集 |

### 丢弃说明

`实际使用数` 指存在 `Life labels/*.json` 寿命标签的电池。缺少标签的电池不纳入 SOH/RUL benchmark。

丢弃电池的典型特征：
- **MICH_EXP**（6 块）：50-100% SOC 区间电池，末循环 SOH ≈ 0.995–1.0
- **NA-ion**（30 块）：末循环 SOH 0.70–0.90，提前终止或无标签
- **SDU**（16 块）：末循环 SOH 0.63–0.79
- **SNL**（9 块）：末循环 SOH 0.83–0.88
- **Tongji**（22 块）：末循环 SOH 0.82–0.90
- **UL_PUR**（8 块）：末循环 SOH 0.71–0.82
- **ZN-coin**（19 块）：末循环 SOH 0.86–1.0

> 完整丢弃清单（含文件名与末循环 SOH）见数据审计日志。

## 关键设计说明

### Seed 变体处理

| 变体组 | 保留版本 | 说明 |
|--------|----------|------|
| CALB / CALB42 / CALB2024 | **CALB** | seed 2021 |
| ZN-coin / ZN-coin42 / ZN-coin2024 | **ZN-coin** | seed 2021 |
| NA-ion / NAion42 / NAion2024 | **NA-ion** | seed 2021 |

### Stanford 系列

| 版本 | 电池数 | 说明 |
|------|--------|------|
| Stanford | 41 | 第一版 release |
| Stanford_2 | 181 | 完整版 |
| 重叠 | 38 | 两版共有 |
| Stanford 独有 | 3 | Ref_100, Ref_101, Ref_102 |

**采用方案**：保留 `Stanford_2` 任务，共 **184 块** 物理电池（Stanford_2/ 的 181 块 + Stanford/ 独有的 3 块）。不再单独保留 `Stanford` 任务。

### MIX_large

17 个独立物理数据源的去重并集。全量 1,344 块，有寿命标签的 1,234 块纳入 benchmark。**不包含任何 seed 变体**，避免同一批物理电池重复。

## 代码适配状态

`data_provider/data_loader.py` 已按上述任务列表接入：

- 18 个任务使用固定随机种子 `2021`，按物理电池随机划分 train/val/test。
- 只包含存在寿命标签的 pkl；缺少标签的电池自动丢弃。
- `SDU` 已添加读取路径；`Stanford_2` 已添加目录和标签读取逻辑，并纳入 Stanford/ 独有的 3 块 Ref 电池。
- `MIX_large` 由 17 个独立数据源的划分拼接而成，避免 seed 变体重复。
