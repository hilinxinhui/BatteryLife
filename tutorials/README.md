# BatteryLife Benchmark Task Definition

本文档定义了 BatteryLife 数据集上的 benchmark 任务列表，供后续分析和实验使用。

## 设计原则

1. **以物理电池为粒度**：任务面向物理电池实体，而非数据集名称。
2. **去重**：同一批物理电池的 seed 变体（如 `CALB`/`CALB42`/`CALB2024`）只保留一个默认版本。
3. **全覆盖**：所有物理数据源均定义独立任务；当前仓库中尚未接入的数据集（`SDU`、`Stanford_2`）也应纳入评估。
4. **Stanford 系列取并集最大化**：采用 `Stanford_2`（完整版）并补充 `Stanford` 独有的 3 块参考电池，最大化覆盖物理电池。

## 任务列表（共 18 个）

```bash
BATTERYLIFE_RUNNABLE_TASKS=(
  "CALB"
  "CALCE"
  "HNEI"
  "HUST"
  "ISU_ILCC"
  "MATR"
  "MICH"
  "MICH_EXP"
  "NA-ion"
  "RWTH"
  "SDU"
  "SNL"
  "Stanford_2"
  "Tongji"
  "UL_PUR"
  "XJTU"
  "ZN-coin"
  "MIX_large"
)
```

## 各任务物理电池数量

以下统计以 `/home/lxh/Desktop/datasets/BatteryLife/v11` 中的 pkl 文件为准。`全量 pkl 数` 表示该数据源目录下的物理电池数；`实际使用数` 表示存在 `Life labels/*.json` 寿命标签、可用于当前 SOH/RUL 任务的电池数。缺少寿命标签的电池在本 benchmark 中丢弃。

| # | 任务 | 全量 pkl 数 | 实际使用数 | 丢弃数 | 说明 |
|---|------|------------:|-----------:|-------:|------|
| 1 | `CALB` | 27 | 27 | 0 | 默认划分（seed 2021） |
| 2 | `CALCE` | 13 | 13 | 0 | |
| 3 | `HNEI` | 14 | 14 | 0 | |
| 4 | `HUST` | 77 | 77 | 0 | |
| 5 | `ISU_ILCC` | 240 | 240 | 0 | |
| 6 | `MATR` | 169 | 169 | 0 | |
| 7 | `MICH` | 40 | 40 | 0 | |
| 8 | `MICH_EXP` | 18 | 12 | 6 | 6 块 50-100% SOC 电池缺少寿命标签 |
| 9 | `NA-ion` | 64 | 34 | 30 | 默认划分（seed 2021）；30 块缺少寿命标签 |
| 10 | `RWTH` | 48 | 48 | 0 | |
| 11 | `SDU` | 86 | 70 | 16 | 16 块缺少寿命标签 |
| 12 | `SNL` | 61 | 52 | 9 | 9 块缺少寿命标签 |
| 13 | `Stanford_2` | **184** | **184** | 0 | 181（`Stanford_2/` 目录）+ 3（`Stanford` 独有的 Ref 电池） |
| 14 | `Tongji` | 130 | 108 | 22 | 22 块缺少寿命标签 |
| 15 | `UL_PUR` | 10 | 2 | 8 | 8 块缺少寿命标签 |
| 16 | `XJTU` | 23 | 23 | 0 | |
| 17 | `ZN-coin` | 140 | 121 | 19 | 默认划分（seed 2021）；19 块缺少寿命标签 |
| 18 | `MIX_large` | **1,344** | **1,234** | **110** | 上述 17 个独立物理数据源的去重并集；缺少寿命标签的电池不纳入 |

### 丢弃电池说明

以下电池在 v11 pkl 中存在，但没有出现在对应的 `Life labels/*.json` 中，因此不纳入当前 SOH/RUL benchmark。`last_soh` 为按当前仓库寿命标签脚本口径计算的归一化末循环 SOH：

`last_soh = max(last_cycle.discharge_capacity_in_Ah) / nominal_capacity_in_Ah / SOC_interval`

#### `MICH_EXP` 丢弃 6 块

- `MICH_13R_pouch_NMC_25C_50-100_0.2-0.2C.pkl`: last_soh=0.9992
- `MICH_14C_pouch_NMC_-5C_50-100_0.2-0.2C.pkl`: last_soh=0.9992
- `MICH_15H_pouch_NMC_45C_50-100_0.2-0.2C.pkl`: last_soh=0.9992
- `MICH_16R_pouch_NMC_25C_50-100_0.2-1.5C.pkl`: last_soh=0.9956
- `MICH_17C_pouch_NMC_-5C_50-100_0.2-1.5C.pkl`: last_soh=0.9956
- `MICH_18H_pouch_NMC_45C_50-100_0.2-1.5C.pkl`: last_soh=0.9956

#### `NA-ion` 丢弃 30 块

- `NA-ion_270040-1-1-64.pkl`: last_soh=0.7688
- `NA-ion_270040-1-4-61.pkl`: last_soh=0.8686
- `NA-ion_270040-2-1-12.pkl`: last_soh=0.8595
- `NA-ion_270040-2-3-12.pkl`: last_soh=0.8748
- `NA-ion_270040-2-4-12.pkl`: last_soh=0.7587
- `NA-ion_270040-2-6-12.pkl`: last_soh=0.7646
- `NA-ion_270040-2-7-12.pkl`: last_soh=0.7841
- `NA-ion_270040-2-8-12.pkl`: last_soh=0.8830
- `NA-ion_270040-3-6-51.pkl`: last_soh=0.8333
- `NA-ion_270040-4-4-45.pkl`: last_soh=0.8437
- `NA-ion_270040-4-5-44.pkl`: last_soh=0.8290
- `NA-ion_270040-4-7-42.pkl`: last_soh=0.8760
- `NA-ion_270040-4-8-41.pkl`: last_soh=0.7040
- `NA-ion_270040-5-4-36.pkl`: last_soh=0.7942
- `NA-ion_270040-6-1-31.pkl`: last_soh=0.8380
- `NA-ion_270040-6-3-29.pkl`: last_soh=0.8702
- `NA-ion_270040-6-4-28.pkl`: last_soh=0.8682
- `NA-ion_270040-6-5-27.pkl`: last_soh=0.6999
- `NA-ion_270040-6-7-25.pkl`: last_soh=0.8647
- `NA-ion_270040-7-2-22.pkl`: last_soh=0.7285
- `NA-ion_270040-7-3-21.pkl`: last_soh=0.8641
- `NA-ion_270040-8-1-20.pkl`: last_soh=0.8649
- `NA-ion_270040-8-2-19.pkl`: last_soh=0.8749
- `NA-ion_270040-8-3-18.pkl`: last_soh=0.7168
- `NA-ion_270040-8-4-17.pkl`: last_soh=0.8762
- `NA-ion_270040-8-6-15.pkl`: last_soh=0.8807
- `NA-ion_270040-8-7-14.pkl`: last_soh=0.8997
- `NA-ion_270040-8-8-13.pkl`: last_soh=0.8647
- `NA-ion_2850-30_20250117105706_DefaultGroup_45_2.pkl`: last_soh=0.8386
- `NA-ion_5000-25_20250115110326_DefaultGroup_38_2.pkl`: last_soh=0.8459

#### `SDU` 丢弃 16 块

- `SDU_Battery_32.pkl`: last_soh=0.6905
- `SDU_Battery_38.pkl`: last_soh=0.7258
- `SDU_Battery_39.pkl`: last_soh=0.6748
- `SDU_Battery_40.pkl`: last_soh=0.6855
- `SDU_Battery_41.pkl`: last_soh=0.6996
- `SDU_Battery_42.pkl`: last_soh=0.6914
- `SDU_Battery_43.pkl`: last_soh=0.6664
- `SDU_Battery_44.pkl`: last_soh=0.6881
- `SDU_Battery_57.pkl`: last_soh=0.7940
- `SDU_Battery_58.pkl`: last_soh=0.7555
- `SDU_Battery_59.pkl`: last_soh=0.7916
- `SDU_Battery_60.pkl`: last_soh=0.7919
- `SDU_Battery_69.pkl`: last_soh=0.6535
- `SDU_Battery_70.pkl`: last_soh=0.6842
- `SDU_Battery_71.pkl`: last_soh=0.6359
- `SDU_Battery_72.pkl`: last_soh=0.6486

#### `SNL` 丢弃 9 块

- `SNL_18650_LFP_15C_0-100_0.5-2C_b.pkl`: last_soh=0.8482
- `SNL_18650_LFP_25C_0-100_0.5-0.5C_a.pkl`: last_soh=0.8764
- `SNL_18650_LFP_25C_0-100_0.5-1C_a.pkl`: last_soh=0.8582
- `SNL_18650_LFP_25C_0-100_0.5-1C_b.pkl`: last_soh=0.8573
- `SNL_18650_LFP_25C_0-100_0.5-1C_c.pkl`: last_soh=0.8845
- `SNL_18650_LFP_25C_0-100_0.5-1C_d.pkl`: last_soh=0.8845
- `SNL_18650_LFP_25C_0-100_0.5-2C_a.pkl`: last_soh=0.8636
- `SNL_18650_LFP_25C_0-100_0.5-2C_b.pkl`: last_soh=0.8545
- `SNL_18650_LFP_35C_0-100_0.5-1C_a.pkl`: last_soh=0.8345

#### `Tongji` 丢弃 22 块

- `Tongji1_CY25-05_1--8.pkl`: last_soh=0.8601
- `Tongji1_CY25-05_1--9.pkl`: last_soh=0.8600
- `Tongji1_CY35-05_1--3.pkl`: last_soh=0.9035
- `Tongji1_CY45-05_1--3.pkl`: last_soh=0.8582
- `Tongji1_CY45-05_1--4.pkl`: last_soh=0.8470
- `Tongji2_CY25-05_1--1.pkl`: last_soh=0.8344
- `Tongji2_CY25-05_1--11.pkl`: last_soh=0.8262
- `Tongji2_CY25-05_1--14.pkl`: last_soh=0.8847
- `Tongji2_CY25-05_1--18.pkl`: last_soh=0.8355
- `Tongji2_CY25-05_1--19.pkl`: last_soh=0.8430
- `Tongji2_CY25-05_1--20.pkl`: last_soh=0.8418
- `Tongji2_CY25-05_1--21.pkl`: last_soh=0.8332
- `Tongji2_CY25-05_1--22.pkl`: last_soh=0.8284
- `Tongji2_CY25-05_1--23.pkl`: last_soh=0.8493
- `Tongji2_CY25-05_1--3.pkl`: last_soh=0.8352
- `Tongji2_CY25-05_1--4.pkl`: last_soh=0.8339
- `Tongji2_CY25-05_1--6.pkl`: last_soh=0.8370
- `Tongji2_CY25-05_1--7.pkl`: last_soh=0.8340
- `Tongji2_CY45-05_1--3.pkl`: last_soh=0.8522
- `Tongji2_CY45-05_1--4.pkl`: last_soh=0.8447
- `Tongji2_CY45-05_1--5.pkl`: last_soh=0.8511
- `Tongji2_CY45-05_1--6.pkl`: last_soh=0.8373

#### `UL_PUR` 丢弃 8 块

- `UL-PUR_N10-EX9_18650_NCA_23C_0-100_0.5-0.5C_i.pkl`: last_soh=0.8238
- `UL-PUR_N10-OV8_18650_NCA_23C_0-100_0.5-0.5C_h.pkl`: last_soh=0.8135
- `UL-PUR_N15-EX4_18650_NCA_23C_0-100_0.5-0.5C_d.pkl`: last_soh=0.8015
- `UL-PUR_N15-OV3_18650_NCA_23C_0-100_0.5-0.5C_c.pkl`: last_soh=0.7947
- `UL-PUR_N20-EX2_18650_NCA_23C_0-100_0.5-0.5C_b.pkl`: last_soh=0.7174
- `UL-PUR_N20-NA5_18650_NCA_23C_0-100_0.5-0.5C_e.pkl`: last_soh=0.7853
- `UL-PUR_N20-NA6_18650_NCA_23C_0-100_0.5-0.5C_f.pkl`: last_soh=0.7535
- `UL-PUR_N20-OV1_18650_NCA_23C_0-100_0.5-0.5C_a.pkl`: last_soh=0.7338

#### `ZN-coin` 丢弃 19 块

- `ZN-coin_2_432-1_20231227204455_01_1.pkl`: last_soh=1.0019
- `ZN-coin_441-2_20231227204859_08_5.pkl`: last_soh=0.9052
- `ZN-coin_441-3_20231227204904_08_6.pkl`: last_soh=0.8700
- `ZN-coin_443-3_20240104212506_09_6.pkl`: last_soh=0.8868
- `ZN-coin_446-3_20240104212550_07_4.pkl`: last_soh=0.9012
- `ZN-coin_447-1_20240104212621_07_5.pkl`: last_soh=0.9292
- `ZN-coin_447-2_20240104212627_07_6.pkl`: last_soh=0.9340
- `ZN-coin_447-3_20240104212631_07_7.pkl`: last_soh=0.9422
- `ZN-coin_448-1_20240104212639_07_8.pkl`: last_soh=0.9175
- `ZN-coin_448-2_20240104212646_06_1.pkl`: last_soh=0.9854
- `ZN-coin_448-3_20240104212651_06_2.pkl`: last_soh=0.8692
- `ZN-coin_449-1_20240104212736_06_3.pkl`: last_soh=0.9638
- `ZN-coin_449-2_20240104212745_06_4.pkl`: last_soh=0.9346
- `ZN-coin_449-3_20240104212753_06_5.pkl`: last_soh=1.0017
- `ZN-coin_451-2_20240116203431_04_3_Batch-3.pkl`: last_soh=0.9856
- `ZN-coin_451-3_20240116203436_04_5_Batch-3.pkl`: last_soh=0.9383
- `ZN-coin_452-1_20240116203442_04_6_Batch-3.pkl`: last_soh=0.9507
- `ZN-coin_452-2_20240116203450_08_5_Batch-3.pkl`: last_soh=0.8663
- `ZN-coin_452-3_20240116204046_08_4_Batch-3.pkl`: last_soh=0.9792

## 关键设计说明

### Seed 变体处理

以下数据集存在多种 seed 变体（`42`、`2021`、`2024`），但物理电池完全相同：

- `CALB` / `CALB42` / `CALB2024` → **保留 `CALB`**（seed 2021）
- `ZN-coin` / `ZN-coin42` / `ZN-coin2024` → **保留 `ZN-coin`**（seed 2021）
- `NA-ion` / `NAion42` / `NAion2024` → **保留 `NA-ion`**（seed 2021）

### Stanford 系列处理

| 数据集 | 电池数 | 说明 |
|--------|--------|------|
| `Stanford` | 41 | 第一版 release |
| `Stanford_2` | 181 | 完整版（两版合并） |
| **重叠** | 38 | 同时存在于两个版本中 |
| **Stanford 独有** | 3 | `Ref_100`, `Ref_101`, `Ref_102` |

**方案 C（采用）**：保留 `Stanford_2` 任务，共 **184 块** 物理电池（`Stanford_2/` 目录的 181 块 + `Stanford/` 目录独有的 3 块参考电池）。不再单独保留 `Stanford` 任务。

### MIX_large 定义

`MIX_large` 为上述 **17 个独立物理数据源的去重并集**。全量 pkl 共 **1,344 块** 电池，其中 **1,234 块**存在寿命标签并纳入当前 SOH/RUL benchmark；缺少寿命标签的电池丢弃。它**不包含**任何 seed 变体（如 `CALB42`、`ZN-coin2024`），避免同一批物理电池重复出现。

## 代码适配状态

当前 `data_provider/data_loader.py` 已按上述任务列表接入 benchmark 划分：

- 18 个 benchmark 任务使用固定随机种子 `2021`，按物理电池文件随机划分 train/val/test。
- 划分只包含存在寿命标签的 pkl 文件；缺少 `Life labels/*.json` 条目的电池会被丢弃。
- `SDU` 已添加 `SDU/` 目录读取路径。
- `Stanford_2` 已添加 `Stanford_2/` 目录和 `Stanford_2_labels.json` 读取逻辑，并额外纳入 `Stanford/` 目录中独有的 `Ref_100`、`Ref_101`、`Ref_102`。
- `MIX_large` 由 17 个独立物理数据源的 benchmark 划分拼接得到，避免 seed 变体重复。
