# 电压弛豫特征提取流水线

这个目录用于把 Zhu 等人在 Nature Communications 2022 论文
**Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation**
中提出的满充后电压弛豫特征，扩展到 BatteryLife 仓库中的 `.pkl`
格式数据集。

流水线分为三个阶段：

1. 审计每个 `.pkl` 电芯文件，判断每个 cycle 是否存在 rest 段，以及是否存在符合论文定义的满充后 relaxation 段。
2. 在可用 relaxation 段上提取电压统计特征。
3. 分析 relaxation 特征与 `SOH`、`RUL`、放电容量之间的相关性。

如果需要从原始 `.pkl` 重新生成结果，从仓库根目录运行：

```bash
python relaxation/run_relaxation_pipeline.py --output-dir relaxation/_work
python relaxation/build_cell_dataset.py --input-dir relaxation/_work --output-dir relaxation
```

`relaxation/_work` 是中间结果目录；确认 `full/`、`fixed/`、`summary/`
生成无误后可以删除。

## 分析策略

为了尽可能保留可用数据集，本工作把特征窗口分成两套结果保存。

1. `full_window`：主分析结果。  
   每个 cycle 使用它完整可用的、符合论文定义的 relaxation 段来提取特征。这个策略最大化数据集和 cycle 覆盖率，适合回答：

   ```text
   只要数据集中存在满充后 relaxation，这些特征是否对 SOH 估计有用？
   ```

2. `fixed_windows`：补充分析结果。  
   从 relaxation 段起点开始，分别截取固定长度窗口，目前包括 `300s`、`600s`、`1800s`。这个策略更适合跨数据集公平比较，因为同一个窗口内的特征都来自相同观测时长；代价是会丢掉 relaxation 太短的数据集或 cycle。

`full_window` 和 `fixed_windows` 回答的是不同问题，做结论时不要把两类结果混在同一张主表里解释。

## 数据集纳入标准

流水线会先审计所有数据集，然后只对满足论文 relaxation 定义的 cycle 提取特征。判据包括：

- 存在近零电流 rest 段；
- rest 段电压接近满充电压；
- rest 段前面是充电电流；
- rest 段后面是放电电流。

有可用满充后 relaxation cycle 的数据集会进入特征输出。没有兼容 cycle 的数据集不会进入特征表，但仍会保留在 `audit/` 结果中，方便追溯排除原因。

## 结果保存结构

最终结果按 BatteryLife 风格组织：不同数据集放在不同目录，同一数据集中的每个电池保存为独立 `.csv` 文件。每个 `.csv` 中：

- 每一行对应一个 cycle；
- 每一列对应一个 relaxation 特征或元数据；
- 最后几列为标签，例如 `SOH`、`RUL`、`discharge_capacity_in_Ah`；
- `cycle_number` 只作为对齐和排查用的元数据，不作为默认模型输入特征。

最终结果保存结构如下：

```text
relaxation/
  full/
    Tongji/
      Tongji1_CY25-05_1--1.csv
      ...
    XJTU/
      XJTU_2C_battery-1.csv
      ...
  fixed/
    300s/
      Tongji/
      XJTU/
      ...
    600s/
      ...
    1800s/
      ...
  summary/
    relaxation_dataset_coverage.csv
    full_window_correlations.csv
    fixed_window_correlations.csv
    full_window_report.md
    fixed_window_report.md
```

各目录含义：

- `full/`：主分析数据集。每个电池一个 `.csv`，使用完整可用 relaxation 段。
- `fixed/`：补充分析数据集。先按固定窗口长度分目录，再按数据集/电池保存。
- `summary/`：轻量汇总结果，包括覆盖率、相关性和报告。

如果问题是“这些 relaxation 特征对 SOH 是否有帮助”，优先看：

```text
relaxation/full/
```

如果问题是“在相同观测时长下，不同数据集的 relaxation 特征表现如何”，优先看：

```text
relaxation/fixed/
```

## 特征字段

论文中的核心电压弛豫统计特征字段为：

- `relax_var`：relaxation 电压序列方差；
- `relax_ske`：relaxation 电压序列偏度；
- `relax_max`：relaxation 电压最大值；
- `relax_min`：relaxation 电压最小值；
- `relax_mean`：relaxation 电压均值；
- `relax_kur`：relaxation 电压超额峰度。

特征表中还会保存一些质量控制和对齐字段，例如：

- `relax_duration_s`
- `relax_points`
- `relax_start_voltage`
- `relax_end_voltage`
- `relax_voltage_delta`
- `SOH`
- `RUL`
- `discharge_capacity_in_Ah`

## 默认窗口

默认固定窗口为：

```text
300s, 600s, 1800s
```

同时总是保存 `full` 窗口。`1800s` 是最接近论文 30 min 设置的固定窗口，但 BatteryLife 中很多数据集没有保存完整 30 min relaxation，因此主分析采用 `full_window`。

## 整理已有结果

## 生成逐电池数据集

如果已经有 `_work/`、`outputs_split/` 等中间结果，可以不重新提取特征，直接生成最终逐电池数据集：

```bash
python relaxation/build_cell_dataset.py --input-dir relaxation/_work --output-dir relaxation
```

这个命令会生成：

```text
relaxation/full/
relaxation/fixed/
relaxation/summary/
```

`outputs*` 目录只属于中间过程或验证过程；在最终结论导向的数据集组织中不需要保留。
