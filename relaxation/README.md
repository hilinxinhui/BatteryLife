# 电压弛豫特征提取

本目录将 Zhu 等人（2022, *Nature Communications*）提出的满充后电压弛豫特征，扩展到 BatteryLife 仓库的 `.pkl` 格式数据集，验证其对 SOH / RUL 估计的有效性。

---

## 1. 分析策略

为兼顾覆盖率与跨数据集可比性，输出两套独立结果：

- **`full_window`**（主分析）：使用每个 cycle 完整的可用 relaxation 段。覆盖率最高，回答的问题是：
  > **只要数据集中存在满充后弛豫，这些特征是否对 SOH 估计有用？**

- **`fixed_windows`**（补充分析）：从 relaxation 起点截取固定时长窗口（300 s / 600 s / 1800 s）。回答的问题是：
  > **在相同观测时长下，不同数据集的弛豫特征表现如何？**

> ⚠️ 两类结果回答的问题不同，不要在同一张主表中混用。

---

## 2. 数据集纳入标准

一个 cycle 的 rest 段被认定为"满充后弛豫"，需同时满足：

1. 电流接近 0；
2. 电压接近满充电压；
3. rest 段前为充电电流；
4. rest 段后为放电电流。

若同一 cycle 存在多个符合条件的段，取最长的一段。没有兼容 cycle 的数据集不会进入特征表，但审计日志中保留排除原因。

---

## 3. 特征字段

| 字段 | 含义 |
|:---|:---|
| `relax_var` | 弛豫电压方差 |
| `relax_ske` | 弛豫电压偏度 |
| `relax_kur` | 弛豫电压超额峰度 |
| `relax_max` | 弛豫电压最大值 |
| `relax_min` | 弛豫电压最小值 |
| `relax_mean` | 弛豫电压均值 |

此外，每个 `.csv` 还包含以下元数据或标签：

- `cycle_number`（对齐用，不作为默认输入特征）
- `relax_duration_s`、`relax_points`、`relax_start_voltage`、`relax_end_voltage`、`relax_voltage_delta`
- `SOH`、`RUL`、`discharge_capacity_in_Ah`

---

## 4. 结果目录结构

```text
relaxation/
├── full/                           # full_window 逐电池 .csv
│   ├── Tongji/
│   ├── XJTU/
│   └── ...
├── fixed/
│   ├── 300s/                       # 300 s 固定窗口
│   ├── 600s/
│   └── 1800s/
└── summary/                        # 覆盖率、相关性、报告
    ├── relaxation_dataset_coverage.csv
    ├── full_window_correlations.csv
    ├── fixed_window_correlations.csv
    ├── full_window_report.md
    ├── fixed_window_report.md
    └── dropped_cells_by_strategy.csv
```

---

## 5. 代码模块与执行顺序

### 5.1 模块说明

| 模块 | 功能 | 输入 | 输出 |
|:---|:---|:---|:---|
| `relaxation_features.py` | **核心特征提取**。遍历所有 `.pkl` 电芯文件，审计每个 cycle 的电流/电压/时间序列，识别满充后 relaxation 段，计算 6 项统计特征，并计算与 SOH / RUL / 放电容量的相关性。 | `dataset/*.pkl`（电芯数据）；`dataset/Life labels/*_labels.json`（寿命标签）；`tutorials/feature_extraction/configs/dataset_intervals.json`（数据集配置） | 默认输出到 `--output-dir` 下：<br>• `audit/relaxation_cycle_audit.csv`（逐 cycle 审计日志）<br>• `audit/relaxation_dataset_coverage.csv`（数据集覆盖率）<br>• `combined/relaxation_cycle_features.csv`（全部特征）<br>• `full_window/relaxation_cycle_features.csv`（full 策略特征）<br>• `fixed_windows/relaxation_cycle_features.csv`（fixed 策略特征）<br>• `relaxation_feature_correlations.csv`（相关性矩阵）<br>• `relaxation_report.md`（Markdown 报告） |
| `run_relaxation_pipeline.py` | **入口脚本**，直接调用 `relaxation_features.py` 的 `main()`。功能与输入输出同 `relaxation_features.py`。 | 同左 | 同左 |
| `organize_outputs.py` | **（可选）输出整理**。将 `relaxation_features.py` 的扁平输出重新分区为 `audit/` / `full_window/` / `fixed_windows/` / `combined/` 目录。仅在需要调整输出结构时独立使用。 | `--input-dir`（含 `relaxation_cycle_features.csv` 等文件的目录，默认 `outputs`） | `--output-dir`（默认 `outputs_split`）下的分区目录 |
| `build_cell_dataset.py` | **逐电池数据集组装**。读取已分区的 `full_window/` 和 `fixed_windows/by_window/` 结果，按 BatteryLife 风格拆分为每个电芯独立的 `.csv`，并复制汇总报告到 `summary/`。 | `--input-dir`（含 `full_window/` 和 `fixed_windows/by_window/` 的目录） | `full/`、`fixed/`、`summary/` |

### 5.2 执行顺序

标准复现只需两步：

```bash
# 第 1 步：从原始 .pkl 提取特征与审计信息
python relaxation/run_relaxation_pipeline.py --output-dir relaxation/_work

# 第 2 步：生成逐电池数据集与汇总报告
python relaxation/build_cell_dataset.py --input-dir relaxation/_work --output-dir relaxation
```

**说明**：
- `relaxation/_work` 为中间目录，确认 `full/`、`fixed/`、`summary/` 生成无误后可删除。
- `run_relaxation_pipeline.py` 内部已通过 `write_output_layout()` 将结果按 `full_window/` 和 `fixed_windows/` 分区，因此可直接作为 `build_cell_dataset.py` 的输入，通常无需额外运行 `organize_outputs.py`。
- 如果已有 `_work/` 等中间结果且不想重新提取特征，可直接运行第 2 步。

### 5.3 关键可调参数（`relaxation_features.py`）

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--current-epsilon-c` | `0.02` | 相对电流阈值系数。判定 rest 的电流阈值为 `max(0.05 A, 0.02 × nominal_capacity)`。 |
| `--voltage-tolerance` | `0.03` | 电压容差（V）。relaxation 段最高电压需 ≥ `cutoff_voltage - 0.03`。 |
| `--windows-s` | `300 600 1800` | 固定窗口时长（秒）。 always 同时输出 `full` 窗口。 |
| `--min-points` | `3` | 有效 relaxation 段最少数据点数。 |
| `--min-duration-s` | `30` | 有效 relaxation 段最少持续时间（秒）。 |

---

## 6. 策略覆盖概况

| 策略 | 保留电芯数 | 说明 |
|:---|---:|:---|
| `full` | 912 | 使用完整 relaxation，覆盖率最高 |
| `fixed/300s` | 365 | relaxation 至少覆盖前 300 s |
| `fixed/600s` | 288 | relaxation 至少覆盖前 600 s |
| `fixed/1800s` | 117 | relaxation 至少覆盖前 1800 s，最接近论文 30 min |

数量减少的两类原因：

1. **无符合定义的满充后弛豫**：整数据集丢弃 `HUST`、`MICH_EXP`、`NA-ion`、`RWTH`、`ZN-coin`；部分丢弃 `CALB`（13/27）、`MATR`（85/169）、`SNL`（25/61）。
2. **固定窗口时长不足**：`full` 中可用的电芯，因 relaxation 持续时间不足以构造对应固定窗口而被丢弃。

各策略下被丢弃的具体电芯名称见 `summary/dropped_cells_by_strategy.csv`（字段：`strategy,dataset,dropped_cell`）。
