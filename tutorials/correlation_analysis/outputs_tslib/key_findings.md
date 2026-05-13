# 关键发现（精简版）

> 数据来源：`SOHbenchmark_TSLib/dataset/BatteryLife` 与 `SOHbenchmark_TSLib/results`；模型性能映射已排除 `MIX_large`。

## 覆盖范围

- 1,382 个电芯特征画像；3,080 行电池级指标；49,280 行特征-预测关系记录
- 覆盖 24 个任务、2 个目标（SOH/RUL）、5 个模型

## 核心发现

1. **特征与真值**：RWTH、NA-ion、HNEI 的部分特征与 SOH 呈极强单调关系（Spearman 最高达 -0.998）；HNEI、XJTU 与 RUL 相关性最强。
2. **特征与误差**：CC 阶段容量（`CC_Q`）、充电时间（`CC_charge_time`）、斜率（`voltage_slope`）同时与标签和模型误差高度相关。
3. **"相关性越高，误差越小"是否成立？**
   - **数据集层面**：部分成立。RUL 的 `max_abs_corr` 与 MAPE 相关系数达 -0.774；SOH 的 `max_abs_corr` 与 MAE 达 -0.600。
   - **逐样本层面**：不成立。特征-真值相关性与特征-误差敏感度的 Spearman 为正（SOH 0.126，RUL 0.155），说明高相关特征也可能对应更强的误差敏感性。

## 稳妥结论

> 高特征-真值相关性通常提升**数据集层面**的可预测性；但在**逐样本层面**，高相关特征也可能对应更强的误差敏感性，不能直接推出"特征相关性越高，该区域模型误差越小"。

## 输出文件

- `hypothesis_feature_error_sensitivity.csv`
- `hypothesis_dataset_model_performance.csv`
- `hypothesis_test_summary.csv`
- `figures/`
