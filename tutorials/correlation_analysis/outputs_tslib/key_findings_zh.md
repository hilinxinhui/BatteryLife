# 关键发现（中文）

数据来源：
- 特征文件：`/Users/lxh/Desktop/codes/SOHbenchmark_TSLib/dataset/BatteryLife`
- Benchmark 结果：`/Users/lxh/Desktop/codes/SOHbenchmark_TSLib/results`
- 本轮模型性能映射已排除 `MIX_large`。

## 覆盖范围

- 特征画像：1,382 个电池/电芯。
- Benchmark 电池级指标：3,080 行。
- 逐样本预测文件：3,080 个 `res.npz`。
- 特征-预测/误差关系记录：49,280 行。
- 覆盖 24 个任务、2 个目标（SOH/RUL）、5 个模型（Attention/CNN/GRU/LSTM/MLP）。

## 特征与真值的关系

- SOH 中，RWTH、NA-ion、HNEI 的部分特征和真值呈现极强单调关系。
- 典型例子：RWTH 的 `CV_charge_time` 与 SOH 的 Spearman 为 -0.9983，`current_slope` 为 -0.9977；NA-ion 的 `CC_Q` 为 0.9966。
- RUL 中，HNEI、XJTU、RWTH 的相关性最强。
- 典型例子：HNEI 的 `voltage_slope` 与 RUL 的 Spearman 为 -0.9801，`CC_charge_time` 为 0.9783，`CC_Q` 为 0.9777；XJTU 的 `CC_Q` 为 0.9659。

## 特征与模型输出/误差的关系

- 从逐样本预测结果看，RUL 误差平均最敏感的特征主要是 `CC_Q`、`CC_charge_time`、`voltage_slope`。
- SOH 误差平均最敏感的特征主要是 `CC_charge_time`、`CC_Q`、`voltage_slope`。
- 这说明 CC 阶段容量、充电时间、斜率类特征不仅与标签相关，也会显著标记模型在不同生命周期区域的误差变化。

## “相关性越高，模型误差越小”是否成立？

这个结论不能简单成立，需要区分两个层面：

1. 在“数据集整体可预测性”层面，结论有一定支持。
   - SOH：数据集内最强特征-真值相关性与 MAE 的 Spearman 为 -0.600，与 RMSE 为 -0.560，与 Pearson 预测质量为 0.648。
   - RUL：数据集内最强特征-真值相关性与 MAPE 的 Spearman 为 -0.739，与 R2 为 0.764。
   - 这些结果说明，当某个数据集存在非常强的特征-标签单调关系时，模型通常更容易获得较好的相对误差或拟合质量。

2. 在“单个特征是否导致更小逐样本误差”层面，结论不成立，甚至会反过来。
   - SOH：特征-真值相关性与特征-绝对误差敏感度的 Spearman 为 0.126。
   - RUL：对应 Spearman 为 0.155。
   - 正相关表示：越与真值相关的特征，越可能也是模型误差随生命周期变化的敏感方向。这并不等价于“误差更小”。

## 建议表述

更稳妥的结论是：

> 高特征-真值相关性通常提升数据集层面的可预测性，尤其体现在 RUL 的相对误差和 SOH 的拟合质量上；但在逐样本层面，高相关特征也可能对应更强的误差敏感性，因此不能直接推出“某个特征相关性越高，该特征对应区域的模型误差越小”。

## 主要输出

- `hypothesis_feature_error_sensitivity.csv`
- `hypothesis_dataset_model_performance.csv`
- `hypothesis_test_summary.csv`
- `figures/`
- `tutorials/correlation_analysis/feature_correlation_analysis_zh.ipynb`
