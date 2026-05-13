# 关键发现

> 数据来源：
> - 特征文件：`SOHbenchmark_TSLib/dataset/BatteryLife`
> - Benchmark 结果：`SOHbenchmark_TSLib/results`
> - 模型性能映射已排除 `MIX_large`

## 覆盖范围

- 特征画像：1,382 个电芯
- Benchmark 电池级指标：3,080 行
- 逐样本预测文件：`res.npz` × 3,080
- 特征-预测/误差关系记录：49,280 行
- 覆盖 24 个任务、2 个目标（SOH/RUL）、5 个模型

## 特征与真值的关系

- **SOH**：RWTH、NA-ion、HNEI 的部分特征与真值呈现极强单调关系。
  - RWTH：`CV_charge_time`（Spearman = -0.9983）、`current_slope`（-0.9977）
  - NA-ion：`CC_Q`（0.9966）
- **RUL**：HNEI、XJTU、RWTH 的相关性最强。
  - HNEI：`voltage_slope`（-0.9801）、`CC_charge_time`（0.9783）、`CC_Q`（0.9777）
  - XJTU：`CC_Q`（0.9659）

## 特征与模型误差的关系

- RUL 误差最敏感的特征：`CC_Q`、`CC_charge_time`、`voltage_slope`
- SOH 误差最敏感的特征：`CC_charge_time`、`CC_Q`、`voltage_slope`

CC 阶段容量、充电时间、斜率类特征不仅与标签相关，也显著标记模型在不同生命周期区域的误差变化。

## "相关性越高，误差越小"是否成立？

**区分两个层面：**

1. **数据集整体可预测性层面**：结论有一定支持。
   - SOH：最强特征-真值相关性与 MAE（Spearman = -0.600）、RMSE（-0.560）、Pearson 预测质量（0.648）
   - RUL：最强特征-真值相关性与 MAPE（-0.739）、R²（0.764）
   - 说明：存在强特征-标签单调关系的数据集，模型通常更容易获得较好表现。

2. **单个特征逐样本误差层面**：结论不成立，甚至相反。
   - SOH：特征-真值相关性与特征-绝对误差敏感度的 Spearman = 0.126
   - RUL：对应 Spearman = 0.155
   - 正相关意味着：越与真值相关的特征，越可能是模型误差随生命周期变化的敏感方向。这不等价于"误差更小"。

## 稳妥结论

> 高特征-真值相关性通常提升数据集层面的可预测性，尤其体现在 RUL 的相对误差和 SOH 的拟合质量上；但在逐样本层面，高相关特征也可能对应更强的误差敏感性，因此不能直接推出"某个特征相关性越高，该特征对应区域的模型误差越小"。

## 主要输出文件

- `hypothesis_feature_error_sensitivity.csv`
- `hypothesis_dataset_model_performance.csv`
- `hypothesis_test_summary.csv`
- `figures/`
- `feature_correlation_analysis_zh.ipynb`
