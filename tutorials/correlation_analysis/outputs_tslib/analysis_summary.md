# 特征关系分析摘要

## 数据覆盖

- 电池特征画像：1,382 个电芯
- Benchmark 电池级指标：3,080 行
- 关联后的指标/画像行：3,080 行（无缺失）
- 逐样本预测关系记录：49,280 行
- 目标：RUL、SOH
- 模型：Attention、CNN、GRU、LSTM、MLP
- 任务：24 个

## 最强特征-真值关联

### SOH（Top 10）

| 数据集 | 特征 | Spearman | Pearson | 样本量 |
|--------|------|----------|---------|--------|
| RWTH | CV_charge_time | -0.9983 | -0.8995 | 105,397 |
| RWTH | current_slope | -0.9977 | -0.9853 | 105,397 |
| NA-ion | CC_Q | 0.9966 | 0.9318 | 12,604 |
| RWTH | CC_Q | 0.9943 | 0.8865 | 105,397 |
| HNEI | CC_charge_time | 0.9879 | 0.9460 | 15,164 |
| HNEI | voltage_slope | -0.9878 | -0.9089 | 15,164 |
| HNEI | CC_Q | 0.9877 | 0.9787 | 15,164 |
| RWTH | current_kurtosis | -0.9850 | -0.9462 | 105,397 |
| HNEI | voltage_skewness | 0.9795 | 0.9681 | 15,164 |
| RWTH | current_skewness | -0.9786 | -0.9669 | 105,397 |

### RUL（Top 10）

| 数据集 | 特征 | Spearman | Pearson | 样本量 |
|--------|------|----------|---------|--------|
| HNEI | voltage_slope | -0.9801 | -0.9017 | 15,164 |
| HNEI | CC_charge_time | 0.9783 | 0.9409 | 15,164 |
| HNEI | CC_Q | 0.9777 | 0.9733 | 15,164 |
| HNEI | voltage_skewness | 0.9754 | 0.9671 | 15,164 |
| XJTU | CC_Q | 0.9659 | 0.8961 | 6,938 |
| HNEI | voltage_mean | -0.9633 | -0.2252 | 15,164 |
| HNEI | current_kurtosis | -0.9624 | -0.9129 | 15,164 |
| HNEI | current_skewness | -0.9604 | -0.9232 | 15,164 |
| RWTH | CV_charge_time | -0.9601 | -0.8995 | 105,397 |
| HNEI | current_slope | -0.9595 | -0.9234 | 15,164 |

## 特征画像与模型性能关联

### RUL
- MAE 最强描述子：`cycle_max`（Spearman = 0.6565，n = 1,540）
- MAPE 最强描述子：`current_slope__std`（Spearman = -0.2772）
- RMSE 最强描述子：`cycle_max`（Spearman = 0.6687）
- R² 最强描述子：`CV_charge_time__spearman_SOH`（Spearman = -0.3173）
- Pearson_r 最强描述子：`CV_charge_time__spearman_SOH`（Spearman = -0.4222）

### SOH
- MAE/MAPE/RMSE 最强描述子：`CC_Q__mean`（Spearman ≈ -0.52 ~ -0.56）
- R²/Pearson_r 最强描述子：`current_entropy__spearman_SOH`（Spearman ≈ 0.43 ~ 0.47）

## 预测输出覆盖

可用预测文件：3,080 个，覆盖 24 个任务 × 2 个目标 × 5 个模型 = 240 组实验。

## 主要输出文件

- `feature_truth_correlations_by_dataset.csv`
- `battery_feature_profiles.csv`
- `sohbenchmark_battery_metrics_long.csv`
- `battery_metrics_with_feature_profiles.csv`
- `feature_profile_performance_mapping.csv`
- `feature_prediction_error_relationships_available_npz.csv`
