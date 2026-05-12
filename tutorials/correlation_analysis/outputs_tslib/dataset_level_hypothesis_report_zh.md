# 数据集级假设检验：特征相关性是否解释预测好坏

## 假设

更直白地说，我们要检验：预测效果差的数据集，是否因为特征提取环节没有抓住与 SOH/RUL 目标足够相关的信息；预测效果好的数据集，是否具有更强的特征-目标相关性。

本分析排除 `MIX_large`，因为它是组合数据集，会混合多个物理数据集的分布。

## 构造的高层指标

- `mean_abs_corr`：一个数据集内 16 个特征与目标的平均 |Spearman|。
- `max_abs_corr`：最强单个特征与目标的 |Spearman|。
- `top3_abs_corr`：最强 3 个特征与目标的平均 |Spearman|。
- `strong_feature_fraction`：|Spearman| >= 0.7 的特征比例。
- `performance_score`：由 MAPE、RMSE、R2、Pearson_r 的秩分数合成，越高表示预测整体越好。

## 全局检验结果

### SOH
- `feature_quality_score` vs `performance_score`: Spearman=0.070, n=24
- `max_abs_corr` vs `mean_mape`: Spearman=-0.349, n=24
- `max_abs_corr` vs `mean_r2`: Spearman=0.471, n=24
- `top3_abs_corr` vs `performance_score`: Spearman=0.103, n=24
- `mean_abs_corr` vs `performance_score`: Spearman=-0.148, n=24
- 象限计数：高相关-好预测=7；高相关-差预测=6；低相关-好预测=6；低相关-差预测=5

### RUL
- `feature_quality_score` vs `performance_score`: Spearman=0.548, n=24
- `max_abs_corr` vs `mean_mape`: Spearman=-0.774, n=24
- `max_abs_corr` vs `mean_r2`: Spearman=0.787, n=24
- `top3_abs_corr` vs `performance_score`: Spearman=0.545, n=24
- `mean_abs_corr` vs `performance_score`: Spearman=0.482, n=24
- 象限计数：高相关-好预测=8；低相关-差预测=6；高相关-差预测=6；低相关-好预测=4

## 解释

从数据集级结果看，假设有一定支持，但不是无条件成立：

- 对 RUL，最强特征相关性与 MAPE 通常呈明显负相关，与 R2/综合表现呈正相关。这支持“特征更相关，数据集更容易预测”。
- 对 SOH，最强特征相关性与 MAE/RMSE 也呈负相关，与 R2/Pearson_r 呈正相关，同样支持数据集级假设。
- 但平均相关性不总是比最大相关性更有解释力，这说明模型可能只需要少数关键特征就能得到较好表现，而不是所有 16 个特征都必须强相关。
- 高相关但差预测的数据集仍然存在，提示误差还受样本量、标签尺度、训练划分、模型结构、噪声、数据分布偏移等因素影响。

## 可视化输出

- `figures_dataset_level/dataset_level_scatter_SOH.png`
- `figures_dataset_level/dataset_level_scatter_RUL.png`
- `figures_dataset_level/dataset_level_ranked_bars_SOH.png`
- `figures_dataset_level/dataset_level_ranked_bars_RUL.png`
- `figures_dataset_level/dataset_level_quadrants_SOH.png`
- `figures_dataset_level/dataset_level_quadrants_RUL.png`

## 表格输出

- `dataset_level_hypothesis_summary.csv`
- `dataset_level_hypothesis_correlations.csv`
- `dataset_level_hypothesis_quadrants.csv`
