# 数据集级假设检验：特征相关性是否解释预测效果

## 假设

检验问题：预测效果差的数据集，是否因为特征提取没有抓住与 SOH/RUL 足够相关的信息；预测效果好的数据集，是否具有更强的特征-目标相关性。

本分析排除 `MIX_large`（组合数据集，混合多个物理数据源分布）。

## 高层指标定义

| 指标 | 含义 |
|------|------|
| `mean_abs_corr` | 数据集内 16 个特征与目标的平均 \|Spearman\| |
| `max_abs_corr` | 最强单个特征与目标的 \|Spearman\| |
| `top3_abs_corr` | 最强 3 个特征与目标的平均 \|Spearman\| |
| `strong_feature_fraction` | \|Spearman\| ≥ 0.7 的特征比例 |
| `performance_score` | 由 MAPE、RMSE、R²、Pearson_r 的秩分数合成，越高越好 |

## 全局检验结果

### SOH
- `feature_quality_score` vs `performance_score`：Spearman = 0.070，n = 24
- `max_abs_corr` vs `mean_mape`：Spearman = -0.349，n = 24
- `max_abs_corr` vs `mean_r2`：Spearman = 0.471，n = 24
- `top3_abs_corr` vs `performance_score`：Spearman = 0.103，n = 24
- `mean_abs_corr` vs `performance_score`：Spearman = -0.148，n = 24
- 象限分布：高相关-好预测 = 7；高相关-差预测 = 6；低相关-好预测 = 6；低相关-差预测 = 5

### RUL
- `feature_quality_score` vs `performance_score`：Spearman = 0.548，n = 24
- `max_abs_corr` vs `mean_mape`：Spearman = -0.774，n = 24
- `max_abs_corr` vs `mean_r2`：Spearman = 0.787，n = 24
- `top3_abs_corr` vs `performance_score`：Spearman = 0.545，n = 24
- `mean_abs_corr` vs `performance_score`：Spearman = 0.482，n = 24
- 象限分布：高相关-好预测 = 8；低相关-差预测 = 6；高相关-差预测 = 6；低相关-好预测 = 4

## 结论

假设**部分成立**，但非无条件：

- **RUL**：最强特征相关性与 MAPE 明显负相关，与 R² / 综合表现正相关。支持"特征更相关，数据集更容易预测"。
- **SOH**：最强特征相关性与 MAE/RMSE 负相关，与 R²/Pearson_r 正相关，同样支持假设。
- **平均相关性**的解释力弱于**最大相关性**，说明模型通常只需少数关键特征即可取得较好表现，不要求全部 16 个特征都强相关。
- 存在"高相关但差预测"的数据集，提示误差还受样本量、标签尺度、训练划分、模型结构、噪声、分布偏移等因素影响。

## 输出文件

- 图表：`figures_dataset_level/`（散点图、排序柱状图、象限图）
- 表格：`dataset_level_hypothesis_summary.csv`、`dataset_level_hypothesis_correlations.csv`、`dataset_level_hypothesis_quadrants.csv`
