# 固定模型后的数据集排序检验

## 检验问题

固定同一个模型，将数据集按预测性能从好到差排序，观察特征-真值相关性是否同步下降。

`performance_score` 在同一 `target + model` 内由 MAPE、RMSE、R²、Pearson_r 的秩分数合成，越高越好。`performance_rank = 1` 表示该模型在该目标上的最佳数据集。

若假设成立，`performance_rank` 与 `feature_quality_score`、`max_abs_corr`、`top3_abs_corr` 的 Spearman 应倾向于负值。

## 趋势概览

| 目标 | 指标 | 平均 Spearman | 负相关模型数/总数 |
|------|------|--------------:|------------------:|
| SOH | rank_vs_feature_quality_score | -0.092 | 4/5 |
| SOH | rank_vs_max_abs_corr | -0.633 | 5/5 |
| SOH | rank_vs_top3_abs_corr | -0.105 | 4/5 |
| RUL | rank_vs_feature_quality_score | -0.540 | 5/5 |
| RUL | rank_vs_max_abs_corr | -0.676 | 5/5 |
| RUL | rank_vs_top3_abs_corr | -0.536 | 5/5 |

## 各模型排序趋势

| 目标 | 模型 | 任务数 | mean_abs_corr | top3_abs_corr | max_abs_corr | feature_quality_score | performance_vs_feature_quality |
|------|------|--------|--------------:|--------------:|-------------:|----------------------:|------------------------------:|
| RUL | Attention | 24 | -0.4019 | -0.4612 | -0.6312 | -0.4691 | 0.4719 |
| RUL | CNN | 24 | -0.5772 | -0.6007 | -0.7219 | -0.6051 | 0.6051 |
| RUL | GRU | 24 | -0.4194 | -0.4769 | -0.6330 | -0.4821 | 0.4821 |
| RUL | LSTM | 24 | -0.5667 | -0.6234 | -0.7236 | -0.6286 | 0.6248 |
| RUL | MLP | 24 | -0.4568 | -0.5170 | -0.6696 | -0.5170 | 0.5170 |
| SOH | Attention | 24 | 0.2964 | 0.1264 | -0.4908 | 0.1430 | -0.1430 |
| SOH | CNN | 24 | 0.1630 | -0.0671 | -0.7036 | -0.0532 | 0.0462 |
| SOH | GRU | 24 | 0.1404 | -0.0227 | -0.6975 | -0.0375 | 0.0432 |
| SOH | LSTM | 24 | -0.0959 | -0.3348 | -0.4542 | -0.2703 | 0.2703 |
| SOH | MLP | 24 | -0.0314 | -0.2293 | -0.8213 | -0.2424 | 0.2394 |

## 最好/最差数据集对照

每个 `target + model` 取性能最好的 3 个和最差的 3 个数据集。完整对照见输出文件 `model_fixed_top_bottom_feature_comparison.csv`。

## 输出文件

- `model_fixed_dataset_rank_feature_table.csv`：完整排序表
- `model_fixed_rank_trend_summary.csv`：每个模型内的递减趋势检验
- `model_fixed_top_bottom_feature_comparison.csv`：最好 3 个与最差 3 个数据集对照
- `figures_model_fixed_rank/`：每个模型一张排序折线图
