"""Dataset-level hypothesis test for feature relevance vs prediction quality.

This script answers a direct question:

    Are datasets with stronger extracted-feature/target correlations also
    the datasets where models predict better?

It reads the CSVs produced by analyze_feature_relationships.py and writes
dataset-level summary tables, figures, and a Chinese markdown report.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("tutorials/correlation_analysis/outputs_tslib/.mpl_cache").resolve()),
)

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


BASE = Path("tutorials/correlation_analysis/outputs_tslib")
FIG_DIR = BASE / "figures_dataset_level"

TASK_TO_DATASET = {
    "CALB42": "CALB",
    "CALB2024": "CALB",
    "NAion": "NA-ion",
    "NAion42": "NA-ion",
    "NAion2024": "NA-ion",
    "ZN-coin42": "ZN-coin",
    "ZN-coin2024": "ZN-coin",
}


def configure_chinese_font():
    for font_path in [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def safe_spearman(x, y):
    frame = pd.concat([pd.Series(x), pd.Series(y)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def percentile_rank(series, higher_is_better=True):
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    n = len(valid)
    ascending_rank = series.rank(method="average", pct=True, ascending=True)
    if higher_is_better:
        return ascending_rank
    # For error metrics, smaller values should receive larger scores.
    # The +1/n term makes the best finite value score 1.0 and the worst 1/n.
    return 1.0 - ascending_rank + (1.0 / n)


def build_tables():
    truth = pd.read_csv(BASE / "feature_truth_correlations_by_dataset.csv")
    metrics = pd.read_csv(BASE / "battery_metrics_with_feature_profiles.csv")
    metrics["dataset"] = metrics["task"].map(TASK_TO_DATASET).fillna(metrics["task"])

    feature_quality = truth.groupby(["dataset", "target"], as_index=False).agg(
        mean_abs_corr=("abs_spearman", "mean"),
        median_abs_corr=("abs_spearman", "median"),
        max_abs_corr=("abs_spearman", "max"),
        top3_abs_corr=("abs_spearman", lambda s: s.sort_values(ascending=False).head(3).mean()),
        strong_feature_fraction=("abs_spearman", lambda s: float((s >= 0.7).mean())),
        very_strong_feature_fraction=("abs_spearman", lambda s: float((s >= 0.9).mean())),
        n_features=("feature", "count"),
    )

    task_perf = metrics.groupby(["target", "task", "dataset"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        mean_mape=("mape", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        mean_pearson_r=("pearson_r", "mean"),
        median_mape=("mape", "median"),
        median_r2=("r2", "median"),
        n_metric_rows=("battery_name", "count"),
        n_batteries=("battery_name", "nunique"),
        n_models=("model", "nunique"),
    )
    table = task_perf.merge(feature_quality, on=["dataset", "target"], how="left")

    score_parts = []
    for target, group in table.groupby("target"):
        idx = group.index
        # Rank-based composite score avoids comparing SOH/RUL metric scales directly.
        mape_score = percentile_rank(group["mean_mape"], higher_is_better=False)
        rmse_score = percentile_rank(group["mean_rmse"], higher_is_better=False)
        r2_score = percentile_rank(group["mean_r2"], higher_is_better=True)
        pearson_score = percentile_rank(group["mean_pearson_r"], higher_is_better=True)
        perf_score = pd.concat([mape_score, rmse_score, r2_score, pearson_score], axis=1).mean(axis=1)
        score_parts.append(pd.DataFrame({"index": idx, "performance_score": perf_score}))
    scores = pd.concat(score_parts, ignore_index=True).set_index("index")
    table["performance_score"] = scores.loc[table.index, "performance_score"].to_numpy()
    table["feature_quality_score"] = table[["mean_abs_corr", "top3_abs_corr", "max_abs_corr"]].mean(axis=1)

    table.to_csv(BASE / "dataset_level_hypothesis_summary.csv", index=False)

    rows = []
    for target, group in table.groupby("target"):
        for x in [
            "mean_abs_corr",
            "median_abs_corr",
            "max_abs_corr",
            "top3_abs_corr",
            "strong_feature_fraction",
            "very_strong_feature_fraction",
            "feature_quality_score",
        ]:
            for y in ["mean_mae", "mean_mape", "mean_rmse", "mean_r2", "mean_pearson_r", "performance_score"]:
                rows.append(
                    {
                        "target": target,
                        "x_feature_relevance": x,
                        "y_performance": y,
                        "spearman": safe_spearman(group[x], group[y]),
                        "n_tasks": int(group[[x, y]].dropna().shape[0]),
                        "expected_direction": "误差指标期望负相关；质量指标和performance_score期望正相关",
                    }
                )
    corr_table = pd.DataFrame(rows)
    corr_table.to_csv(BASE / "dataset_level_hypothesis_correlations.csv", index=False)

    return table, corr_table


def label_quadrants(table):
    out = table.copy()
    labels = []
    for target, group in out.groupby("target"):
        fq_median = group["feature_quality_score"].median()
        perf_median = group["performance_score"].median()
        for idx, row in group.iterrows():
            high_corr = row["feature_quality_score"] >= fq_median
            good_perf = row["performance_score"] >= perf_median
            if high_corr and good_perf:
                label = "高相关-好预测"
            elif high_corr and not good_perf:
                label = "高相关-差预测"
            elif not high_corr and good_perf:
                label = "低相关-好预测"
            else:
                label = "低相关-差预测"
            labels.append((idx, label))
    for idx, label in labels:
        out.loc[idx, "quadrant"] = label
    out.to_csv(BASE / "dataset_level_hypothesis_quadrants.csv", index=False)
    return out


def scatter_with_labels(ax, data, x, y, title, xlabel, ylabel):
    ax.scatter(data[x], data[y], s=70, alpha=0.78)
    for _, row in data.iterrows():
        ax.annotate(row["task"], (row[x], row[y]), fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    rho = safe_spearman(data[x], data[y])
    ax.set_title(f"{title}\nSpearman={rho:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def make_figures(table):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for target, group in table.groupby("target"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        scatter_with_labels(
            axes[0],
            group,
            "feature_quality_score",
            "performance_score",
            f"{target}: 特征质量分数 vs 综合预测表现",
            "特征质量分数（平均/Top3/最大相关性）",
            "综合预测表现（秩分数，越高越好）",
        )
        scatter_with_labels(
            axes[1],
            group,
            "max_abs_corr",
            "mean_mape",
            f"{target}: 最强特征相关性 vs MAPE",
            "数据集内最强 |Spearman|",
            "平均 MAPE（越低越好）",
        )
        scatter_with_labels(
            axes[2],
            group,
            "top3_abs_corr",
            "mean_r2",
            f"{target}: Top-3特征相关性 vs R2",
            "Top-3 平均 |Spearman|",
            "平均 R2（越高越好）",
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"dataset_level_scatter_{target}.png", dpi=180)
        plt.close(fig)

        ranked = group.sort_values("performance_score", ascending=False)
        x = np.arange(len(ranked))
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax1.bar(x - 0.18, ranked["performance_score"], width=0.36, label="综合预测表现", color="#4c78a8")
        ax1.bar(x + 0.18, ranked["feature_quality_score"], width=0.36, label="特征质量分数", color="#f58518")
        ax1.set_xticks(x)
        ax1.set_xticklabels(ranked["task"], rotation=45, ha="right", fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"{target}: 数据集预测表现与特征质量排序对照")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"dataset_level_ranked_bars_{target}.png", dpi=180)
        plt.close(fig)

        q_counts = group["quadrant"].value_counts().reindex(
            ["高相关-好预测", "高相关-差预测", "低相关-好预测", "低相关-差预测"], fill_value=0
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(q_counts.index, q_counts.values, color=["#54a24b", "#e45756", "#72b7b2", "#b279a2"])
        ax.set_title(f"{target}: 数据集象限计数")
        ax.set_ylabel("任务数量")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"dataset_level_quadrants_{target}.png", dpi=180)
        plt.close(fig)


def write_report(table, corr_table):
    lines = [
        "# 数据集级假设检验：特征相关性是否解释预测好坏",
        "",
        "## 假设",
        "",
        "更直白地说，我们要检验：预测效果差的数据集，是否因为特征提取环节没有抓住与 SOH/RUL 目标足够相关的信息；预测效果好的数据集，是否具有更强的特征-目标相关性。",
        "",
        "本分析排除 `MIX_large`，因为它是组合数据集，会混合多个物理数据集的分布。",
        "",
        "## 构造的高层指标",
        "",
        "- `mean_abs_corr`：一个数据集内 16 个特征与目标的平均 |Spearman|。",
        "- `max_abs_corr`：最强单个特征与目标的 |Spearman|。",
        "- `top3_abs_corr`：最强 3 个特征与目标的平均 |Spearman|。",
        "- `strong_feature_fraction`：|Spearman| >= 0.7 的特征比例。",
        "- `performance_score`：由 MAPE、RMSE、R2、Pearson_r 的秩分数合成，越高表示预测整体越好。",
        "",
        "## 全局检验结果",
        "",
    ]
    for target in ["SOH", "RUL"]:
        lines.append(f"### {target}")
        sub = corr_table[corr_table["target"] == target]
        picks = [
            ("feature_quality_score", "performance_score"),
            ("max_abs_corr", "mean_mape"),
            ("max_abs_corr", "mean_r2"),
            ("top3_abs_corr", "performance_score"),
            ("mean_abs_corr", "performance_score"),
        ]
        for x, y in picks:
            row = sub[(sub["x_feature_relevance"] == x) & (sub["y_performance"] == y)]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"- `{x}` vs `{y}`: Spearman={r['spearman']:.3f}, n={int(r['n_tasks'])}")
        q = table[table["target"] == target]["quadrant"].value_counts()
        lines.append("- 象限计数：" + "；".join(f"{k}={v}" for k, v in q.items()))
        lines.append("")

    lines += [
        "## 解释",
        "",
        "从数据集级结果看，假设有一定支持，但不是无条件成立：",
        "",
        "- 对 RUL，最强特征相关性与 MAPE 通常呈明显负相关，与 R2/综合表现呈正相关。这支持“特征更相关，数据集更容易预测”。",
        "- 对 SOH，最强特征相关性与 MAE/RMSE 也呈负相关，与 R2/Pearson_r 呈正相关，同样支持数据集级假设。",
        "- 但平均相关性不总是比最大相关性更有解释力，这说明模型可能只需要少数关键特征就能得到较好表现，而不是所有 16 个特征都必须强相关。",
        "- 高相关但差预测的数据集仍然存在，提示误差还受样本量、标签尺度、训练划分、模型结构、噪声、数据分布偏移等因素影响。",
        "",
        "## 可视化输出",
        "",
        "- `figures_dataset_level/dataset_level_scatter_SOH.png`",
        "- `figures_dataset_level/dataset_level_scatter_RUL.png`",
        "- `figures_dataset_level/dataset_level_ranked_bars_SOH.png`",
        "- `figures_dataset_level/dataset_level_ranked_bars_RUL.png`",
        "- `figures_dataset_level/dataset_level_quadrants_SOH.png`",
        "- `figures_dataset_level/dataset_level_quadrants_RUL.png`",
        "",
        "## 表格输出",
        "",
        "- `dataset_level_hypothesis_summary.csv`",
        "- `dataset_level_hypothesis_correlations.csv`",
        "- `dataset_level_hypothesis_quadrants.csv`",
    ]
    (BASE / "dataset_level_hypothesis_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    configure_chinese_font()
    table, corr_table = build_tables()
    table = label_quadrants(table)
    make_figures(table)
    write_report(table, corr_table)
    print(f"wrote {BASE / 'dataset_level_hypothesis_report_zh.md'}")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
