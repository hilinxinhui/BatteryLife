"""Model-fixed ranking test for feature relevance vs prediction quality.

This is a small, direct test:

1. Fix target and model.
2. Rank datasets from best to worst prediction performance.
3. Attach dataset-level feature/target correlations.
4. Check whether feature relevance decreases as performance gets worse.
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
FIG_DIR = BASE / "figures_model_fixed_rank"

TASK_TO_DATASET = {
    "CALB42": "CALB",
    "CALB2024": "CALB",
    "NAion": "NA-ion",
    "NAion42": "NA-ion",
    "NAion2024": "NA-ion",
    "ZN-coin42": "ZN-coin",
    "ZN-coin2024": "ZN-coin",
}


def configure_chinese_font() -> None:
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


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([pd.Series(x), pd.Series(y)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    n = len(valid)
    ascending_rank = series.rank(method="average", pct=True, ascending=True)
    if higher_is_better:
        return ascending_rank
    return 1.0 - ascending_rank + (1.0 / n)


def build_feature_quality() -> pd.DataFrame:
    truth = pd.read_csv(BASE / "feature_truth_correlations_by_dataset.csv")
    out = truth.groupby(["dataset", "target"], as_index=False).agg(
        mean_abs_corr=("abs_spearman", "mean"),
        median_abs_corr=("abs_spearman", "median"),
        max_abs_corr=("abs_spearman", "max"),
        top3_abs_corr=("abs_spearman", lambda s: s.sort_values(ascending=False).head(3).mean()),
        strong_feature_fraction=("abs_spearman", lambda s: float((s >= 0.7).mean())),
        n_features=("feature", "count"),
    )
    out["feature_quality_score"] = out[["mean_abs_corr", "top3_abs_corr", "max_abs_corr"]].mean(axis=1)
    return out


def build_ranked_table() -> pd.DataFrame:
    metrics = pd.read_csv(BASE / "battery_metrics_with_feature_profiles.csv")
    metrics = metrics[metrics["task"] != "MIX_large"].copy()
    metrics["dataset"] = metrics["task"].map(TASK_TO_DATASET).fillna(metrics["task"])
    feature_quality = build_feature_quality()

    grouped = metrics.groupby(["target", "model", "task", "dataset"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        mean_mape=("mape", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        mean_pearson_r=("pearson_r", "mean"),
        n_batteries=("battery_name", "nunique"),
        n_metric_rows=("battery_name", "count"),
    )

    score_frames = []
    for (target, model), group in grouped.groupby(["target", "model"]):
        score = pd.concat(
            [
                percentile_rank(group["mean_mape"], higher_is_better=False),
                percentile_rank(group["mean_rmse"], higher_is_better=False),
                percentile_rank(group["mean_r2"], higher_is_better=True),
                percentile_rank(group["mean_pearson_r"], higher_is_better=True),
            ],
            axis=1,
        ).mean(axis=1)
        score_frames.append(pd.DataFrame({"index": group.index, "performance_score": score}))

    scores = pd.concat(score_frames, ignore_index=True).set_index("index")
    grouped["performance_score"] = scores.loc[grouped.index, "performance_score"].to_numpy()
    ranked = grouped.merge(feature_quality, on=["dataset", "target"], how="left")
    ranked = ranked.sort_values(["target", "model", "performance_score"], ascending=[True, True, False])
    ranked["performance_rank"] = ranked.groupby(["target", "model"])["performance_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return ranked


def build_trend_table(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in ranked.groupby(["target", "model"]):
        row = {
            "target": target,
            "model": model,
            "n_tasks": len(group),
            "rank_vs_mean_abs_corr": safe_spearman(group["performance_rank"], group["mean_abs_corr"]),
            "rank_vs_top3_abs_corr": safe_spearman(group["performance_rank"], group["top3_abs_corr"]),
            "rank_vs_max_abs_corr": safe_spearman(group["performance_rank"], group["max_abs_corr"]),
            "rank_vs_feature_quality_score": safe_spearman(group["performance_rank"], group["feature_quality_score"]),
            "performance_vs_feature_quality_score": safe_spearman(
                group["performance_score"], group["feature_quality_score"]
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_top_bottom_table(ranked: pd.DataFrame, n_each: int = 3) -> pd.DataFrame:
    rows = []
    for (target, model), group in ranked.groupby(["target", "model"]):
        best = group.nsmallest(n_each, "performance_rank").copy()
        worst = group.nlargest(n_each, "performance_rank").copy()
        best["group"] = "性能最好"
        worst["group"] = "性能最差"
        rows.append(pd.concat([best, worst], ignore_index=True))
    return pd.concat(rows, ignore_index=True)


def make_figures(ranked: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for (target, model), group in ranked.groupby(["target", "model"]):
        group = group.sort_values("performance_rank")
        x = np.arange(len(group))
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax1.plot(x, group["performance_score"], marker="o", label="预测性能分数", color="#4c78a8")
        ax1.plot(x, group["feature_quality_score"], marker="o", label="特征质量分数", color="#f58518")
        ax1.plot(x, group["max_abs_corr"], marker="o", label="最强特征 |Spearman|", color="#54a24b")
        ax1.set_xticks(x)
        ax1.set_xticklabels(group["task"], rotation=45, ha="right", fontsize=8)
        ax1.set_ylim(0, 1.05)
        rho = safe_spearman(group["performance_rank"], group["feature_quality_score"])
        ax1.set_title(f"{target}-{model}: 固定模型后，从好到差排序的特征相关性变化\nrank vs 特征质量 Spearman={rho:.3f}")
        ax1.set_xlabel("数据集（按预测性能从好到差排序）")
        ax1.set_ylabel("分数 / 相关性")
        ax1.legend()
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"ranked_feature_relevance_{target}_{model}.png", dpi=180)
        plt.close(fig)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(ranked: pd.DataFrame, trends: pd.DataFrame, top_bottom: pd.DataFrame) -> None:
    lines = [
        "# 固定模型后的数据集排序检验",
        "",
        "## 检验问题",
        "",
        "固定同一个模型后，把数据集按预测性能从好到差排序，并观察这些数据集上的特征-真值相关性是否同步下降。",
        "",
        "这里的 `performance_score` 是在同一 `target + model` 内由 MAPE、RMSE、R2、Pearson_r 的秩分数合成，越高越好。",
        "`performance_rank=1` 表示该模型在该目标上的最佳数据集。",
        "",
        "如果假设成立，则 `performance_rank` 与 `feature_quality_score`、`max_abs_corr`、`top3_abs_corr` 的 Spearman 相关应倾向于负值。",
        "",
        "## 趋势概览",
        "",
    ]

    for target in ["SOH", "RUL"]:
        sub = trends[trends["target"] == target].copy()
        lines.append(f"### {target}")
        for col in ["rank_vs_feature_quality_score", "rank_vs_max_abs_corr", "rank_vs_top3_abs_corr"]:
            vals = sub[col].dropna()
            neg_count = int((vals < 0).sum())
            lines.append(f"- `{col}`：平均 Spearman={vals.mean():.3f}；负相关模型数={neg_count}/{len(vals)}")
        lines.append("")

    lines += [
        "## 每个模型的排序趋势",
        "",
        frame_to_markdown(trends.round(4)),
        "",
        "## 最好/最差数据集对照",
        "",
        "下面每个 `target + model` 取性能最好的 3 个和最差的 3 个数据集，便于直观看差异。",
        "",
        top_bottom[
            [
                "target",
                "model",
                "group",
                "performance_rank",
                "task",
                "performance_score",
                "mean_mape",
                "mean_r2",
                "feature_quality_score",
                "mean_abs_corr",
                "top3_abs_corr",
                "max_abs_corr",
            ]
        ]
        .round(4)
        .pipe(frame_to_markdown),
        "",
        "## 输出文件",
        "",
        "- `model_fixed_dataset_rank_feature_table.csv`：完整排序表。",
        "- `model_fixed_rank_trend_summary.csv`：每个模型内的递减趋势检验。",
        "- `model_fixed_top_bottom_feature_comparison.csv`：每个模型最好 3 个与最差 3 个数据集。",
        "- `figures_model_fixed_rank/`：每个模型一张排序折线图。",
    ]
    (BASE / "model_fixed_rank_feature_test_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_chinese_font()
    ranked = build_ranked_table()
    trends = build_trend_table(ranked)
    top_bottom = build_top_bottom_table(ranked)

    ranked.to_csv(BASE / "model_fixed_dataset_rank_feature_table.csv", index=False)
    trends.to_csv(BASE / "model_fixed_rank_trend_summary.csv", index=False)
    top_bottom.to_csv(BASE / "model_fixed_top_bottom_feature_comparison.csv", index=False)
    make_figures(ranked)
    write_report(ranked, trends, top_bottom)

    print(f"wrote {BASE / 'model_fixed_rank_feature_test_zh.md'}")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
