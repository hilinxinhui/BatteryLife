"""Prepare Chinese report assets and notebook for correlation analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tutorials/correlation_analysis/outputs_tslib/.mpl_cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


BASE = Path("tutorials/correlation_analysis/outputs_tslib")
FIG_DIR = BASE / "figures"
NB_PATH = Path("tutorials/correlation_analysis/feature_correlation_analysis_zh.ipynb")


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
            return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return None

FEATURE_ORDER = [
    "voltage_mean",
    "voltage_std",
    "voltage_kurtosis",
    "voltage_skewness",
    "CC_Q",
    "CC_charge_time",
    "voltage_slope",
    "voltage_entropy",
    "current_mean",
    "current_std",
    "current_kurtosis",
    "current_skewness",
    "CV_Q",
    "CV_charge_time",
    "current_slope",
    "current_entropy",
]

TASK_TO_DATASET = {
    "CALB42": "CALB",
    "CALB2024": "CALB",
    "NAion": "NA-ion",
    "NAion42": "NA-ion",
    "NAion2024": "NA-ion",
    "ZN-coin42": "ZN-coin",
    "ZN-coin2024": "ZN-coin",
}


def safe_spearman(x, y):
    frame = pd.concat([pd.Series(x), pd.Series(y)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def load_tables():
    truth = pd.read_csv(BASE / "feature_truth_correlations_by_dataset.csv")
    pred = pd.read_csv(BASE / "feature_prediction_error_relationships_available_npz.csv")
    metrics = pd.read_csv(BASE / "battery_metrics_with_feature_profiles.csv")
    pred["dataset"] = pred["task"].map(TASK_TO_DATASET).fillna(pred["task"])
    metrics["dataset"] = metrics["task"].map(TASK_TO_DATASET).fillna(metrics["task"])
    return truth, pred, metrics


def build_hypothesis_tables(truth, pred, metrics):
    pred_feature = pred.groupby(["target", "task", "dataset", "model", "feature"], as_index=False).agg(
        mean_abs_error_sensitivity=("feature_abs_error_spearman", lambda s: s.abs().mean()),
        mean_signed_error_sensitivity=("feature_abs_error_spearman", "mean"),
        mean_abs_rel_error_sensitivity=("feature_rel_error_spearman", lambda s: s.abs().mean()),
        mean_abs_pred_link=("feature_pred_spearman", lambda s: s.abs().mean()),
        n_batteries=("battery_name", "nunique"),
        mean_rows=("n_rows", "mean"),
    )
    feature_hyp = pred_feature.merge(
        truth[["dataset", "target", "feature", "spearman", "abs_spearman"]],
        on=["dataset", "target", "feature"],
        how="left",
    ).rename(columns={"spearman": "feature_truth_spearman", "abs_spearman": "feature_truth_abs_spearman"})
    feature_hyp.to_csv(BASE / "hypothesis_feature_error_sensitivity.csv", index=False)

    relevance = truth.groupby(["dataset", "target"], as_index=False).agg(
        mean_feature_truth_corr=("abs_spearman", "mean"),
        max_feature_truth_corr=("abs_spearman", "max"),
        top3_feature_truth_corr=("abs_spearman", lambda s: s.sort_values(ascending=False).head(3).mean()),
        n_valid_feature_corr=("abs_spearman", "count"),
    )
    performance = metrics.groupby(["target", "task", "dataset", "model"], as_index=False).agg(
        mae=("mae", "mean"),
        mape=("mape", "mean"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
        pearson_r=("pearson_r", "mean"),
        n_batteries=("battery_name", "nunique"),
    )
    dataset_model_hyp = performance.merge(relevance, on=["dataset", "target"], how="left")
    dataset_model_hyp.to_csv(BASE / "hypothesis_dataset_model_performance.csv", index=False)

    rows = []
    for target, sub in feature_hyp.groupby("target"):
        rows.append(
            {
                "level": "feature_error_sensitivity",
                "target": target,
                "x": "feature_truth_abs_spearman",
                "y": "mean_abs_error_sensitivity",
                "spearman": safe_spearman(sub["feature_truth_abs_spearman"], sub["mean_abs_error_sensitivity"]),
                "n": int(sub[["feature_truth_abs_spearman", "mean_abs_error_sensitivity"]].dropna().shape[0]),
                "interpretation": "正值表示越相关的特征越容易成为误差敏感方向；负值才支持“相关性越高误差越小”。",
            }
        )
    for target, sub in dataset_model_hyp.groupby("target"):
        for rel_col in ["mean_feature_truth_corr", "max_feature_truth_corr", "top3_feature_truth_corr"]:
            for metric in ["mae", "mape", "rmse", "r2", "pearson_r"]:
                rows.append(
                    {
                        "level": "dataset_model_performance",
                        "target": target,
                        "x": rel_col,
                        "y": metric,
                        "spearman": safe_spearman(sub[rel_col], sub[metric]),
                        "n": int(sub[[rel_col, metric]].dropna().shape[0]),
                        "interpretation": "误差指标期望负相关，拟合质量指标 r2/pearson_r 期望正相关。",
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(BASE / "hypothesis_test_summary.csv", index=False)
    return feature_hyp, dataset_model_hyp, summary


def heatmap(matrix, row_labels, col_labels, title, path, vmin=0.0, vmax=1.0, cmap="viridis"):
    height = max(5, 0.32 * len(row_labels))
    width = max(10, 0.48 * len(col_labels))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(truth, pred, feature_hyp, dataset_model_hyp):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for target in ["SOH", "RUL"]:
        pivot = truth[truth["target"] == target].pivot(index="dataset", columns="feature", values="abs_spearman")
        pivot = pivot.reindex(columns=FEATURE_ORDER)
        heatmap(
            pivot.to_numpy(),
            pivot.index.tolist(),
            pivot.columns.tolist(),
            f"{target}: 每个数据集/每个特征与真值的 |Spearman|",
            FIG_DIR / f"truth_corr_heatmap_{target}.png",
        )

        err = (
            feature_hyp[feature_hyp["target"] == target]
            .groupby(["task", "feature"])["mean_abs_error_sensitivity"]
            .mean()
            .reset_index()
            .pivot(index="task", columns="feature", values="mean_abs_error_sensitivity")
            .reindex(columns=FEATURE_ORDER)
        )
        heatmap(
            err.to_numpy(),
            err.index.tolist(),
            err.columns.tolist(),
            f"{target}: 每个任务/每个特征与绝对误差的平均 |Spearman|",
            FIG_DIR / f"error_sensitivity_heatmap_{target}.png",
        )

        for dataset, sub in truth[truth["target"] == target].groupby("dataset"):
            sub = sub.set_index("feature").reindex(FEATURE_ORDER)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(np.arange(len(FEATURE_ORDER)), sub["abs_spearman"].to_numpy())
            ax.set_title(f"{target} - {dataset}: 特征与真值的 |Spearman|")
            ax.set_ylabel("|Spearman|")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(np.arange(len(FEATURE_ORDER)))
            ax.set_xticklabels(FEATURE_ORDER, rotation=45, ha="right", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"truth_corr_bar_{target}_{dataset}.png", dpi=160)
            plt.close(fig)

        sub = feature_hyp[feature_hyp["target"] == target].dropna(
            subset=["feature_truth_abs_spearman", "mean_abs_error_sensitivity"]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub["feature_truth_abs_spearman"], sub["mean_abs_error_sensitivity"], s=16, alpha=0.45)
        rho = safe_spearman(sub["feature_truth_abs_spearman"], sub["mean_abs_error_sensitivity"])
        ax.set_title(f"{target}: 特征-真值相关性 vs 特征-误差敏感度 (rho={rho:.3f})")
        ax.set_xlabel("特征与真值的 |Spearman|")
        ax.set_ylabel("特征与绝对误差的平均 |Spearman|")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"feature_corr_vs_error_sensitivity_{target}.png", dpi=180)
        plt.close(fig)

        sub = dataset_model_hyp[dataset_model_hyp["target"] == target].dropna(
            subset=["max_feature_truth_corr", "mape"]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub["max_feature_truth_corr"], sub["mape"], s=28, alpha=0.6)
        rho = safe_spearman(sub["max_feature_truth_corr"], sub["mape"])
        ax.set_title(f"{target}: 数据集最强特征相关性 vs MAPE (rho={rho:.3f})")
        ax.set_xlabel("数据集内最强特征-真值 |Spearman|")
        ax.set_ylabel("平均 MAPE")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"dataset_max_corr_vs_mape_{target}.png", dpi=180)
        plt.close(fig)


def write_chinese_findings(summary):
    def get(level, target, x, y):
        row = summary[(summary["level"] == level) & (summary["target"] == target) & (summary["x"] == x) & (summary["y"] == y)]
        return float(row.iloc[0]["spearman"]) if not row.empty else np.nan

    text = f"""# 关键发现（中文）

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
   - SOH：数据集内最强特征-真值相关性与 MAE 的 Spearman 为 {get('dataset_model_performance','SOH','max_feature_truth_corr','mae'):.3f}，与 RMSE 为 {get('dataset_model_performance','SOH','max_feature_truth_corr','rmse'):.3f}，与 Pearson 预测质量为 {get('dataset_model_performance','SOH','max_feature_truth_corr','pearson_r'):.3f}。
   - RUL：数据集内最强特征-真值相关性与 MAPE 的 Spearman 为 {get('dataset_model_performance','RUL','max_feature_truth_corr','mape'):.3f}，与 R2 为 {get('dataset_model_performance','RUL','max_feature_truth_corr','r2'):.3f}。
   - 这些结果说明，当某个数据集存在非常强的特征-标签单调关系时，模型通常更容易获得较好的相对误差或拟合质量。

2. 在“单个特征是否导致更小逐样本误差”层面，结论不成立，甚至会反过来。
   - SOH：特征-真值相关性与特征-绝对误差敏感度的 Spearman 为 {get('feature_error_sensitivity','SOH','feature_truth_abs_spearman','mean_abs_error_sensitivity'):.3f}。
   - RUL：对应 Spearman 为 {get('feature_error_sensitivity','RUL','feature_truth_abs_spearman','mean_abs_error_sensitivity'):.3f}。
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
"""
    (BASE / "key_findings_zh.md").write_text(text, encoding="utf-8")


def nb_markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def nb_code(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}


def write_notebook():
    cells = [
        nb_markdown(
            "# BatteryLife 特征相关性与模型误差分析\n\n"
            "本 Notebook 面向中文读者，说明本次分析如何联动当前 BatteryLife 仓库与 `SOHbenchmark_TSLib` 仓库，"
            "并复现以下三类问题：\n\n"
            "1. 每个数据集、每个特征与 SOH/RUL 真值之间的相关性；\n"
            "2. 每个特征与模型输出、绝对误差、相对误差之间的关系；\n"
            "3. 特征相关性结构是否能映射到模型性能，尤其是“相关性越高，模型潜在误差是否越小”。\n"
        ),
        nb_markdown(
            "## 指标定义\n\n"
            "- **Pearson 相关系数**：衡量两个变量的线性关系，取值范围 [-1, 1]。绝对值越大，线性关系越强。\n"
            "- **Spearman 秩相关系数**：先把变量转换成秩，再计算 Pearson，衡量单调关系。它比 Pearson 更适合电池退化这种非线性但单调的过程。\n"
            "- **|Spearman|**：忽略方向，只衡量相关强度。例如 -0.9 和 0.9 都表示很强的单调关系。\n"
            "- **特征-误差敏感度**：对每个 `res.npz` 中的逐样本预测，计算某个特征与 `|y_pred - y_true|` 的 Spearman。其绝对值越大，说明误差更强地沿该特征方向变化。\n"
            "- **相对误差敏感度**：把绝对误差换成 `|y_pred - y_true| / |y_true|` 后重复上述计算。\n"
            "- **数据集层面的相关性画像**：对一个数据集的 16 个特征，计算平均、最大、Top-3 平均的 |Spearman|，再与模型的 MAE/MAPE/RMSE/R2/Pearson 进行二级相关分析。\n"
        ),
        nb_code(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "BASE = Path('outputs_tslib')\n"
            "if not BASE.exists():\n"
            "    BASE = Path('tutorials/correlation_analysis/outputs_tslib')\n"
            "FIG = BASE / 'figures'\n"
            "truth = pd.read_csv(BASE / 'feature_truth_correlations_by_dataset.csv')\n"
            "pred_rel = pd.read_csv(BASE / 'feature_prediction_error_relationships_available_npz.csv')\n"
            "feature_hyp = pd.read_csv(BASE / 'hypothesis_feature_error_sensitivity.csv')\n"
            "dataset_hyp = pd.read_csv(BASE / 'hypothesis_dataset_model_performance.csv')\n"
            "summary = pd.read_csv(BASE / 'hypothesis_test_summary.csv')\n"
            "truth.head()"
        ),
        nb_markdown("## 每个数据集、每个特征与真值的相关性热图\n\n下面两张图分别对应 SOH 和 RUL。行是数据集，列是 16 个特征，颜色表示 `|Spearman|`。"),
        nb_code(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIG / 'truth_corr_heatmap_SOH.png')))\n"
            "display(Image(filename=str(FIG / 'truth_corr_heatmap_RUL.png')))"
        ),
        nb_markdown("## 遍历每个数据集：每个特征的相关性柱状图\n\n下面的代码会按目标和数据集遍历，逐一显示 16 个特征与真值的 `|Spearman|`。如需保存，图片已经在 `outputs_tslib/figures/` 中生成。"),
        nb_code(
            "for target in ['SOH', 'RUL']:\n"
            "    print(f'===== {target} =====')\n"
            "    for dataset in sorted(truth['dataset'].unique()):\n"
            "        path = FIG / f'truth_corr_bar_{target}_{dataset}.png'\n"
            "        if path.exists():\n"
            "            print(dataset)\n"
            "            display(Image(filename=str(path)))"
        ),
        nb_markdown("## 特征与模型误差的敏感度\n\n下面热图显示：每个任务、每个特征与绝对误差的平均 `|Spearman|`。颜色越深，说明模型误差越沿该特征方向变化。"),
        nb_code(
            "display(Image(filename=str(FIG / 'error_sensitivity_heatmap_SOH.png')))\n"
            "display(Image(filename=str(FIG / 'error_sensitivity_heatmap_RUL.png')))"
        ),
        nb_markdown("## 核心假设检验：相关性越高，误差是否越小？\n\n先看特征级别：横轴是特征与真值的 `|Spearman|`，纵轴是特征与绝对误差的平均 `|Spearman|`。如果“相关性越高，误差越小”在特征级别成立，我们应该看到负相关；如果是正相关，则说明高相关特征也可能是误差变化最敏感的方向。"),
        nb_code(
            "display(Image(filename=str(FIG / 'feature_corr_vs_error_sensitivity_SOH.png')))\n"
            "display(Image(filename=str(FIG / 'feature_corr_vs_error_sensitivity_RUL.png')))\n"
            "summary[summary['level'] == 'feature_error_sensitivity']"
        ),
        nb_markdown("## 数据集/模型层面的假设检验\n\n再看数据集层面：用一个数据集中最强特征-真值相关性，去解释不同模型在该任务上的平均 MAPE。这里更接近“这个数据集是否更容易预测”。"),
        nb_code(
            "display(Image(filename=str(FIG / 'dataset_max_corr_vs_mape_SOH.png')))\n"
            "display(Image(filename=str(FIG / 'dataset_max_corr_vs_mape_RUL.png')))\n"
            "summary[summary['level'] == 'dataset_model_performance'].sort_values('spearman', key=lambda s: s.abs(), ascending=False).head(20)"
        ),
        nb_markdown(
            "## 结论\n\n"
            "从当前结果看，不能简单说“单个特征相关性越高，模型逐样本误差越小”。"
            "在特征级别，高相关特征反而往往也是误差最敏感的方向之一。"
            "但是，在数据集整体层面，若某个数据集存在很强的特征-真值单调关系，模型通常会表现出更好的相对误差或拟合质量。"
            "因此更准确的表述是：高相关性提高了数据集层面的可预测性，但不保证逐样本误差单调降低。"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    configure_chinese_font()
    truth, pred, metrics = load_tables()
    feature_hyp, dataset_model_hyp, summary = build_hypothesis_tables(truth, pred, metrics)
    make_figures(truth, pred, feature_hyp, dataset_model_hyp)
    write_chinese_findings(summary)
    write_notebook()
    print(f"wrote {BASE / 'key_findings_zh.md'}")
    print(f"wrote {NB_PATH}")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
