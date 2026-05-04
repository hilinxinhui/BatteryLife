"""
Count samples (cells and cycles) for each of the 25 runnable datasets,
broken down by train / val / test split.

A "sample" here refers to one cycle row in the extracted feature CSV.
Each cell (battery) contributes multiple cycles.

The 25 datasets are:
  - 15 MIX_large constituents: HUST, MATR, SNL, RWTH, MICH, MICH_EXP,
    UL_PUR, CALCE, HNEI, Tongji1, Tongji2, Tongji3, Stanford, ISU_ILCC, XJTU
  - 9 standalone (incl. seen/unseen sub-variants): ZN-coin, ZN42, ZN2024,
    CALB, CALB42, CALB2024, NA-ion, NAion42, NAion2024
  - 1 combination: MIX_large
"""

import os
import sys

# Ensure we can import from the parent repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_provider.data_split_recorder import split_recorder

EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "..", "extracted_features")


def pkl_to_csv(pkl_name: str) -> str:
    """Map a .pkl cell filename to the corresponding .csv filename."""
    return pkl_name.replace(".pkl", ".csv")


def count_cycles(csv_path: str) -> int:
    """Count data rows (cycles) in a feature CSV file."""
    try:
        with open(csv_path, "r") as f:
            # Subtract 1 for the header row
            return sum(1 for _ in f) - 1
    except FileNotFoundError:
        return 0


def get_dataset_dir(cell_name: str) -> str:
    """
    Determine which extracted_features subdirectory a cell belongs to.
    Cell name prefixes map to dataset directories.
    """
    # MICH_EXP cells start with "MICH_" + digit; MICH cells start with "MICH_BL"/"MICH_MC"
    if cell_name.startswith("MICH_"):
        suffix = cell_name[len("MICH_"):]
        if suffix and suffix[0].isdigit():
            return "MICH_EXP"
        return "MICH"

    prefix_map = [
        ("HUST_", "HUST"),
        ("MATR_", "MATR"),
        ("SNL_", "SNL"),
        ("RWTH_", "RWTH"),
        ("UL-PUR_", "UL_PUR"),
        ("CALCE_", "CALCE"),
        ("HNEI_", "HNEI"),
        ("Tongji1_", "Tongji"),
        ("Tongji2_", "Tongji"),
        ("Tongji3_", "Tongji"),
        ("Stanford_", "Stanford"),
        ("ISU-ILCC_", "ISU_ILCC"),
        ("XJTU_", "XJTU"),
        ("ZN-coin_", "ZN-coin"),
        ("CALB_", "CALB"),
        ("NA-ion_", "NA-ion"),
    ]
    for prefix, ddir in prefix_map:
        if cell_name.startswith(prefix):
            return ddir
    return None


def count_split(cell_files, dataset_dir_override=None):
    """
    Count cells and total cycles for a list of cell .pkl filenames.

    Returns (num_cells, num_cycles, missing_cells).
    """
    total_cells = 0
    total_cycles = 0
    missing = 0

    for pkl_name in cell_files:
        csv_name = pkl_to_csv(pkl_name)
        if dataset_dir_override:
            csv_path = os.path.join(EXTRACTED_DIR, dataset_dir_override, csv_name)
        else:
            ds_dir = get_dataset_dir(pkl_name)
            if ds_dir is None:
                missing += 1
                continue
            csv_path = os.path.join(EXTRACTED_DIR, ds_dir, csv_name)

        cycles = count_cycles(csv_path)
        if cycles > 0:
            total_cells += 1
            total_cycles += cycles
        else:
            missing += 1

    return total_cells, total_cycles, missing


# ---------------------------------------------------------------------------
# Dataset definitions: name → (train_list, val_list, test_list, dir_override)
# dir_override is used when all cells are in one subdirectory (e.g. sub-variants)
# ---------------------------------------------------------------------------

def build_datasets():
    """Return list of (name, train_files, val_files, test_files, dir_override)."""
    ds = []

    # --- MIX_large constituents (15) ---
    ds.append(("HUST", split_recorder.HUST_train_files, split_recorder.HUST_val_files, split_recorder.HUST_test_files, "HUST"))
    ds.append(("MATR", split_recorder.MATR_train_files, split_recorder.MATR_val_files, split_recorder.MATR_test_files, "MATR"))
    ds.append(("SNL", split_recorder.SNL_train_files, split_recorder.SNL_val_files, split_recorder.SNL_test_files, "SNL"))
    ds.append(("RWTH", split_recorder.RWTH_train_files, split_recorder.RWTH_val_files, split_recorder.RWTH_test_files, "RWTH"))
    ds.append(("MICH", split_recorder.MICH_train_files, split_recorder.MICH_val_files, split_recorder.MICH_test_files, "MICH"))
    ds.append(("MICH_EXP", split_recorder.MICH_EXP_train_files, split_recorder.MICH_EXP_val_files, split_recorder.MICH_EXP_test_files, "MICH_EXP"))
    ds.append(("UL_PUR", split_recorder.UL_PUR_train_files, split_recorder.UL_PUR_val_files, split_recorder.UL_PUR_test_files, "UL_PUR"))
    ds.append(("CALCE", split_recorder.CALCE_train_files, split_recorder.CALCE_val_files, split_recorder.CALCE_test_files, "CALCE"))
    ds.append(("HNEI", split_recorder.HNEI_train_files, split_recorder.HNEI_val_files, split_recorder.HNEI_test_files, "HNEI"))
    ds.append(("Stanford", split_recorder.Stanford_train_files, split_recorder.Stanford_val_files, split_recorder.Stanford_test_files, "Stanford"))
    ds.append(("ISU_ILCC", split_recorder.ISU_ILCC_train_files, split_recorder.ISU_ILCC_val_files, split_recorder.ISU_ILCC_test_files, "ISU_ILCC"))
    ds.append(("XJTU", split_recorder.XJTU_train_files, split_recorder.XJTU_val_files, split_recorder.XJTU_test_files, "XJTU"))

    # Tongji split into Tongji1 / Tongji2 / Tongji3
    for tj_id in ("Tongji1", "Tongji2", "Tongji3"):
        prefix = tj_id + "_"
        train = [f for f in split_recorder.Tongji_train_files if f.startswith(prefix)]
        val = [f for f in split_recorder.Tongji_val_files if f.startswith(prefix)]
        test = [f for f in split_recorder.Tongji_test_files if f.startswith(prefix)]
        ds.append((tj_id, train, val, test, "Tongji"))

    # --- Standalone datasets + sub-variants (9) ---
    ds.append(("ZN-coin", split_recorder.ZNcoin_train_files, split_recorder.ZNcoin_val_files, split_recorder.ZNcoin_test_files, "ZN-coin"))
    ds.append(("ZN42", split_recorder.ZN_42_train_files, split_recorder.ZN_42_val_files, split_recorder.ZN_42_test_files, "ZN-coin"))
    ds.append(("ZN2024", split_recorder.ZN_2024_train_files, split_recorder.ZN_2024_val_files, split_recorder.ZN_2024_test_files, "ZN-coin"))
    ds.append(("CALB", split_recorder.CALB_train_files, split_recorder.CALB_val_files, split_recorder.CALB_test_files, "CALB"))
    ds.append(("CALB42", split_recorder.CALB_42_train_files, split_recorder.CALB_42_val_files, split_recorder.CALB_42_test_files, "CALB"))
    ds.append(("CALB2024", split_recorder.CALB_2024_train_files, split_recorder.CALB_2024_val_files, split_recorder.CALB_2024_test_files, "CALB"))
    ds.append(("NA-ion", split_recorder.NAion_2021_train_files, split_recorder.NAion_2021_val_files, split_recorder.NAion_2021_test_files, "NA-ion"))
    ds.append(("NAion42", split_recorder.NAion_42_train_files, split_recorder.NAion_42_val_files, split_recorder.NAion_42_test_files, "NA-ion"))
    ds.append(("NAion2024", split_recorder.NAion_2024_train_files, split_recorder.NAion_2024_val_files, split_recorder.NAion_2024_test_files, "NA-ion"))

    # --- Combination ---
    ds.append(("MIX_large", split_recorder.MIX_large_train_files, split_recorder.MIX_large_val_files, split_recorder.MIX_large_test_files, None))

    return ds


def main():
    datasets = build_datasets()
    print(f"Total datasets: {len(datasets)}")
    assert len(datasets) == 25, f"Expected 25 datasets, got {len(datasets)}"

    rows = []
    grand_total = {"train_cells": 0, "val_cells": 0, "test_cells": 0,
                   "train_cycles": 0, "val_cycles": 0, "test_cycles": 0}

    for name, train, val, test, dir_ov in datasets:
        tr_cells, tr_cycles, tr_miss = count_split(train, dir_ov)
        va_cells, va_cycles, va_miss = count_split(val, dir_ov)
        te_cells, te_cycles, te_miss = count_split(test, dir_ov)

        rows.append((name, tr_cells, va_cells, te_cells,
                     tr_cells + va_cells + te_cells,
                     tr_cycles, va_cycles, te_cycles,
                     tr_cycles + va_cycles + te_cycles,
                     tr_miss + va_miss + te_miss))

        grand_total["train_cells"] += tr_cells
        grand_total["val_cells"] += va_cells
        grand_total["test_cells"] += te_cells
        grand_total["train_cycles"] += tr_cycles
        grand_total["val_cycles"] += va_cycles
        grand_total["test_cycles"] += te_cycles

    # Sort: MIX_large at the end, others alphabetically
    rows.sort(key=lambda r: (r[0] == "MIX_large", r[0]))

    # ---------- Print to stdout ----------
    sep = " | "
    header = ("Dataset", "Train Cells", "Val Cells", "Test Cells", "Total Cells",
              "Train Cycles", "Val Cycles", "Test Cycles", "Total Cycles", "Missing")

    print(sep.join(f"{h:>14}" if i > 0 else f"{h:<14}" for i, h in enumerate(header)))
    print("-" * 160)

    for r in rows:
        name, tr_c, va_c, te_c, tot_c, tr_y, va_y, te_y, tot_y, miss = r
        flags = " ⚠" if miss > 0 else ""
        print(f"{name + flags:<14}" + sep + sep.join(
            f"{v:>14,}" for v in (tr_c, va_c, te_c, tot_c, tr_y, va_y, te_y, tot_y)
        ))

    # Grand total
    gt = grand_total
    print("-" * 160)
    total_cells = gt["train_cells"] + gt["val_cells"] + gt["test_cells"]
    total_cycles = gt["train_cycles"] + gt["val_cycles"] + gt["test_cycles"]
    print(f"{'GRAND TOTAL':<14}" + sep + sep.join(
        f"{v:>14,}" for v in (gt["train_cells"], gt["val_cells"], gt["test_cells"],
                               total_cells, gt["train_cycles"], gt["val_cycles"],
                               gt["test_cycles"], total_cycles)
    ))

    # ---------- Write Markdown ----------
    md_path = os.path.join(os.path.dirname(__file__), "dataset_sample_counts.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BatteryLife 特征提取数据集样本统计\n\n")
        f.write(f"统计日期：自动生成  \n")
        f.write(f"数据来源：`extracted_features/` 目录下的 CSV 文件  \n")
        f.write(f"训练/验证/测试集划分来源：`data_provider/data_split_recorder.py`  \n")
        f.write(f"共计 **{len(datasets)}** 个可运行的数据集。\n\n")
        f.write("> 注：\"样本\"指一个循环（cycle），即每个 CSV 文件中的一行数据。\n")
        f.write("> \"Cells\" 指电池单体数量。\n\n")

        f.write("## 总览\n\n")
        f.write("| Dataset | Train Cells | Val Cells | Test Cells | Total Cells | Train Cycles | Val Cycles | Test Cycles | Total Cycles |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for r in rows:
            name, tr_c, va_c, te_c, tot_c, tr_y, va_y, te_y, tot_y, miss = r
            flags = " ⚠" if miss > 0 else ""
            f.write(f"| {name}{flags} | {tr_c:,} | {va_c:,} | {te_c:,} | {tot_c:,} | {tr_y:,} | {va_y:,} | {te_y:,} | {tot_y:,} |\n")

        f.write(f"| **GRAND TOTAL** | **{gt['train_cells']:,}** | **{gt['val_cells']:,}** | **{gt['test_cells']:,}** | **{total_cells:,}** | **{gt['train_cycles']:,}** | **{gt['val_cycles']:,}** | **{gt['test_cycles']:,}** | **{total_cycles:,}** |\n")

        f.write("\n## 数据集说明\n\n")
        f.write("### MIX_large 构成数据集（15 个）\n\n")
        f.write("这 15 个数据集共同构成 MIX_large。其中 Tongji 在 data loader 中被细分为 Tongji1 / Tongji2 / Tongji3（按电池编号前缀划分），所以 MIX_large 实际覆盖 13 个物理数据集但映射为 15 个训练数据集。\n\n")

        f.write("### 独立数据集（9 个）\n\n")
        f.write("以下数据集不属于 MIX_large，各自独立训练和评估：\n\n")
        f.write("- **ZN-coin / ZN42 / ZN2024**：ZN-coin 数据集的三种 seen/unseen 协议划分变体\n")
        f.write("- **CALB / CALB42 / CALB2024**：CALB 数据集的三种 seen/unseen 协议划分变体\n")
        f.write("- **NA-ion / NAion42 / NAion2024**：NA-ion 数据集的三种 seen/unseen 协议划分变体\n\n")

        f.write("### 组合数据集（1 个）\n\n")
        f.write("- **MIX_large**：由 13 个物理数据集合并而成（HUST + MATR + SNL + RWTH + MICH + MICH_EXP + UL_PUR + CALCE + HNEI + Tongji + Stanford + ISU_ILCC + XJTU）\n\n")

        f.write("### 未纳入训练的数据集\n\n")
        f.write("- **SDU** 和 **Stanford_2**：这两个数据集存在于 `extracted_features/` 中，但 `data_split_recorder.py` 中未定义其训练/验证/测试集划分，因此未包含在上述 25 个数据集中。\n")

        if any(r[9] > 0 for r in rows):
            f.write("\n## 注意事项\n\n")
            f.write("带 ⚠ 标记的数据集存在部分 cell 的 CSV 文件缺失（可能特征提取失败或未完成）。\n")

    print(f"\nMarkdown saved to: {md_path}")


if __name__ == "__main__":
    main()
