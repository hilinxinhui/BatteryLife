# Relaxation Feature Report

## Coverage

- Datasets processed: 1
- Cells processed: 55
- Cycles audited: 27595
- Paper-compatible full-charge relaxation cycles: 27595

Datasets with the highest compatible-cycle ratios:

dataset  cells  cycles  cycles_with_any_rest  cycles_with_paper_compatible_relaxation  paper_compatible_cycle_ratio  median_selected_duration_s  median_selected_points  feature_rows available_windows
   XJTU     55   27595                 27595                                    27595                           1.0                       300.0                   293.0         45741         300s,full

## Strongest SOH Correlations

Filtered to correlation rows with `n >= 100`.

dataset window    feature target     n   pearson  spearman  abs_pearson  abs_spearman
   XJTU   full  relax_kur    SOH 27595 -0.379040 -0.570491     0.379040      0.570491
   XJTU   full  relax_ske    SOH 27595 -0.305608 -0.540113     0.305608      0.540113
   XJTU   full relax_mean    SOH 27595  0.222741  0.506688     0.222741      0.506688
   XJTU   full  relax_min    SOH 27595  0.186491  0.488968     0.186491      0.488968
   XJTU   full  relax_var    SOH 27595 -0.082743 -0.406307     0.082743      0.406307
   XJTU   full  relax_max    SOH 27595  0.221807  0.385672     0.221807      0.385672
