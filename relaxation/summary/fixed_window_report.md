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
   XJTU   300s  relax_kur    SOH 18146 -0.127186 -0.572328     0.127186      0.572328
   XJTU   300s  relax_ske    SOH 18146 -0.258167 -0.543845     0.258167      0.543845
   XJTU   300s relax_mean    SOH 18146  0.227774  0.506519     0.227774      0.506519
   XJTU   300s  relax_min    SOH 18146  0.188531  0.488583     0.188531      0.488583
   XJTU   300s  relax_var    SOH 18146 -0.088010 -0.403608     0.088010      0.403608
   XJTU   300s  relax_max    SOH 18146  0.251894  0.389826     0.251894      0.389826
