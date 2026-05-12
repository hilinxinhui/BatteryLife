# Relaxation Feature Report

## Coverage

- Datasets processed: 18
- Cells processed: 1382
- Cycles audited: 2179522
- Paper-compatible full-charge relaxation cycles: 1645099

Datasets with the highest compatible-cycle ratios:

   dataset  cells  cycles  cycles_with_any_rest  cycles_with_paper_compatible_relaxation  paper_compatible_cycle_ratio  median_selected_duration_s  median_selected_points  feature_rows    available_windows
      XJTU     23    6938                  6938                                     6938                      1.000000                  300.000000                   293.0         11570            300s,full
    UL_PUR     10    2245                  2245                                     2245                      1.000000                 1205.060000                    12.0          2245                 full
       SDU     86   49146                 49146                                    49146                      1.000000                  950.655950                    97.0         49146                 full
     CALCE     13   14298                 14298                                    14297                      0.999930                   65.030208                     4.0         14297                 full
  ISU_ILCC    240 1194846               1194479                                  1194414                      0.999638                  592.000000                   120.0       2860353 1800s,300s,600s,full
  Stanford     41   39329                 39329                                    39305                      0.999390                  692.182222                   152.0         39309       300s,600s,full
Stanford_2    181  188621                188621                                   188489                      0.999300                  716.636637                   157.0        188497       300s,600s,full
      MICH     40   19895                 19887                                    19881                      0.999296                 1624.090000                   145.0         65501 1800s,300s,600s,full
      HNEI     14   15164                 14757                                    14754                      0.972962                  158.656000                     6.0         14754                 full
    Tongji    130   59038                 59038                                    54364                      0.920831                 1680.101001                    15.0         54364                 full

## Strongest SOH Correlations

Filtered to correlation rows with `n >= 100`.

 dataset window    feature target      n   pearson  spearman  abs_pearson  abs_spearman
    XJTU   300s relax_mean    SOH   4632  0.859827  0.816162     0.859827      0.816162
    XJTU   300s  relax_min    SOH   4632  0.849494  0.809586     0.849494      0.809586
    XJTU   300s  relax_var    SOH   4632 -0.870542 -0.762141     0.870542      0.762141
    XJTU   300s  relax_kur    SOH   4632 -0.647519 -0.740703     0.647519      0.740703
    XJTU   300s  relax_ske    SOH   4632 -0.394585 -0.519984     0.394585      0.519984
    XJTU   300s  relax_max    SOH   4632  0.472491  0.423872     0.472491      0.423872
    CALB   300s  relax_min    SOH   1379  0.726779  0.325865     0.726779      0.325865
    CALB   300s relax_mean    SOH   1379  0.721565  0.298710     0.721565      0.298710
    CALB   300s  relax_max    SOH   1379  0.721128  0.287919     0.721128      0.287919
    CALB   300s  relax_var    SOH   1379  0.601459  0.176998     0.601459      0.176998
ISU_ILCC  1800s  relax_min    SOH  12163  0.139470  0.165804     0.139470      0.165804
ISU_ILCC  1800s relax_mean    SOH  12163  0.138257  0.162650     0.138257      0.162650
ISU_ILCC  1800s  relax_max    SOH  12163  0.113195  0.147424     0.113195      0.147424
    CALB   300s  relax_ske    SOH   1379  0.093093  0.099448     0.093093      0.099448
ISU_ILCC  1800s  relax_var    SOH  12163 -0.084985 -0.087481     0.084985      0.087481
ISU_ILCC   600s  relax_min    SOH 582900  0.069986  0.085054     0.069986      0.085054
ISU_ILCC   600s  relax_max    SOH 582900  0.070095  0.084289     0.070095      0.084289
ISU_ILCC   600s relax_mean    SOH 582900  0.069212  0.082999     0.069212      0.082999
ISU_ILCC  1800s  relax_ske    SOH  12163 -0.065162 -0.062317     0.065162      0.062317
    MICH  1800s  relax_kur    SOH   5975 -0.037270 -0.060356     0.037270      0.060356
