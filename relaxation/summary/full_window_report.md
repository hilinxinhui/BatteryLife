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

dataset window    feature target     n   pearson  spearman  abs_pearson  abs_spearman
 Tongji   full relax_mean    SOH 54364  0.916733  0.919623     0.916733      0.919623
    SDU   full relax_mean    SOH 49146  0.369608  0.854914     0.369608      0.854914
    SDU   full  relax_min    SOH 49146  0.389607  0.836116     0.389607      0.836116
   XJTU   full relax_mean    SOH  6938  0.864284  0.828606     0.864284      0.828606
   XJTU   full  relax_min    SOH  6938  0.853630  0.822269     0.853630      0.822269
 Tongji   full  relax_max    SOH 54364  0.758417  0.798585     0.758417      0.798585
   MICH   full  relax_kur    SOH 19881 -0.674660 -0.795993     0.674660      0.795993
   MICH   full  relax_ske    SOH 19881  0.682416  0.793433     0.682416      0.793433
   XJTU   full  relax_var    SOH  6938 -0.865507 -0.774628     0.865507      0.774628
    SDU   full  relax_var    SOH 49146 -0.646035 -0.752374     0.646035      0.752374
   XJTU   full  relax_kur    SOH  6938 -0.653511 -0.743272     0.653511      0.743272
  CALCE   full  relax_min    SOH 14297  0.700648  0.739414     0.700648      0.739414
  CALCE   full relax_mean    SOH 14297  0.715754  0.718209     0.715754      0.718209
  CALCE   full  relax_var    SOH 14297 -0.591598 -0.711893     0.591598      0.711893
   MICH   full  relax_min    SOH 19881  0.779618  0.683176     0.779618      0.683176
 UL_PUR   full  relax_min    SOH  2245  0.643670  0.669640     0.643670      0.669640
 UL_PUR   full  relax_var    SOH  2245 -0.640589 -0.657236     0.640589      0.657236
    SDU   full  relax_kur    SOH 49146  0.645356  0.656863     0.645356      0.656863
    SDU   full  relax_ske    SOH 49146  0.571515  0.589073     0.571515      0.589073
 UL_PUR   full  relax_ske    SOH  2245  0.455256  0.518813     0.455256      0.518813
