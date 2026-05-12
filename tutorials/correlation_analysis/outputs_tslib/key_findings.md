# Key Findings

Data source:
- Features: `/Users/lxh/Desktop/codes/SOHbenchmark_TSLib/dataset/BatteryLife`
- Benchmark results: `/Users/lxh/Desktop/codes/SOHbenchmark_TSLib/results`
- `MIX_large` is excluded from model-performance mapping.

Coverage:
- Feature profiles: 1,382 batteries/cells
- Benchmark metric rows: 3,080
- Joined metric/profile rows: 3,080, with 0 missing feature profiles
- Per-sample prediction relationship rows: 49,280
- Covered experiments: 240, from 24 tasks x 2 targets x 5 models

Strong feature-truth links:
- SOH: strongest dataset-level links are mainly RWTH, NA-ion, and HNEI.
- SOH top examples: RWTH `CV_charge_time` Spearman -0.9983; RWTH `current_slope` -0.9977; NA-ion `CC_Q` 0.9966.
- RUL: strongest dataset-level links are mainly HNEI, XJTU, and RWTH.
- RUL top examples: HNEI `voltage_slope` -0.9801; HNEI `CC_charge_time` 0.9783; HNEI `CC_Q` 0.9777; XJTU `CC_Q` 0.9659.

Feature-output/error links:
- Across per-battery prediction files, RUL errors are most strongly associated on average with `CC_Q`, `CC_charge_time`, and `voltage_slope`.
- SOH errors are most strongly associated on average with `CC_charge_time`, `CC_Q`, and `voltage_slope`.
- This suggests the CC-stage capacity/time/slope features are not only label-informative, but also identify regions where models become more or less reliable.

Feature profile to performance mapping:
- For RUL, error metrics are dominated by lifespan scale: `cycle_max`, `RUL_max`, `n_cycles`, and `RUL_range` have Spearman around 0.66 with RMSE/MAE.
- For SOH, `CC_Q__mean` is the strongest global descriptor for MAE/MAPE/RMSE, with negative Spearman around -0.50 to -0.56.
- For SOH correlation quality (`r2`, `pearson_r`), `current_entropy__spearman_SOH` is one of the strongest descriptors.

Primary output tables:
- `feature_truth_correlations_by_dataset.csv`
- `battery_feature_profiles.csv`
- `sohbenchmark_battery_metrics_long.csv`
- `battery_metrics_with_feature_profiles.csv`
- `feature_profile_performance_mapping.csv`
- `feature_prediction_error_relationships_available_npz.csv`
