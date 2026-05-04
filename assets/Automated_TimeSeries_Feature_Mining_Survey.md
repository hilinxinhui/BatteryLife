# Automated Time-Series Feature Mining Survey

整理日期: 2026-04-27

## Scope

这份文档整理了我们前面讨论过、建议纳入电池退化建模评测框架的七类自动化特征挖掘方法。这里故意不把范围限制在“已经用于电池”的方法，而是同时纳入了具有明显迁移潜力的通用时序 pipeline。

文档重点保留三类信息:

- 方法类别与核心思路
- 代表论文 URL
- 可直接上手的软件 URL, 优先放官方 GitHub 或 PyPI

## 1. Large-Scale Generic Feature Libraries

核心思路: 先从单变量或多变量时序中批量提取大量通用统计、频域、复杂性、非线性动力学特征, 再做筛选与建模。适合做自动化 baseline, 也适合和后续相关性筛选、树模型重要性排序串联。

代表论文:

- `hctsa`: https://doi.org/10.1016/j.cels.2017.10.001
- `catch22`: https://doi.org/10.1007/s10618-019-00647-x

软件:

- `hctsa` GitHub: https://github.com/benfulcher/hctsa
- `catch22` GitHub: https://github.com/DynamicsAndNeuralSystems/catch22
- `pycatch22` PyPI: https://pypi.org/project/pycatch22/
- `tsfresh` GitHub: https://github.com/blue-yonder/tsfresh
- `tsfresh` PyPI: https://pypi.org/project/tsfresh/

对电池退化建模的意义:

- 适合作为“低门槛、高覆盖”的首批候选特征来源
- 可以和 IC/DV/DTV 等电池专用特征并联
- `catch22` 特别适合做轻量化强基线

## 2. Local Subsequences, Dictionary Methods, and Motif Discovery

核心思路: 直接从局部子序列、重复模式、异常片段和离散化单词中构造特征, 适合发现充放电曲线、增量容量曲线中的局部形态差异。

代表论文:

- `Shapelet`: https://doi.org/10.1007/s10618-010-0179-5
- `Matrix Profile`: https://doi.org/10.1007/s10618-017-0519-9
- `WEASEL 2.0`: https://doi.org/10.1007/s10994-023-06395-w

软件:

- `aeon` GitHub: https://github.com/aeon-toolkit/aeon
- `aeon` PyPI: https://pypi.org/project/aeon/
- `STUMPY` GitHub: https://github.com/TDAmeritrade/stumpy
- `STUMPY` PyPI: https://pypi.org/project/stumpy/
- `pyts` GitHub: https://github.com/johannfaouzi/pyts
- `pyts` PyPI: https://pypi.org/project/pyts/

对电池退化建模的意义:

- 比纯统计特征更容易抓住局部退化形态
- 很适合与 SAX/SFA、频繁模式挖掘串联
- `Matrix Profile` 还能给后续变点检测和异常分析提供候选片段

## 3. Random Convolution Feature Transforms

核心思路: 使用大量随机卷积核或近似确定性的卷积核对时序做变换, 再把变换后的特征交给线性模型或浅层模型。优点是自动化程度高、速度快、通常有很强的经验表现。

代表论文:

- `ROCKET`: https://doi.org/10.1007/s10618-020-00701-z
- `MiniROCKET`: https://doi.org/10.1145/3447548.3467231

软件:

- `sktime` GitHub: https://github.com/sktime/sktime
- `sktime` PyPI: https://pypi.org/project/sktime/
- `aeon` GitHub: https://github.com/aeon-toolkit/aeon
- `aeon` PyPI: https://pypi.org/project/aeon/

对电池退化建模的意义:

- 很适合做“几乎不依赖人工设计”的自动化特征基线
- 可以把原始电压、电流、容量、温度序列统一映射成下游回归器可用的表征
- 原始论文以分类为主, 但其变换输出完全可以接 SOH 或 RUL 回归模型

## 4. Interval Features and Changepoint Pipelines

核心思路: 不再只看整条序列的全局统计量, 而是自动寻找高判别区间、阶段边界和结构突变点, 再从每个区间或阶段提取聚合特征。

代表论文:

- `r-STSF`: https://doi.org/10.1007/s10618-023-00978-w
- `ruptures`: https://joss.theoj.org/papers/10.21105/joss.01026
- `Interval Feature Transformation`: https://doi.org/10.3390/app10165428

软件:

- `rSTSF` PyPI: https://pypi.org/project/rSTSF/
- `ruptures` GitHub: https://github.com/deepcharles/ruptures
- `ruptures` PyPI: https://pypi.org/project/ruptures/
- `aeon` GitHub: https://github.com/aeon-toolkit/aeon

对电池退化建模的意义:

- 电池退化常常先体现在特定 SOC、电压平台或某类循环阶段
- 区间特征和变点特征可以自然刻画“阶段性退化”
- 适合和电池循环级 metadata 一起组合成多层级特征

## 5. Relational and Programmatic Automated Feature Synthesis

核心思路: 把特征工程视为多表聚合、变换组合和程序搜索问题。特别适合“电池-循环-工步-环境-统计量”这种层级化数据结构。

代表论文:

- `Deep Feature Synthesis`: https://doi.org/10.1109/DSAA.2015.7344858
- `OneBM`: https://arxiv.org/abs/1706.00327
- `ExploreKit`: https://doi.org/10.1109/ICDM.2016.0123
- `autofeat`: https://doi.org/10.1007/978-3-030-43823-4_10

软件:

- `Featuretools` GitHub: https://github.com/alteryx/featuretools
- `Featuretools` PyPI: https://pypi.org/project/featuretools/
- `autofeat` GitHub: https://github.com/cod3licious/autofeat
- `autofeat` PyPI: https://pypi.org/project/autofeat/
- `getML / FastProp` GitHub: https://github.com/getml/getml-community
- `getML` Docs: https://docs.getml.com/

对电池退化建模的意义:

- 如果后续要把原始曲线、循环级统计量、工况标签、老化条件一起建模, 这一类方法非常关键
- 它们不只是在“提特征”, 也在自动组织 feature hierarchy
- 对跨数据集统一 schema 很有帮助

## 6. Self-Supervised Representation Learning and Time-Series Foundation Models

核心思路: 不显式手工列举大量特征, 而是先学一个通用时序表示, 再把 embedding 作为下游 SOH、RUL、容量轨迹预测的输入。适合小样本迁移、跨平台泛化和预训练复用。

代表论文:

- `TS2Vec`: https://arxiv.org/abs/2106.10466
- `MOMENT`: https://arxiv.org/abs/2402.03885
- `Time Series Foundation Model Survey`: https://arxiv.org/abs/2504.04011

软件:

- `TS2Vec` GitHub: https://github.com/zhihanyue/ts2vec
- `MOMENT` GitHub: https://github.com/moment-timeseries-foundation-model/moment
- `momentfm` PyPI: https://pypi.org/project/momentfm/

对电池退化建模的意义:

- 如果希望减少手工定义健康因子, 这类方法是最直接的替代路线
- 适合做跨实验室、跨化学体系、跨工况迁移
- 可把 embedding 视为“自动发现的退化特征”

## 7. LLM and Agentic End-to-End Automated Feature Engineering

核心思路: 让 LLM 不只是生成文本解释, 而是直接参与特征工程闭环, 包括理解数据字典、提出特征变换、编排工具链、评估新特征、汇总结果, 形成 agentic pipeline。

代表论文:

- `LLM-FE`: https://arxiv.org/abs/2503.14434
- `DCATS / Empowering Time Series Forecasting with LLM-Agents`: https://arxiv.org/abs/2508.04231
- `TimeSeriesGym`: https://arxiv.org/abs/2505.13291

软件:

- `LLM-FE` GitHub: https://github.com/nikhilsab/LLMFE
- `TimeSeriesGym` GitHub: https://github.com/moment-timeseries-foundation-model/TimeSeriesGym

对电池退化建模的意义:

- 短期最现实的方案不是让 LLM 直接替代所有时序算法, 而是让它做外层 orchestration
- 可让 LLM 自动调用 `tsfresh`、`catch22`、`ROCKET`、`Matrix Profile`、`ruptures`、`Featuretools`、`IC/DV` 提取器并形成闭环筛选
- 这类方法很适合作为“全流程自动化退化特征挖掘”的重点探索方向
- 截至 2026-04-27, 我没有确认到 `DCATS` 的独立公开代码仓库, 更适合先按论文方法论纳入调研

## Suggested Evaluation Framing

如果后续要把这七类方法纳入统一实验框架, 建议把评测切成三层:

- `Feature generation`: 原始曲线、循环级统计量、工况 metadata 进入何种自动化 pipeline
- `Feature selection / ranking`: 相关系数、稳定性筛选、树模型重要性、Permutation Importance、SHAP
- `Downstream prediction`: 统一接到 SOH、RUL、寿命分类、容量轨迹预测等任务

一个很重要的注意点:

- 有些方法原始论文主要做分类, 但并不妨碍把它们视为“通用自动特征提取器”后接回归器
- 有些方法更像表示学习而非显式特征工程, 但在实验设计上完全可以把它们的 embedding 视为 feature set
- LLM/agent 类方法的软件生态变化很快, 正式实验前建议再核对一次仓库可用性与许可证
