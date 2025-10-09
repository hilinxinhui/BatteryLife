# (KDD 2025) BatteryLife

This is the official repository for [BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction](https://dl.acm.org/doi/10.1145/3711896.3737372). If you find this repository useful, we would appreciate citations to our paper and stars to this repository.

## Highlights

- **The largest battery life dataset:** BatteryLife is created by integrating 16 datasets, providing 99,000 samples from 990 batteries with life labels. This is 2.5 times the size of BatteryML, which is the previous largest battery life resource.
- **The most diverse battery life dataset:** BatteryLife contains 8 battery formats, 59 chemical systems, 9 operation temperatures, and 421 charge/discharge protocols. Compared with the previous largest battery life resource (BatteryML), BatteryLife furnishes 4 times format, 11.8 times chemical system, 1.8 times operating temperature, and 2.2 times charge/discharge protocol.
- **A comprehensive benchmark for battery life prediction:** BatteryLife provides 18 benchmark methods with open-source codes in this repository. The 18 benchmark methods include popular methods for battery life prediction, popular baselines in time series analysis, and a series of baselines proposed by this work.

## Data availability

The processed datasets can be accessed via multiple ways:

1. You can download the datasets from [Huggingface](https://huggingface.co/datasets/Hongwxx/BatteryLife_processed/tree/main) [[tutorial]](./assets/Data_download.md).
2. You can download the datasets from [Zenodo](https://zenodo.org/records/15531867).

Note that brief introductions to each dataset are available under the directory of each dataset.

All the raw datasets are publicly available, interested users can download them from the following links:

- Zn-ion, Na-ion, and CALB datasets: [Zenodo link](https://zenodo.org/records/15013636) [Huggingface link](https://huggingface.co/datasets/Hongwxx/BatteryLife_Raw/tree/main) [[tutorial]](./assets/Data_download.md#how-to-download-the-raw-data-from-huggingface)
- CALCE: [link](https://calce.umd.edu/battery-data)
- MATR: [Three batches](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and [Batch 9](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94)
- HUST: [link](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
- RWTH: [link](https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip)
- ISU\_ILCC: [link](https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234)
- XJTU: [link](https://zenodo.org/records/10963339)
- Tongji: [link](https://zenodo.org/records/6405084)
- Stanford: [link](https://data.matr.io/8/)
- HNEI, SNL, MICH, MICH_EXP and UL_PUR datasets: [BatteryArchive](https://www.batteryarchive.org/index.html).

## Benchmark results of Battery Life Prediction (BLP) task

The benchmark result for battery life prediction. The comparison methods are split into five types, including

1. Dummy, a baseline that uses the mean of training labels as the prediction.
2. MLPs, a series of multilayer perceptron models including DLinear, MLP, and CPMLP.
3. Transformers, a series of transformer models including PatchTST, Autoformer, iTransformer, Transformer, and CPTransformer.
4. CNNs, a series of convolutional neural network models including CNN and MICN.
5. RNNs, a series of recurrent neural network models including CPGRU, CPBiGRU, CPLSTM, CPBiLSTM, GRU, BiGRU, LSTM, and BiLSTM.

|   Datasets    |   Li-ion    |   Li-ion    |   Zn-ion    |   Zn-ion    |   Na-ion    |   Na-ion    |    CALB     |    CALB     |
| :-----------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|  **Metrics**  |  **MAPE**   | **15%-Acc** |  **MAPE**   | **15%-Acc** |  **MAPE**   | **15%-Acc** |  **MAPE**   | **15%-Acc** |
|     Dummy     | 0.831±0.000 | 0.296±0.000 | 1.297±0.214 | 0.083±0.047 | 0.404±0.029 | 0.067±0.094 | 1.811±0.550 | 0.267±0.094 |
|    DLinear    | 0.586±0.028 | 0.275±0.017 | 0.814±0.026 | 0.124±0.020 | 0.319±0.031 | 0.329±0.042 | 0.164±0.049 | 0.601±0.114 |
|      MLP      | 0.233±0.010 | 0.503±0.013 | 0.805±0.103 | 0.079±0.055 | 0.281±0.067 | 0.364±0.098 | 0.149±0.014 | 0.641±0.115 |
|     CPMLP     | 0.179±0.003 | 0.620±0.004 | 0.558±0.034 | 0.297±0.084 | 0.274±0.026 | 0.337±0.038 | 0.140±0.009 | 0.704±0.053 |
|   PatchTST    | 0.288±0.042 | 0.430±0.053 | 0.716±0.024 | 0.133±0.001 | 0.396±0.094 | 0.258±0.070 | 0.347±0.045 | 0.511±0.139 |
|  Autoformer   | 0.437±0.093 | 0.287±0.067 | 0.987±0.243 | 0.106±0.039 | 0.372±0.047 | 0.177±0.128 | 0.761±0.061 | 0.329±0.121 |
| iTransformer  | 0.209±0.015 | 0.516±0.028 | 0.690±0.110 | 0.188±0.037 | 0.321±0.087 | 0.249±0.178 | 0.164±0.020 | 0.649±0.044 |
|  Transformer  |      -      |      -      |      -      |      -      |      -      |      -      |      -      |      -      |
| CPTransformer | 0.184±0.003 | 0.573±0.016 | 0.515±0.067 | 0.202±0.084 | 0.255±0.036 | 0.406±0.084 | 0.149±0.005 | 0.672±0.107 |
|      CNN      | 0.337±0.068 | 0.371±0.050 | 0.928±0.093 | 0.115±0.029 | 0.307±0.047 | 0.273±0.027 | 0.278±0.011 | 0.582±0.032 |
|     MICN      | 0.249±0.004 | 0.494±0.019 | 0.579±0.101 | 0.227±0.127 | 0.305±0.040 | 0.335±0.065 | 0.233±0.050 | 0.471±0.257 |
|     CPGRU     | 0.189±0.008 | 0.585±0.013 | 0.616±0.049 | 0.289±0.076 | 0.298±0.063 | 0.203±0.160 | 0.141±0.012 | 0.681±0.178 |
|    CPBiGRU    | 0.190±0.001 | 0.566±0.034 | 0.774±0.202 | 0.193±0.156 | 0.282±0.055 | 0.395±0.008 | 0.160±0.015 | 0.686±0.063 |
|    CPLSTM     | 0.196±0.006 | 0.585±0.020 | 0.932±0.227 | 0.085±0.028 | 0.272±0.051 | 0.386±0.009 | 0.156±0.073 | 0.613±0.153 |
|   CPBiLSTM    | 0.191±0.007 | 0.421±0.255 | 0.645±0.049 | 0.150±0.104 | 0.299±0.043 | 0.399±0.001 | 0.173±0.075 | 0.663±0.247 |
|   GRU&BiGRU   |     NA      |     NA      |     NA      |     NA      |     NA      |     NA      |     NA      |     NA      |
|  LSTM&BiLSTM  |     NA      |     NA      |     NA      |     NA      |     NA      |     NA      |     NA      |     NA      |

## Quick start

### Install

```
pip install -r requirements.txt
# You should also install BatteryML (https://github.com/microsoft/BatteryML)
```

<!-- ```Shell
conda create -n BatteryLife python=3.10 -y
conda activate BatteryLife
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # torch2.8.0 with cuda12.8
pip install transformers datasets accelerate deepspeed peft denseweight reformer_pytorch evaluate huggingface_hub
pip install numpy pandas matplotlib seaborn scikit_learn scipy sympy
pip install Requests joblib tqdm
pip install jupyter ipython notebook
pip install wandb

# pip install BatteryML
# to install BatteryML, please refer to https://github.com/microsoft/BatteryML
git clone https://github.com/microsoft/BatteryML.git
cd path/to/BatteryML
pip install -r requirements.txt
pip install .
``` -->

```Shell
conda create -n BatteryLife_new python=3.10 -y
conda activate BatteryLife_new

git clone https://github.com/microsoft/BatteryML.git
cd path/to/BatteryML
pip install -r requirements.txt
pip install .

cd path/to/BatteryLife
pip install -r requirements.txt
```

### Preprocessing [[tutorial](./assets/Preprocess.md)]

After downloading all raw datasets provided in "Data availability" section, you can run the following script to obtain the processed datasets:

```
python preprocess_scripts.py
```

If you download the processed datasets, you can skip this step.

### Train the model [[tutorial](./assets/Model_training.md)]

Before you start training, please move all **processed datasets (such as, HUST, MATR, et al.)** folders and **Life labels** folder (downloaded from Hugginface or Zenodo websites) into `./dataset` folder under the root folder.

After that, just feel free to run any benchmark method. For example:

```sh
sh ./train_eval_scripts/CPTransformer.sh
```

### Evaluate the model

If you want to evaluate a model in detail. We have provided the evaluation script. You can use it as follows:

```sh
sh ./train_eval_scripts/evaluate.sh
```

### Fine-tuning [[tutorial](./assets/Transfer_learning.md#Fine-tuning)]

If you want to fine-tune the pretrained model to another dataset. We have provided the fine-tuning script and the [tutorial](./assets/Transfer_learning.md#Fine-tuning). You can use it as follows:

```shell
sh ./train_eval_scripts/finetune_script.sh
```

### Domain adaptation [[tutorial](./assets/Transfer_learning.md#domain-adaptation)]

If you want to do the domain adaptation to another dataset. We have provided the domain adaptation script and the [tutorial](./assets/Transfer_learning.md#domain-adaptation). You can use it as follows:

```shell
sh ./train_eval_scripts/domain_adaptation_script.sh
```

## Documention

- The main information is described in our [BatteryLife paper](https://arxiv.org/abs/2502.18807). 
- The data structure of the standardized data is described in [Data_structure_description.md](./assets/Data_structure_description.md). 
- Further details of data statistics are available at [Further_details_of_data_statistics.md](./assets/Further_details_of_data_statistics.md).
- Further details of processed charge and discharge capacity data are available at [Further_details_of_processed_charge_and_discharge_capacity_data.md](./assets/Further_details_of_processed_charge_and_discharge_capacity_data.md).

## Welcome contributions

Advancing AI4Battery requires standardized datasets. However, the available battery life datasets are typically stored in different places and in different formats. We have put great efforts into integrating 13 previously available datasets and 3 of our datasets. BatteryLife serves as a platform to share all kinds of standardized datasets. We warmly welcome contributions from the community to further enhance this collection by providing datasets standardized according to the BatteryLife standards.

If you are interested in contributing, please either submit a pull request or contact us via email at rtan474@connect.hkust-gz.edu.cn and whong719@connect.hkust-gz.edu.cn. To integrate your data into the BatteryLife repositories, please provide:

- Raw datasets
- Processed datasets
- Preprocessing scripts (for reproducibility)
- A list of contributors (for acknowledgment in the repo)
- Papers related to the data generation (we will prompt users to cite these in the repository's Citation section).


## Citation

If you use the benchmark, processed datasets, or the raw datasets produced by this work, you should cite the BatteryLife paper:

```
@inproceedings{10.1145/3711896.3737372,
author = {Tan, Ruifeng and Hong, Weixiang and Tang, Jiayue and Lu, Xibin and Ma, Ruijun and Zheng, Xiang and Li, Jia and Huang, Jiaqiang and Zhang, Tong-Yi},
title = {BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3737372},
doi = {10.1145/3711896.3737372},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {5789–5800},
numpages = {12},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```

- Additionally, please cite the original papers that conducted experiments. Please cite [BatteryArchive](https://www.batteryarchive.org/index.html) as the data source for the HNEI, SNL, MICH, MICH_EXP, and UL_PUR datasets.
- Please cite [BatteryML](https://iclr.cc/virtual/2024/poster/17628) if you use the processed CALCE, MATR, HUST, HNEI, RWTH, SNL, and UL_PUR datasets. Our preprocessing for these 7 datasets relies heavily on BatteryML's preprocessing scripts.


## Acknowledgement

This repo is constructed based on the following repos:

- https://github.com/thuml/Time-Series-Library
- https://github.com/microsoft/BatteryML

## All thanks to our contributors

<a href="https://github.com/Ruifeng-Tan/BatteryLife/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Ruifeng-Tan/BatteryLife" />
</a>
