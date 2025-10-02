# Model training tutorial

Note: For the Li-ion dataset, we obtain the best hyperparameters using 2021 as the random seed. After that, we evaluate the model performance using the same hyperparameters using 2024 and 42 as the random seeds. The reported results are the mean±standard deviation. As for other datasets, we obtain the hyperparameters that perform the best on the validation set regarding the average results from three runs with three random seeds (42, 2021, and 2024), and report the corresponding model performance on the testing sets. In this vein, we introduce the specific steps for each dataset using `CPTransformer `as an example.

## Li-ion

First, you should run the following script:

```shell
sh ./train_eval_scripts/CPTransformer.sh # set "seed" as 2021 in the script. set "dataset" as MIX_large.
```

You can tune the hyperparameters like `d_ff` and `d_model `until you obtain a satisfactory set of hyperparameters that perform the best on the validation set. After that, you use the same hyperparameters and evaluate the model performance using 42 and 2024 as the random seed.

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 42 in the script. Train the model using the best hyperparameters. set "dataset" as MIX_large.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 2024 in the script. Train the model using the best hyperparameters. set "dataset" as MIX_large.
```

You will see the model performance at the end of model training. After that you can report the mean±standard deviation of MAPE and 15%-Acc.



## Zn-ion

First, you should run the following script:

```shell
sh ./train_eval_scripts/CPTransformer.sh # set "seed" as 2021 in the script. set "dataset" as ZN-coin.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 42 in the script. Train the model using the best hyperparameters. set "dataset" as ZN-coin42.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 2024 in the script. Train the model using the best hyperparameters. set "dataset" as ZN-coin2024.
```

You can tune the hyperparameters like `d_ff` and `d_model` until you obtain a satisfactory set of hyperparameters that perform the best on the validation sets.

You will see the model performance at the end of model training. After that you can report the mean±standard deviation of MAPE and 15%-Acc.



## Na-ion

First, you should run the following script:

```shell
sh ./train_eval_scripts/CPTransformer.sh # set "seed" as 2021 in the script. set "dataset" as NAion.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 42 in the script. Train the model using the best hyperparameters. set "dataset" as NAion42.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 2024 in the script. Train the model using the best hyperparameters. set "dataset" as NAion2024.
```

You can tune the hyperparameters like `d_ff` and `d_model` until you obtain a satisfactory set of hyperparameters that perform the best on the validation sets.

You will see the model performance at the end of model training. After that you can report the mean±standard deviation of MAPE and 15%-Acc.



## CALB

First, you should run the following script:

```shell
sh ./train_eval_scripts/CPTransformer.sh # set "seed" as 2021 in the script. set "dataset" as CALB.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 42 in the script. Train the model using the best hyperparameters. set "dataset" as CALB42.
```

```shell
sh ./train_eval_scripts/CPTransformer.sh # set seed as 2024 in the script. Train the model using the best hyperparameters. set "dataset" as CALB2024.
```

You can tune the hyperparameters like `d_ff` and `d_model` until you obtain a satisfactory set of hyperparameters that perform the best on the validation sets.

You will see the model performance at the end of model training. After that you can report the mean±standard deviation of MAPE and 15%-Acc.



## Note for Dummy model usage

Because the Dummy model simply uses the average life of the training data as the prediction, the usage of the Dummy model is different from other models. An example is shown below:

```shell
python ./models/Dummy.py [dataset] [random_seed]
# example for CALB dataset with random seed 2021: python ./models/Dummy.py CALB 2021
```

