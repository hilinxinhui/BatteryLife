# Data Download Tutorial

There is a tutorial for teaching users how to download the `raw data` and `processed data` of the `BatteryLife` project from huggingface website.

## How to download the processed data from huggingface

**Data downloading tutorial for processed data:**

**Step 1:** Log in to the ([Hugging Face – The AI community building the future.](https://huggingface.co/))

**Step 2:** Get the access token for yourself. 

- Firstly, click on your avatar then click the `Access token` button.
- Secondly, click the `Create new token` button.
- Thirdly, enter `Processed` into `Token name` textbox.
- Fourthly, select `Read access to contents of all repos under your personal namespace` and `Read access to contents of all public gated repos you can access` options under the `Repositories`.
- Finally, click the `Create token` button at the end.

**Step 3:** Save the access token you just created to another place.

**Step 4:** Enter the command below in the terminal of the `BatteryLife` project.

- ```python
  pip install --upgrade huggingface_hub
  ```

**Step 5:** Enter the command below in the terminal of the `BatteryLife` project to download the processed data.

- ```
  huggingface-cli download --repo-type dataset --token your_access_token --resume-download Battery-Life/BatteryLife_Processed --cache-dir /path/to/your/folder/
  ```

```Shell
hf download --repo-type dataset --token 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 Battery-Life/BatteryLife_Processed --cache-dir ./ds
```

then the processed data will be downloaded into the `/path/to/your/folder/BatteryLife/test/datasets--Battery-Life--BatteryLife_Processed/snapshots` file.

## (ZN-ion/NA-ion/CALB raw data downloading) How to download the raw data from huggingface

The raw datasets on the huggingface website are ZN-ion, NA-ion, and CALB datasets. The raw data of the Li-ion dataset can be downloaded from their websites, please refer to the `Data availability` section of [README](../README.md).

**Data downloading tutorial for raw data:**

- Simply follow the processed data downloading tutorial but use the different commands in `Step 5`. For raw data downloading, please enter the command below in the terminal:

- ```
  huggingface-cli download --repo-type dataset --token your_access_token --resume-download Battery-Life/BatteryLife_Raw --cache-dir /path/to/your/folder/
  ```

then the raw data will be downloaded into the `/path/to/your/folder/BatteryLife/test/datasets--Battery-Life--BatteryLife_Raw/snapshots` file.
