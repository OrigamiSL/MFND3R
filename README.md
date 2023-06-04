# MFND<sup>3</sup>R: Multivariate Forecasting Network with Deep Data-Driven Reconstruction for Engineering Application
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.11.0](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 8.2.0](https://img.shields.io/badge/cudnn-8.2.0-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of MFND<sup>3</sup>R in the following paper: 
[MFND<sup>3</sup>R: Multivariate Forecasting Network with Deep Data-Driven Reconstruction for Engineering Application](Manuscript submitted to EAAI).

## The Entire Architecture of MFND<sup>3</sup>R
<p align="center">
<img src="./img/Architecture.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 1.</b> The entire architecture of MFND<sup>3</sup>R.
</p>


## Requirements

- Python 3.8.8
- matplotlib == 3.3.4
- numpy == 1.20.1
- pandas == 1.2.4
- scipy == 1.9.0
- scikit_learn == 0.24.1
- torch == 1.11.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

ETT, ECL and weather dataset were acquired at: [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). Solar and PEMS dataset were acquired at: [here](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). M4 dataset was acquired at: [M4](https://github.com/Mcompetitions/M4-methods). Ship dataset was in `./data/Ship/` and its raw data was in [https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/index.html](https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/index.html).

### Data Preparation
After you acquire raw data of all datasets, please separately place them in corresponding folders at `./MFND3R/data`. 

We place ETT in the folder `./ETT-data`, ECL in the folder `./electricity`  and weather in the folder `./weather` of [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) (the folder tree in the link is shown as below) into folder `./data` and rename them from `./ETT-data`,`./electricity` and `./weather` to `./ETT`, `./ECL` and`./weather` respectively. We rename the file of ECL from `electricity.csv` to `ECL.csv` and rename its last variable from `OT` to original `MT_321`.

```
|-autoformer
| |-ETT-data
| | |-ETTh1.csv
| | |-ETTh2.csv
| | |-ETTm1.csv
| | |-ETTm2.csv
| |
| |-electricity
| | |-electricity.csv
| |
| |-weather
| | |-weather.csv
```

We place Solar in the folder `./financial` and PEMS in the folder `./PEMS` of [here](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing) (the folder tree in the link is shown as below) into the folder `./data` and rename them as `./Solar` and `./PEMS` respectively. 

```
|-dataset
| |-financial
| | |-solar_AL.txt
| |
| |-PEMS
| | |-PEMS08.npz
```

As for M4 dataset, we place the folders `./Dataset` and `./Point Forecasts` of [M4](https://github.com/Mcompetitions/M4-methods) (the folder tree in the link is shown as below) into the folder `./data/M4`. Moreover, we unzip the file `./Point Forecasts/submission-Naive2.rar` to the current directory.

```
|-M4-methods
| |-Dataset
| | |-Test
| | | |-Daily-test.csv
| | | |-Hourly-test.csv
| | | |-Monthly-test.csv
| | | |-Quarterly-test.csv
| | | |-Weekly-test.csv
| | | |-Yearly-test.csv
| | |-Train
| | | |-Daily-train.csv
| | | |-Hourly-train.csv
| | | |-Monthly-train.csv
| | | |-Quarterly-train.csv
| | | |-Weekly-train.csv
| | | |-Yearly-train.csv
| | |-M4-info.csv
| |-Point Forecasts
| | |-submission-Naive2.rar
```

Then you can get the folder tree shown as below:

```
|-data
| |-ECL
| | |-ECL.csv
| |
| |-ETT
| | |-ETTh1.csv
| | |-ETTh2.csv
| | |-ETTm1.csv
| | |-ETTm2.csv
| |
| |-M4
| | |-Dataset
| | | |-Test
| | | | |-Daily-test.csv
| | | | |-Hourly-test.csv
| | | | |-Monthly-test.csv
| | | | |-Quarterly-test.csv
| | | | |-Weekly-test.csv
| | | | |-Yearly-test.csv
| | | |-Train
| | | | |-Daily-train.csv
| | | | |-Hourly-train.csv
| | | | |-Monthly-train.csv
| | | | |-Quarterly-train.csv
| | | | |-Weekly-train.csv
| | | | |-Yearly-train.csv
| | | |-M4-info.csv
| | |-Point Forecasts
| | | |-submission-Naive2.csv
| |
| |-PEMS
| | |-PEMS08.npz
| |
| |-Ship
| | |-Ship1.csv
| | |-Ship2.csv
| |
| |-Solar
| | |-solar_AL.txt
| |
| |-weather
| | |-weather.csv

```

## Baseline
We select eight typical deep time series forecasting models based on CNN (SCINet), RNN (LSTNet), GNN (GTA), Transformer (Non-stationary Transformer, PatchTST, Crossformer) and Perceptron/MLP (DLinear, FiLM) as baselines in multivariate forecasting experiments. Two traditional time series forecasting models {ARIMA, Simple Exponential Smooth (SES)} and two deep forecasting model (N-HiTS, discover_PLF) are chosen as additional baselines only when handling the univariate dataset M4. Their source codes origins are given below:

| Baseline | Source Code |
|:---:|:---:|
| Non-stationary Transformer | [https://github.com/thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers) |
| PatchTST | [https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) |
| Crossformer | [https://github.com/Thinklab-SJTU/Crossformer](https://github.com/Thinklab-SJTU/Crossformer) |
| SCINet | [https://github.com/cure-lab/SCINet](https://github.com/cure-lab/SCINet) |
| LSTNet | [https://github.com/laiguokun/LSTNet](https://github.com/laiguokun/LSTNet) |
| discover_PLF | [https://github.com/houjingyi-ustb/discover_PLF](https://github.com/houjingyi-ustb/discover_PLF) | 
| DLinear | [https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) |
| N-HiTS | [https://github.com/cchallu/n-hits](https://github.com/cchallu/n-hits) |
| FiLM | [https://github.com/tianzhou2011/FiLM](https://github.com/tianzhou2011/FiLM) |
| GTA | [https://github.com/ZEKAICHEN/GTA](https://github.com/ZEKAICHEN/GTA) |

Moreover, the default experiment settings/parameters of aforementioned ten baselines are given below respectively:

<table>
<tr>
<th>Baselines</th>
<th>Settings/Parameters name</th>
<th>Descriptions</th>
<th>Default mechanisms/values</th>
</tr>
<tr>
<th rowspan=7>Non-stationary Transformer</th>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>512</th>
</tr>
<tr>
<th>d_ff</th>
<th>Dimension of fcn</th>
<th>2048</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>8</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>2</th>
</tr>
<tr>
<th>d_layers</th>
<th>The number of decoder layers</th>
<th>1</th>
</tr>
<tr>
<th>p_hidden_dims</th>
<th>Hidden layer dimensions of projector (List)</th>
<th>[128, 128]</th>
</tr>
<tr>
<th>p_hidden_layers</th>
<th>The number of hidden layers in projector</th>
<th>2</th>
</tr>
<tr>
<th rowspan=2>PatchTST</th>
<th>patch_len</th>
<th>Patch length</th>
<th>16</th>
</tr>
<tr>
<th>stride</th>
<th>The stride length</th>
<th>8</th>
</tr>
<tr>
<th rowspan=2>discover_PLF</th>
<th>hidden_size</th>
<th>The number of hidden dimensions</th>
<th>128</th>
</tr>
<tr>
<th>num_layers</th>
<th>The number of decoder layers</th>
<th>1</th>
</tr>
<tr>
<th rowspan=5>Crossformer</th>
<th>seq_len</th>
<th>Segment length (L_seq)</th>
<th>6</th>
</tr>
<tr>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>64</th>
</tr>
<tr>
<th>d_ff</th>
<th>Dimension of fcn</th>
<th>128</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>2</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>2</th>
</tr>
<tr>
<th rowspan=3>SCINet</th>
<th>hidden-size</th>
<th>The number of hidden dimensions</th>
<th>8</th>
</tr>
<tr>
<th>levels</th>
<th>SCINet block levels</th>
<th>3</th>
</tr>
<tr>
<th>stacks</th>
<th>The number of SCINet blocks</th>
<th>1</th>
</tr>
<tr>
<th rowspan=5>LSTNet</th>
<th>hidCNN</th>
<th>The number of CNN hidden units</th>
<th>100</th>
</tr>
<tr>
<th>hidRNN</th>
<th>The number of RNN hidden units</th>
<th>100</th>
</tr>
<tr>
<th>window</th>
<th>Window size</th>
<th>168</th>
</tr>
<tr>
<th>CNN_kernel</th>
<th>The kernel size of the CNN layers</th>
<th>6</th>
</tr>
<tr>
<th>hidSkip</th>
<th>The skip-length Recurrent-skip layer</th>
<th>24</th>
</tr>
<tr>
<th>DLinear</th>
<th>moving_avg</th>
<th>The window size of moving average</th>
<th>25</th>
</tr>
<tr>
<th rowspan=4>N-HiTS</th>
<th>n_pool_kernel_size</th>
<th>Pooling kernel size</th>
<th>[4, 4, 4]</th>
</tr>
<tr>
<th>n_blocks</th>
<th>The number of blocks in stacks</th>
<th>[1, 1, 1]</th>
</tr>
<tr>
<th>n_x_hidden</th>
<th>Coefficients hidden dimensions</th>
<th>512</th>
</tr>
<tr>
<th>n_freq_downsample</th>
<th>The number of stacks' coefficients</th>
<th>[60, 8, 1]</th>
</tr>
<tr>
<th rowspan=6>FiLM</th>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>512</th>
</tr>
<tr>
<th>d_ff</th>
<th>Dimension of fcn</th>
<th>2048</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>8</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>2</th>
</tr>
<tr>
<th>d_layers</th>
<th>The number of decoder layers</th>
<th>1</th>
</tr>
<tr>
<th>modes1</th>
<th>The number of Fourier modes to multiply</th>
<th>32</th>
</tr>
<tr>
<th rowspan=5>GTA</th>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>512</th>
</tr>
<tr>
<th>d_ff</th>
<th>Dimension of fcn</th>
<th>2048</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>8</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>2</th>
</tr>
<tr>
<th>d_layers</th>
<th>The number of decoder layers</th>
<th>1</th>
</tr>
</table>

## Usage
Commands for training and testing MFND<sup>3</sup>R of all datasets are in `./scripts/MFND3R.sh`.

More parameter information please refer to `main.py`.

We provide a complete command for training and testing MFND<sup>3</sup>R:

```
python -u main.py --model <model> --mode <mode> --data <data> --root_path <root_path> --features <features> --input_len <input_len> --pred_len <pred_len> --ODA_layers <ODA_layers> --VRCA_layers <VRCA_layers> --d_model <d_model> --learning_rate <learning_rate> --dropout <dropout> --batch_size <batch_size> --use_RevIN --train_epochs <train_epochs> --patience <patience> --itr <itr>
```

Here we provide a more detailed and complete command description for training and testing the model:

| Parameter name | Description of parameter |
|:---:|:---:|
| model | The model of experiment. This can be set to `MFND3R` |
| mode | Forecasting format |
|      data      | The dataset name  |
|   root_path    | The root path of the data file  |
|   data_path    | The data file name  |
|    features    | The forecasting task. This can be set to `M`,`S` (M : multivariate forecasting, S : univariate forecasting |
|     target     |  Target feature in `S` task   |
|  checkpoints   | Location of model checkpoints |
| input_len |Input sequence length |
| pred_len | Prediction sequence length |
| enc_in | Input size |
| c_out | Output size  |
| d_model | Dimension of model |
| representation | Representation dims in the end of the intra-reconstruction phase |
| dropout | Dropout  |
| ODA_layers |The number of ODA layers |
| VRCA_layers |The number of VRCA layers |
| alpha | The significant level of Cucconi test |
| itr | Experiments times |
| train_epochs | Train epochs of the second stage |
| batch_size | The batch size of training input data in the second stage |
| patience | Early stopping patience  |
| learning_rate | Optimizer learning rate |
| loss | Loss function |
| use_RevIN | Whether to use RevIN|


## Results

The experiment parameters of each data set are formated in the `MFND3R.sh` files in the directory `./scripts/`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse and mae results or draw better prediction figures.

<p align="center">
<img src="./img/Multivariate.png" height = "800" alt="" align=center />
<br><br>
<b>Figure 2.</b> Multivariate forecasting results
</p>

<p align="center">
<img src="./img/Univariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 3.</b> Univariate forecasting results
</p>


## Contact
If you have any questions, feel free to contact Li Shen through Email (shenli@buaa.edu.cn) or Github issues. Pull requests are highly welcomed!
