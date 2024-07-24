## Data preprocessing
### MNRE dataset
Due to the large size of MNRE dataset, please download the dataset from [original repository](https://github.com/thecharm/MNRE). 

Unzip the data and place them in the directory `data`

```shell
mkdir ckpt
```
We also use the detected visual objects provided in previous work, which can be downloaded using the commend:

```shell
cd data/
wget 120.27.214.45/Data/re/multimodal/data.tar.gz
tar -xzvf data.tar.gz
```

## Dependencies

Install all necessary dependencies:

```shell
conda create -n focalmre python==3.7
conda activate focalmre
pip install -r requirements.txt
```


## Training the model

The best hyperparameters we found have been witten in run_mre.sh file.

You can simply run the script for multimodal relation extraction:

```shell
sh run_mre.sh
```

## Test the model

You can simply run the script to test saved checkpoint:

```shell
sh run_test.sh
```

You can use our fine-tuned models for testing, which can be downloaded from the following link: https://drive.google.com/file/d/1Nff_sSB4n7p_qoE9ryK7qxmAYSW1TF2z/view?usp=sharing

