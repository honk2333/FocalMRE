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

You can simply run the bash script for multimodal relation extraction:
```shell
sh run_mre.sh
```

## Test the model
Anonymous Repository is unable to upload large model files. Considering anonymity, we are temporarily unable to upload checkpoint to the cloud storage. All materials will be made public later.

