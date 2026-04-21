# SFAM

## 1. Build environment
You can run the following script to configure the necessary environment:

```
conda env create -f environment.yml
conda activate SFAM
```



## 2. Download Data
We conducted our research mainly based on DeepfakeBench. You can download the processed datasets from DeepfakeBench, and make sure to follow the structure of the datasets folder. 


## 3. Preprocessing (optional)
You can skip this step, if you only want to use the processed data from DeepfakeBench.

Otherwise, you need to do data preprocessing strictly following DeepfakeBench.


## 4. Rearrangement (optional)
After the preprocessing above, you need to set the parameters in `./preprocessing/config.yaml` for each dataset. After that, run the following script:
```
cd preprocessing

python rearrange.py
```

You will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. 


## 5. Training 

-We adopt a two-stage training strategy, first we train the PaITA module with:

```
python training/train.py
```

-Once you get the pre-trained weight of PaITA module, make sure you change the chech point path in `./training/train_moe.py` with your own saved weight to start the second training stage：
```
python training/train_moe.py
```


## 6. Test
If you want to evaluate the detector to get the cross-AUC result, you can test the model with:

```
python training/test.py --cross_auc True
```






