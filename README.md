# GEO-BERT
## Introduction
-----------------------------------
Virtual screening is widely acknowledged to accelerate early-stage drug discovery, but building highly accurate and practically useful models is extremely challenging. To this end, we release a newly developed Geometry-based BERT (GEO-BERT) algorithm that is dedicated to building more accurate and convenient models for molecular property prediction in drug discovery. Due to the incorporation of three-dimensional structural information of small molecules, GEO-BERT demonstrated state-of the-art prediction performance, when evaluated on most public datasets. Unlike all the previous work that lack proof-of-concept, GEO-BERT has been applied to drug discovery and contributed to the discovery of two novel and potent DYRK1A inhibitors. We believe that a wide range of pharmaceutical scientists will benefit from the release of this GEO-BERT model. 

![image](https://github.com/user-attachments/assets/1620b1e7-0ba2-4dbf-9190-2f93f128a512)


## Installation
-----------------------------------
```bash
conda create -n GEO-BERT python==3.7
conda activate GEO-BERT
conda install -c openbabel openbabel
pip install rdkit
pip install numpy
pip install pandas
conda install conda-forge/label/rc::matplotlib
conda install conda-forge::hyperopt
conda install conda-forge::scikit-learn
pip install tensorflow-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow==2.6.0
pip install keras==2.6

For other packages, users may install the latest version.
```

## Folders
-----------------------------------
A brief introduction to each python file:

1. conformer_select: incuding codes for processing pre-training datasets.

2. pretrain_new: incuding codes for GEO-BERT pre-training tasks.

3. dataset_new: incuding codes for datasets for pre-training tasks.

4. dataset_scoffold_random: incuding codes for datasets for fine-tuning on public datasets.

5. dataset_new_DYRK1A: incuding codes for datasets for fine-tuning on DYRK1A datasets.

6. Class_hyperopt: incuding codes for GEO-BERT fine-tuning on public datasets.

7. Class_hyperopt_DYRK1A: incuding codes for GEO-BERT fine-tuning on DYRK1A datasets.

8. utilis_new-hyperop: incuding codes for converting molecules to three-dimensional conformations and calculating bond lengths and bond angles.

9. data: the public dataset and DYRK1A dataset for finetuing.

## Use example of GEO-BERT
-----------------------------------
1. Prepare for datasets and model weights

* First, use the command "cd GEO-BERT" to enter the current direcotry of GEO-BERT, and then use the command "unzip Medium. zip" to decompress the weight file with the file structure of "bert_weightsMedium_1.h5".

* Second, use the command "cd data" to enter the direcotry of data, and then use the command "unzip pretrain_datasets.zip" and "unzip datasets.zip" to decompress the pretraining and finetuning datasets, with the file structure of "data/chembl_conformer_select_145wan.txt". Put eight public datasets and DYRK1A datasets in "data/clf/...".

2. Pre-training

* Required files:

   utils_new_hyperopt.py, model_new_hyperopt.py, dataset_new.py, pretrain_new.py, data/chembl_conformer_select_145wan.txt
   
* use the command "python pretrain_new.py" to pre-train for GEO-BERT.

3. Fine-tuning for public datasets.

* Required files for BBBP dataset:

   utils_new_hyperopt.py, model_new_hyperopt.py,dataset_scaffold_random.py,Class_hyperopt.py,data/clf/BBBP.csv

* Use the command "python Class_hyperopt.py" to fine-tune for GEO-BERT on public datasets.

4. Fine-tuning for DYRK1A dataset.

* Required files for DYRK1A dataset:

   utils_new_hyperopt.py,model_new_hyperopt.py,dataset_new_DYRK1A.py, Class_hyperopt_DYRK1A.py, data/clf/DYRK1A_train.csv, data/clf/DYRK1A_test.csv

* Use the command "python Class_hyperopt_DYRK1A.py" to fine-tune for GEO-BERT on DYRK1A dataset.

## Application of GEO-BERT
-----------------------------------
1. Prepare for model weights

* Use the command "cd GEO-BERT" to enter the current direcotry of GEO-BERT, and then use the command "unzip model_weights_DYRK1A.zip" to decompress the weight file with the file structure of "model_weights_DYRK1A.h5".
  
2. Start predicting

* Use the command " python predict.py --input_path "Your_SMILES_file.txt" --output_path "Your_Prediction_output.txt" "

## Acknowledgments
-----------------------------------
This program is partially built on MG-BERT and FG-BERT. We are grateful for their open-source codes.

## References
-----------------------------------
If you find GEO-BERT useful, please cite: 

Xiang Zhang, Chenliang Qian, Bochao Yang, Hongwei Jin, Song Wu, Jie Xia*, Fan Yang*, Liangren Zhang. Geometry-based BERT: an experimentally validated deep learning model for molecular property prediction in drug discovery, bioRxiv 2024.12.24.630211; doi: https://doi.org/10.1101/2024.12.24.630211

