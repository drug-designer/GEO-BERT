# GEO-BERT
## Introduction

Virtual screening is widely acknowledged to accelerate early-stage drug discovery, but building highly accurate and practically useful models is extremely challenging. To this end, we release a newly developed Geometry-based BERT (GEO-BERT) algorithm that is dedicated to building more accurate and convenient models for molecular property prediction in drug discovery. Due to the incorporation of three-dimensional structural information of small molecules, GEO-BERT demonstrated state-of the-art prediction performance, when evaluated on most public datasets. Unlike all the previous work that lack proof-of-concept, GEO-BERT has been applied to drug discovery and contributed to the discovery of two novel and potent DYRK1A inhibitors. We believe that a wide range of pharmaceutical scientists will benefit from the release of this GEO-BERT model. 

![image](https://github.com/user-attachments/assets/1620b1e7-0ba2-4dbf-9190-2f93f128a512)


## Installation

```bash
conda create -n GEO-BERT python==3.7
conda install -c openbabel openbabel
pip install tensorflow==2.3
pip install rdkit
pip install numpy
pip install pandas
conda install conda-forge::matplotlib
conda install conda-forge::hyperopt
conda install conda-forge::scikit-learn
For other packages, users may install the latest version.
```

## Folders

A brief introduction to each code file:

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

### Prepare for datasets and model weights

1. First, use the command "cd GEO-BERT" to enter the current path of GEO-BERT, and then use the command "unzip Medium. zip" to decompress the weight file with the file structure of "Medium/bert_weights_encoded Medium_1. h5".

2. Secondly, use the command "cd data" to enter the path of data, and then use the command "unzip pretrain_datasets.zip" and "unzip datasets.zip" to decompress the pretraining and finetuning datasets, with the file structure of "data/chembl_conformer_select_145wan.txt". Put eight public datasets and DYRK1A datasets into "data/clf/...".

### Pre-training

Required files:

1. utils_new_hyperopt.py

2. model_new_hyperopt.py

3. dataset_new.py

4. pretrain_new.py

5. data/chembl_conformer_select_145wan.txt

Use the command "python pretrain_new.py" to pre-train for GEO-BERT.

### Fine-tuning for public datasets.

Required files for BBBP dataset:

1. utils_new_hyperopt.py

2. model_new_hyperopt.py

3. dataset_scaffold_random.py

4. Class_hyperopt.py

5. data/clf/BBBP.csv

Use the command "python Class_hyperopt.py" to fine-tune for GEO-BERT on public datasets.

### Fine-tuning for DYRK1A dataset.

Required files for DYRK1A dataset:

1. utils_new_hyperopt.py

2. model_new_hyperopt.py

3. dataset_new_DYRK1A.py

4. Class_hyperopt_DYRK1A.py

5. data/clf/DYRK1A_train.csv, data/clf/DYRK1A_test.csv

Use the command "python Class_hyperopt_DYRK1A.py" to fine-tune for GEO-BERT on DYRK1A dataset.

## Acknowledgments

This program is partially built on MG-BERT and FG-BERT. We are grateful for their open-source codes.
