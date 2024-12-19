# GEO-BERT
## Introduction

Virtual screening is widely acknowledged to accelerate early-stage drug discovery, but building highly accurate and practically useful models is extremely challenging. To this end, we release a newly developed Geometry-based BERT (GEO-BERT) algorithm that is dedicated to building more accurate and convenient models for molecular property prediction in drug discovery. Due to the incorporation of three-dimensional structural information of small molecules, GEO-BERT demonstrated state-of the-art prediction performance, when evaluated on most public datasets. Unlike all the previous work that lack proof-of-concept, GEO-BERT has been applied to drug discovery and contributed to the discovery of two novel and potent DYRK1A inhibitors. We believe that a wide range of pharmaceutical scientists will benefit from the release of this GEO-BERT model. 

![image](https://github.com/user-attachments/assets/1620b1e7-0ba2-4dbf-9190-2f93f128a512)


## Environmental Installation

```bash
conda create -n GEO-BERT python==3.7
conda install -c openbabel openbabel
pip install tensorflow==2.3
pip install rdkit
pip install numpy
pip install pandas
pip install matplotlib
pip install hyperopt
pip install scikit-learn
Other packages can be installed using the latest version.
```

## Folder Structure

pretrain_new: contains code for GEO-BERT pre training tasks.

dataset-scoffold_random: contains code for building datasets for pre training and fine-tuning.

utilis_new-hyperop: contains code for converting molecules into three-dimensional structures and calculating bond lengths and bond angles.

data: contains the dataset for each downstream task.

## Example of using GEO-BERT

### Example of pre-training

Required files:

utils_new_hyperopt.py

model_new_hyperopt.py

dataset_scaffold_random.py

pretrain_new.py

data.txt

python pretrain.py

### Example of fine-tuning

Required files for DYRK1A dataset:

utils_new_hyperopt.py

model_new_hyperopt.py

dataset_new_DYRK1A.py

Class_hyperopt_DYRK1A.py

DYRK1A.csv

python Class_hyperopt_DYRK1A.py

## Acknowledgments

This code is partially built on MG-BERT and FG-BERT. We are deeply grateful for their open-source code contributions.
