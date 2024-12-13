# GEO-BERT

## Environmental Installation

```bash
**conda create -n GEO-BERT python==3.7**
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
