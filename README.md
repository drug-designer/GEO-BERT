# GEO-BERT

## 环境安装示例

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
其他软件包可以使用最新版本进行安装。
```

## 文件夹结构

pretrain_new: 包含GEO-BERT预训练任务的代码。

dataset_scoffold_random: 包含构建用于预训练和微调的数据集的代码。

utils_new_hyperopt: 包含将分子转换为三维结构并计算键长和键角的代码。

data: 包含每个下游任务的数据集。

## GEO-BERT 的使用示例

### 预训练示例

所需文件：

utils_new_hyperopt.py

model_new_hyperopt.py

dataset_scaffold_random.py

pretrain.py

data.txt

python pretrain.py

### 微调示例

DYRK1A 数据集所需文件：

utils_new_hyperopt.py

model_new_hyperopt.py

dataset_new_DYRK1A.py

Class_hyperopt_DYRK1A.py

DYRK1A.csv

python Class_hyperopt_DYRK1A.py

## 致谢
该代码部分基于 MG-BERT和FG-BERT 构建。非常感谢他们的开源代码！

