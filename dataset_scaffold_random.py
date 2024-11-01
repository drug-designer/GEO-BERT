from macpath import split
from operator import concat
import re
from cProfile import label
from cgi import test
from tkinter import Label
import pandas as pd
import numpy as np
import tensorflow as tf
from utils_new_hyperopt import smiles2adjoin
from rdkit import Chem
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from itertools import compress

str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br': 9,
           'B': 10, 'I': 11, 'Si': 12, 'Se': 13, '<unk>': 14, '<mask>': 15, '<global>': 16,
           'single': 17, 'double': 18, 'triple': 19,'ben_double': 20}
         
num2str =  {i:j for j,i in str2num.items()}


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset.iloc[train_idx]
    valid_dataset = dataset.iloc[valid_idx]
    test_dataset = dataset.iloc[test_idx]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


class Graph_Classification_Dataset(object):  # Graph classification task data set processing
    def __init__(self,path,smiles_field='Smiles',label_field='Label',max_len=100,seed=1,batch_size=32,a=2,addH=True):   #源代码 label_field=label  max_len=500
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')   
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field  
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH

    def get_data(self):

        '''Randomized Split Dataset'''
        # data = self.df
        # data = data.fillna(666)
        # train_idx = []
        # idx = data.sample(frac=0.8).index
        # train_idx.extend(idx)
        # train_data = data[data.index.isin(train_idx)]
        # data = data[~data.index.isin(train_idx)]
        # test_idx = []
        # idx = data[~data.index.isin(train_data)].sample(frac=0.5).index
        # test_idx.extend(idx)
        # test_data = data[data.index.isin(test_idx)]
        # val_data = data[~data.index.isin(train_idx+test_idx)]

        '''Splitting the dataset by random molecular scaffold, random_scaffold_split'''
        # data = self.df
        # data = data.fillna(666)
        # train_ids, val_ids, test_ids = random_scaffold_split(data, sizes=(0.8, 0.1, 0.1), balanced=True,seed=self.seed)
        # train_data = data.iloc[train_ids]
        # val_data = data.iloc[val_ids]
        # test_data = data.iloc[test_ids]
        # df_train_data = pd.DataFrame(train_data)
        # df_test_data = pd.DataFrame(test_data)
        # df_val_data = pd.DataFrame(val_data)

        '''Scaffold Split Dataset, scaffold_split'''
        data = self.df
        smiles_list= data[self.smiles_field]    
        df_train_data,df_val_data,df_test_data = scaffold_split(data, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False)

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (df_train_data[self.smiles_field], df_train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(batch_size=self.batch_size, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]),  tf.TensorShape([None, None]),tf.TensorShape([self.a]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((df_test_data[self.smiles_field], df_test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]),  tf.TensorShape([None, None]),tf.TensorShape([self.a]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((df_val_data[self.smiles_field], df_val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]),  tf.TensorShape([None, None]),tf.TensorShape([self.a]))).cache().prefetch(100)

        #self.dataset1 = self.dataset1.prefetch(tf.data.experimental.AUTOTUNE)
        #self.dataset2 = self.dataset2.prefetch(tf.data.experimental.AUTOTUNE)
        #self.dataset3 = self.dataset3.prefetch(tf.data.experimental.AUTOTUNE)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):  
        smiles = smiles.numpy().decode()
        
        atoms_list, adjoin_matrix,distance_angle_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]


        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        temp2 = np.ones((len(nums_list), len(nums_list)))
        temp2[1:, 1:] = distance_angle_matrix*(1e-4)
        #temp2[1:, 1:] = 0
        distance_angle_matrix = temp2


        x = np.array(nums_list).astype('int64')
        y = np.array(label).astype('int64')

        return x, adjoin_matrix,distance_angle_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,distance_angle_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.float32,tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        distance_angle_matrix.set_shape([None,None])
        y.set_shape([None])

        return x, adjoin_matrix ,distance_angle_matrix, y


class Graph_Regression_Dataset(object):  #  Graph regression task dataset processing
    def __init__(self,path,smiles_field='Smiles',label_field=label,seed=1,batch_size=32,a=1,max_len=100,normalize=True,addH=True):  #原代码 max_len=100
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min

    def get_data(self):

        ''' random split'''
        # data = self.df        
        # train_idx = []
        # idx = data.sample(frac=0.8).index
        # train_idx.extend(idx)
        # train_data = data[data.index.isin(train_idx)]

        # data = data[~data.index.isin(train_idx)]
        # test_idx = []
        # idx = data[~data.index.isin(train_data)].sample(frac=0.5).index
        # test_idx.extend(idx)
        # test_data = data[data.index.isin(test_idx)]

        # val_data = data[~data.index.isin(train_idx+test_idx)]

        '''random_scaffold_split'''
        # data = self.df
        # train_ids, val_ids, test_ids = random_scaffold_split(self.df, sizes=(0.8, 0.1, 0.1), balanced= False,seed=self.seed)
        # train_data = self.df.iloc[train_ids]
        # val_data = self.df.iloc[val_ids]
        # test_data = self.df.iloc[test_ids]

        # df_train_data = pd.DataFrame(train_data)
        # df_test_data = pd.DataFrame(test_data)
        # df_val_data = pd.DataFrame(val_data)

        '''scaffold_split'''
        data = self.df
        smiles_list= data[self.smiles_field]    
        df_train_data,df_val_data,df_test_data = scaffold_split(data, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False)

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (df_train_data[self.smiles_field], df_train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(batch_size=self.batch_size, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((df_test_data[self.smiles_field], df_test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),tf.TensorShape([self.a]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((df_val_data[self.smiles_field], df_val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        #self.dataset1 = self.dataset1.prefetch(tf.data.experimental.AUTOTUNE)
        #self.dataset2 = self.dataset2.prefetch(tf.data.experimental.AUTOTUNE)
        #self.dataset3 = self.dataset3.prefetch(tf.data.experimental.AUTOTUNE)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix,distance_angle_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]

        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        temp2 = np.ones((len(nums_list), len(nums_list)))
        temp2[1:, 1:] = distance_angle_matrix*(1e-4)
        distance_angle_matrix = temp2
        
        x = np.array(nums_list).astype('int64')
        y = np.array(label).astype('float32')

        return x, adjoin_matrix,distance_angle_matrix,y 

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,distance_angle_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        distance_angle_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , distance_angle_matrix,y


