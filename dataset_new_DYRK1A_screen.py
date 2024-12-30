import pandas as pd
import numpy as np
from utils_new_hyperopt import smiles2adjoin
import tensorflow as tf
   



str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br': 9,
           'B': 10, 'I': 11, 'Si': 12, 'Se': 13, '<unk>': 14, '<mask>': 15, '<global>': 16,
           'single': 17, 'double': 18, 'triple': 19,'ben_double': 20}


num2str =  {i:j for j,i in str2num.items()} 





class Graph_Classification_Dataset(object):
    def __init__(self, path_train, smiles_field='Smiles',label_field='Label',max_len=100,addH=True):
        if path_train.endswith('.txt') or path_train.endswith('.tsv'):
            self.df_train = pd.read_csv(path_train,sep='\t')
        else:
            self.df_train = pd.read_csv(path_train)  
            #self.df_test = pd.read_csv(path_test) 

        self.smiles_field = smiles_field 
        self.vocab = str2num
        self.devocab = num2str
        self.df_train = self.df_train[self.df_train[smiles_field].str.len() <= max_len] 
        #self.df_test = self.df_test[self.df_test[smiles_field].str.len() <= max_len]
        self.addH = addH
        

    def get_data(self):
        train_data = self.df_train
        #val_data = self.df_test
        #test_data = self.df_test

        # train_data = self.df_train[: 0.8*self.df_train] 
        # val_data = self.df_train[0.8*self.df_train :]
        # test_data = self.df_test

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field]))    
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(1, padded_shapes=(     
            tf.TensorShape([None]), tf.TensorShape([None, None]),tf.TensorShape([None, None]))).cache().prefetch(1)    
        
        return self.dataset1

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode() 

       
        atoms_list, adjoin_matrix,distance_angle_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp1 = np.ones((len(nums_list),len(nums_list)))
        temp1[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp1)*(-1e9)

        temp2 = np.ones((len(nums_list), len(nums_list)))
        temp2[1:, 1:] = distance_angle_matrix*(1e-4)
        distance_angle_matrix = temp2

        x = np.array(nums_list).astype('int64')
 

        return x, adjoin_matrix,distance_angle_matrix 

    def tf_numerical_smiles(self, smiles):
        x,adjoin_matrix,distance_angle_matrix = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        distance_angle_matrix.set_shape([None,None])


        return x, adjoin_matrix , distance_angle_matrix
  
