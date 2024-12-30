import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from dataset_new_DYRK1A_screen import Graph_Classification_Dataset
import os
from model_new_hyperopt import PredictModel, BertModel
from sklearn.metrics import roc_auc_score, precision_score, recall_score, matthews_corrcoef, f1_score
from hyperopt import fmin, tpe, hp
from tensorflow.python.client import device_lib
import os
import csv
import pandas as pd
import time
import argparse


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header

def get_task_names(path, use_compound_names=False):  
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names

def count_parameters(model):
    total_params = 0
    for variable in model.trainable_variables: 
        shape = variable.shape
        params = 1
        for dim in shape:
            params *= dim
        total_params += params
    return total_params

def main(seed, input_path, output_path):
    
    # tasks = ['BBBP', 'bace', 'HIV','clintox', 'tox21', 'muv', 'sider','toxcast_data']
    
    task = 'DYRK1A'

    if task =='DYRK1A':
        label = ['label']

    arch = {'name': 'Medium', 'path': 'model_weights_DYRK1A'}
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else '' 
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    vocab_size = 21

    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed) 

    train_dataset = Graph_Classification_Dataset(input_path, smiles_field='SMILES',addH=True).get_data()  
                                            
    x, adjoin_matrix, distance_angle_matrix = next(iter(train_dataset.take(1)))
    
    seq = tf.cast(tf.math.equal(x, 0), tf.float32) 
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=4, vocab_size=vocab_size,a=len(label),
                         dense_dropout = 0.1)
    
    pred_temp = model(x, mask=mask, training=False, adjoin_matrix=adjoin_matrix,distance_angle_matrix=distance_angle_matrix)

    model.load_weights(
        arch['path']+'/model_weights_{}.h5'.format(task))  
    
    predictions = []
    
    for (batch, (x, adjoin_matrix, distance_angle_matrix)) in enumerate(train_dataset):

        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]

        pred = model(x, mask=mask, training=False, adjoin_matrix=adjoin_matrix, distance_angle_matrix=distance_angle_matrix)
        pred_sigmoid = tf.sigmoid(pred)
        predictions.append(pred_sigmoid.numpy().flatten()) 

        print('predicted samples: ', batch)   

    predictions_binary = ['active' if p > 0.5 else 'inactive' for p in np.concatenate(predictions)]

    # Load original txt
    df = pd.read_csv(input_path) 

    # Add predictions to the DataFrame
    df['prediction'] = predictions_binary

    # Save new txt
    df.to_csv(output_path, index=False)

    print('Prediction completed')


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="input.txt")
parser.add_argument('--output_path', type=str, default="output.txt")

args = parser.parse_args()



start_time = time.time()

if __name__ == "__main__": 
    main(seed=0, input_path=args.input_path, output_path=args.output_path)  # You can set your seed value here
    print('Time: {} secs\n'.format(time.time() - start_time)) 







    
    
