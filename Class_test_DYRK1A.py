import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from dataset_new_DYRK1A import Graph_Classification_Dataset
import os
from model_new_hyperopt import PredictModel, BertModel
from sklearn.metrics import roc_auc_score, precision_score, recall_score, matthews_corrcoef, f1_score
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def compute_confusion_matrix(precited, expected):          
    part = precited ^ expected  
    part = part.astype(np.int64)
    pcount = np.bincount(part)  
    tp_list = list(precited & expected)  
    fp_list = list(precited & ~expected)  
    tp = tp_list.count(1) 
    fp = fp_list.count(1)  
    tn = pcount[0] - tp  
    fn = pcount[1] - fp  
    return tp, fp, tn, fn

task = 'DYRK1A'

def main(args):

    if task =='DYRK1A':
        label = ['label']
   
    arch = {'name': 'Medium', 'path': 'Medium'}   
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''
    trained_epoch = 1
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    vocab_size = 21

    num_heads = args['num_heads']
    dense_dropout = args['dense_dropout']   
    learning_rate = args['learning_rate'] 
    batch_size = args['batch_size']
    seed = 0
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)  

    train_dataset, test_dataset, val_dataset = Graph_Classification_Dataset('data/clf/DYRK1A_IC50_train.csv', 
                                                                             'data/clf/DYRK1A_IC50_test.csv',
                                                                             smiles_field='SMILES',
                                                               label_field='Type(active or not)',
                                                               addH=True,
                                                               batch_size=batch_size).get_data()  
                                                  
    x, adjoin_matrix, distance_angle_matrix,y = next(iter(train_dataset.take(1))) 
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,a=len(label),
                         dense_dropout = dense_dropout) 

    if pretraining:
        
        temp = BertModel(num_layers=num_layers, d_model=d_model,
                         dff=dff, num_heads=num_heads, vocab_size=vocab_size) 

        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix,distance_angle_matrix=distance_angle_matrix)
        temp.load_weights( 
            arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'], trained_epoch))
        temp.encoder.save_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        del temp
        
        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix, distance_angle_matrix=distance_angle_matrix)
        model.encoder.load_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        print('load_wieghts')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    auc = -10
    stopping_monitor = 0
    for epoch in range(200):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for x, adjoin_matrix, distance_angle_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x, mask=mask, training=True,adjoin_matrix=adjoin_matrix,distance_angle_matrix=distance_angle_matrix)
                loss = 0
                for i in range(len(label)):
                    y_label = y[:,i]
                    y_pred = preds[:,i]
                    validId = np.where((y_label == 0) | (y_label == 1))[0]
                    if len(validId) == 0:
                        continue
                    y_t = tf.gather(y_label,validId)
                    y_p = tf.gather(y_pred,validId)
            
                    loss += loss_object(y_t, y_p)
                loss = loss/(len(label))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss.numpy().item()))

        y_true = {}
        y_preds = {}
        for i in range(len(label)):
            y_true[i] = []
            y_preds[i] = []

        for x, adjoin_matrix, distance_angle_matrix,y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False,distance_angle_matrix=distance_angle_matrix)
            for i in range(len(label)):
                y_label = y[:,i]
                y_pred = preds[:,i]
                y_true[i].append(y_label)
                y_preds[i].append(y_pred)
        y_tr_dict = {}
        y_pr_dict = {}
        for i in range(len(label)):
            y_tr = np.array([])
            y_pr = np.array([])
            for j in range(len(y_true[i])):
                a = np.array(y_true[i][j])
                b = np.array(y_preds[i][j])
                y_tr = np.concatenate((y_tr,a))
                y_pr = np.concatenate((y_pr,b))
            y_tr_dict[i] = y_tr
            y_pr_dict[i] = y_pr

        auc_list,mcc_list,accuracy_list,precision_list,recall_list,f1_list,SP_list = [],[],[],[],[],[],[]

        for i in range(len(label)):
            y_label = y_tr_dict[i]
            y_pred = y_pr_dict[i]
            validId = np.where((y_label== 0) | (y_label == 1))[0]
            if len(validId) == 0:
                continue
            y_t = tf.gather(y_label,validId)
            y_p = tf.gather(y_pred,validId)
            if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
                AUC = float('nan')
                auc_list.append(AUC)
                continue
            y_p = tf.sigmoid(y_p).numpy()
            y_pl = np.where(y_p >= 0.5, 1, 0)
            y_p=y_p.tolist()
            y_pl=y_pl.tolist()
            
            AUC_new = roc_auc_score(y_t, y_p, average=None)
            MCC = matthews_corrcoef(y_t, y_pl)
            precision = precision_score(y_t, y_pl)
            recall = recall_score(y_t, y_pl)
            f1 = f1_score(y_t, y_pl)
            Y1 = np.array(y_pl).flatten()
            T1 = np.array(y_t).flatten()
            Y1=Y1.astype(np.int64)
            T1=T1.astype(np.int64)
            tp, fp, tn, fn = compute_confusion_matrix(Y1, T1)
            ACC = (tp + tn) / (tp + fp + tn + fn)
            SE = tp / (tp + fn)
            SP = tn / (tn + fp)

            auc_list.append(AUC_new)
            mcc_list.append(MCC)
            accuracy_list.append(ACC)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            SP_list.append(SP)
              
        auc_new = np.nanmean(auc_list)
        mcc_new = np.nanmean(mcc_list)
        ACC_new = np.nanmean(accuracy_list)
        precision_new = np.nanmean(precision_list)
        recall_new = np.nanmean(recall_list)
        f1_new = np.nanmean(f1_list)
        SP_new = np.nanmean(SP_list)

        print('val auc:{:.4f} mcc:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} SP:{:.4f} '
              .format(auc_new,mcc_new,ACC_new,precision_new,recall_new,f1_new,SP_new))
        if auc_new> auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch, pretraining_str),
                    [y_true, y_preds])
            if not os.path.exists('model_weights_DYRK1A'):
                os.makedirs('model_weights_DYRK1A')
            model.save_weights('model_weights_DYRK1A/model_weights_DYRK1A.h5')
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 30: 
            break

    y_true = {}
    y_preds = {}
    for i in range(len(label)):
        y_true[i] = []
        y_preds[i] = []
    model.load_weights('model_weights_DYRK1A/model_weights_DYRK1A.h5')
    for x, adjoin_matrix,distance_angle_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix,training=False,distance_angle_matrix=distance_angle_matrix)
        for i in range(len(label)):
            y_label = y[:,i]
            y_pred = preds[:,i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)  
    y_tr_dict = {}
    y_pr_dict = {}
    for i in range(len(label)):
        y_tr = np.array([])
        y_pr = np.array([])
        for j in range(len(y_true[i])):
            a = np.array(y_true[i][j])
            if a.ndim == 0:
                continue
            b = np.array(y_preds[i][j])
            y_tr = np.concatenate((y_tr,a))
            y_pr = np.concatenate((y_pr,b))
        y_tr_dict[i] = y_tr
        y_pr_dict[i] = y_pr
    auc_list,mcc_list,accuracy_list,precision_list,recall_list,f1_list,SP_list = [],[],[],[],[],[],[]
    
    for i in range(len(label)):
        y_label = y_tr_dict[i]
        y_pred = y_pr_dict[i]
        
        validId = np.where((y_label== 0) | (y_label == 1))[0]
        if len(validId) == 0:
            continue
        y_t = tf.gather(y_label,validId)
        y_p = tf.gather(y_pred,validId)
        if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
            AUC = float('nan')
            auc_list.append(AUC)
            continue
        y_p = tf.sigmoid(y_p).numpy()
        y_pl = np.where(y_p >= 0.5, 1, 0)
        AUC_new = roc_auc_score(y_t, y_p, average=None)
        MCC = matthews_corrcoef(y_t, y_pl)
        precision = precision_score(y_t, y_pl)
        recall = recall_score(y_t, y_pl)
        f1 = f1_score(y_t, y_pl)
        Y1 = np.array(y_pl).flatten()
        T1 = np.array(y_t).flatten()
        Y1=Y1.astype(np.int64)
        T1=T1.astype(np.int64)
        tp, fp, tn, fn = compute_confusion_matrix(Y1, T1)
        ACC = (tp + tn) / (tp + fp + tn + fn)
        SE = tp / (tp + fn)
        SP = tn / (tn + fp)

        auc_list.append(AUC_new)
        mcc_list.append(MCC)
        accuracy_list.append(ACC)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        SP_list.append(SP)

    auc_new = np.nanmean(auc_list)
    mcc_new = np.nanmean(mcc_list)
    ACC_new = np.nanmean(accuracy_list)
    precision_new = np.nanmean(precision_list)
    recall_new = np.nanmean(recall_list)
    f1_new = np.nanmean(f1_list)
    SP_new = np.nanmean(SP_list)
    print('test auc:{:.4f} mcc:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} SP:{:.4f} '
              .format(auc_new,mcc_new,ACC_new,precision_new,recall_new,f1_new,SP_new))

    return auc, auc_new, auc_list,mcc_new,ACC_new,precision_new,recall_new,f1_new,SP_new
                                                           

if __name__ == '__main__':  

    args = {'batch_size': 16, 'dense_dropout': 0.4, 'learning_rate': 7.68841061442732e-05, 'num_heads': 8}

    auc, test_auc, a_list, mcc_new,ACC_new,precision_new,recall_new,f1_new,SP_new= main(args)
    print("test auc:", test_auc)
    print("test mcc:", mcc_new)
    print("test accuracy:", ACC_new)
    print("test precision:", precision_new)
    print("test recall:", recall_new)
    print("test f1:", f1_new)
    print("test SP:", SP_new)
