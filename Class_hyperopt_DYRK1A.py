import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
#from dataset_new_DYRK1A import Graph_Classification_Dataset
from dataset_scaffold_random import Graph_Classification_Dataset
import os
from model_new_hyperopt import PredictModel, BertModel
from sklearn.metrics import roc_auc_score, precision_score, recall_score, matthews_corrcoef, f1_score
from hyperopt import fmin, tpe, hp
#from utils import get_task_names
from tensorflow.python.client import device_lib
import os
import csv



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

def main(seed, args):

    # tasks = ['BBBP', 'bace', 'HIV','clintox', 'tox21', 'muv', 'sider','toxcast_data']

    task = 'toxcast_data'

    if task == 'DYRK1A_IC50_all':
        label = ['Label']

    elif task =='BBBP':
        label = ['p_np']

    elif task =='bace':
        label = ['Class']

    elif task == 'HIV':
        label = ['HIV_active']  

    elif task == 'clintox':
        label = ['FDA_APPROVED', 'CT_TOX']

    elif task == 'tox21':
        label = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        
    elif task == 'muv':
        label = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652',	'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832',	'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
    ]
        
    elif task == 'sider':
        label = ['Hepatobiliary disorders','Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations','Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 
        'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders', 
        'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 
        'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders', 'Injury, poisoning and procedural complications'
    ]

    elif task == 'toxcast_data':
        label = get_task_names('data/clf/toxcast_data.csv')

   

    arch = {'name': 'Medium', 'path': '8wan_medium_weights_new1_4'}
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
    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)

    """
    train_dataset, test_dataset , val_dataset = Graph_Classification_Dataset('data/DYRK1A/DYRK1A_IC50_train.csv', 
                                                                             'data/DYRK1A/DYRK1A_IC50_test.csv',
                                                                             smiles_field='SMILES',
                                                               label_field='Type(active or not)',addH=True).get_data()  
    """

    train_dataset, test_dataset, val_dataset = Graph_Classification_Dataset('data/clf/toxcast_data.csv', smiles_field='smiles',
                                                           label_field=label, seed=seed,batch_size=batch_size,a = len(label), addH=True).get_data()  #源代码 label_field=label
                                                        
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
        

        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix,distance_angle_matrix=distance_angle_matrix)
        model.encoder.load_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        print('load_wieghts')

    total_params = count_parameters(model)
    print('*'*100)
    print("Total Parameters:", total_params)
    print('*'*100)
    
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
            #print(y_p)
            MCC = matthews_corrcoef(y_t, y_pl)
            precision = precision_score(y_t, y_pl)
            recall = recall_score(y_t, y_pl)
            f1 = f1_score(y_t, y_pl)

            def compute_confusion_matrix(precited, expected):
               
                part = precited ^ expected  # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
                part = part.astype(np.int64)
                
                pcount = np.bincount(part)  # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
                tp_list = list(precited & expected)  # 将TP的计算结果转换为list
                fp_list = list(precited & ~expected)  # 将FP的计算结果转换为list
                tp = tp_list.count(1)  # 统计TP的个数
                fp = fp_list.count(1)  # 统计FP的个数
                tn = pcount[0] - tp  # 统计TN的个数
                fn = pcount[1] - fp  # 统计FN的个数
                return tp, fp, tn, fn
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
            model.save_weights('classification_weights/{}_{}.h5'.format(task, seed))
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
    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))
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
        #print(y_p)
        MCC = matthews_corrcoef(y_t, y_pl)
        precision = precision_score(y_t, y_pl)
        recall = recall_score(y_t, y_pl)
        f1 = f1_score(y_t, y_pl)

        def compute_confusion_matrix(precited, expected):
            part = precited ^ expected  # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
            part=part.astype(np.int64)
            pcount = np.bincount(part)  # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
            tp_list = list(precited & expected)  # 将TP的计算结果转换为list
            fp_list = list(precited & ~expected)  # 将FP的计算结果转换为list
            tp = tp_list.count(1)  # 统计TP的个数
            fp = fp_list.count(1)  # 统计FP的个数
            tn = pcount[0] - tp  # 统计TN的个数
            fn = pcount[1] - fp  # 统计FN的个数
            return tp, fp, tn, fn
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

    return auc, auc_new, auc_list     
                                                           

space = {"dense_dropout": hp.quniform("dense_dropout", 0, 0.1, 0.01), 
        "learning_rate": hp.loguniform("learning_rate", np.log(2e-5), np.log(10e-5)),
        "batch_size":hp.choice("batch_size", [16,32,48,64]),    
        "num_heads":hp.choice("num_heads", [4,8]),
        }



def hy_main(args):
    auc_list = []
    test_auc_list = []
    test_all_auc_list = []
    x = 0
    for seed in [0]:  #[0,1,2,3,4,5,6,7,8,9]
        print(seed)
        auc, test_auc, a_list= main(seed, args)
        auc_list.append(auc)
        test_auc_list.append(test_auc)
        test_all_auc_list.append(a_list)
        x+= test_auc
    auc_list.append(np.mean(auc_list))
    test_auc_list.append(np.mean(test_auc_list))
    print(auc_list)
    print(test_auc_list)
    print(test_all_auc_list)
    print(args["dense_dropout"])
    print(args["learning_rate"])
    print(args["batch_size"])
    print(args["num_heads"])
    return -x/10


best = fmin(hy_main, space, algo = tpe.suggest, max_evals= 30)
print(best)
best_dict = {}
a = [16,32,48,64]
b = [4, 8]
best_dict["dense_dropout"] = best["dense_dropout"]
best_dict["learning_rate"] = best["learning_rate"]
best_dict["batch_size"] = a[best["batch_size"]]
best_dict["num_heads"] = b[best["num_heads"]]
print(best_dict)
print(hy_main(best_dict))

# if __name__ == '__main__':

#     args = {"dense_dropout":0.4, "learning_rate":5.147496336624254e-05, "batch_size":32, "num_heads":8}
#     auc_list = []
#     test_auc_list = []
#     test_all_auc_list = []
#     for seed in [0,1,2,3,4,5,6,7,8,9]:
#         print(seed)
#         auc, test_auc, a_list= main(seed, args)
#         auc_list.append(auc)
#         test_auc_list.append(test_auc)  
#         test_all_auc_list.append(a_list)
#     auc_list.append(np.mean(auc_list))
#     test_auc_list.append(np.mean(test_auc_list))
#     print(auc_list)
#     print(test_auc_list)
#     print(test_all_auc_list)





