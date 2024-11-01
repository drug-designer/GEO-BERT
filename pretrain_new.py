import tensorflow as tf
from model_new import  BertModel
from dataset_new import Graph_Bert_Dataset
import time
import os
import tensorflow.keras as keras

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
#
# learning_rate = CustomSchedule(128)


optimizer = tf.keras.optimizers.Adam(1e-4)

small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights','addH':True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights','addH':True}
medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': '8wan_medium_weights_new1_4','addH':True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights','addH':True}
medium_balanced = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'weights_balanced','addH':True}
medium_without_H = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'weights_without_H','addH':False}

arch = medium3      ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']
addH = arch['addH']


dff = d_model*2
vocab_size =21
dropout_rate = 0.1

model = BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)

train_dataset, test_dataset = Graph_Bert_Dataset(path='data/chembl_conformer_select_145wan.txt',smiles_field='Smiles',addH=addH).get_data()

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None,None), dtype=tf.float32),   
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),   
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_step(x, adjoin_matrix,y, char_weight,distance_angle_matrix,epoch):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]    
    #print(mask)

    with tf.GradientTape() as tape:

        temp = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix,distance_angle_matrix=distance_angle_matrix)
        if epoch==0 and batch==0:
            model.load_weights('8wan_medium_weights_new1_4/bert_weightsMedium_4.h5') 
            print('load successful')

        predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=True,distance_angle_matrix=distance_angle_matrix)    
        loss = loss_function(y,predictions,sample_weight=char_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(y,predictions,sample_weight=char_weight)


@tf.function(input_signature=train_step_signature)
def test_step(x, adjoin_matrix,y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
    test_accuracy.update_state(y,predictions,sample_weight=char_weight)



for epoch in range(19):
    start = time.time()
    train_loss.reset_states()


    for (batch, (x, adjoin_matrix ,y , char_weight,distance_angle_matrix)) in enumerate(train_dataset):
        
        #print(x)
        #print(adjoin_matrix)
        #print(y)
        #print(char_weight)


        train_step(x, adjoin_matrix, y , char_weight,distance_angle_matrix,epoch)

        if batch % 15 == 0:    #500
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 5, batch, train_loss.result()))
            print('Accuracy: {:.4f}'.format(train_accuracy.result()))
            #
            # for x, adjoin_matrix ,y , char_weight in test_dataset:
            #     test_step(x, adjoin_matrix, y , char_weight)
            # print('Test Accuracy: {:.4f}'.format(test_accuracy.result()))
            # test_accuracy.reset_states()
            train_accuracy.reset_states()
            #for i, w in enumerate(model.weights):
              #print(i, w.name)

    model.save_weights(arch['path'] + '/bert_weights{}_{}.h5'.format(arch['name'], epoch + 5))

    print(arch['path'] + '/bert_weights{}_{}.h5'.format(arch['name'], epoch+5))
    print('Epoch {} Loss {:.4f}'.format(epoch + 5, train_loss.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Accuracy: {:.4f}'.format(train_accuracy.result()))
    print('Saving checkpoint')


    





