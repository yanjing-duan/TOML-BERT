import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from dataset import Graph_Low_Fidelity_Classification_Dataset
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import PredictModel,BertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'small_weights', 'addH': True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3',
           'addH': True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights', 'addH': True}

def main(seed=7, arch = medium3, pretraining = True, trained_epoch = 8, task = 'x_bbbp', max_epoch = 100):

    if pretraining:
        print('After self-pretraining, task:',task)
    elif not pretraining:
        print('No self-pretraining, task:',task)

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    # dropout_rate = 0.1

    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset = Graph_Low_Fidelity_Classification_Dataset('data/all_low_fidelity_data/{}.csv'.format(task), smiles_field='Smiles',
                                                               label_field=task,addH=addH).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.15)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')
    elif not pretraining:
        print("No pre-training")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    auc= -10
    stopping_monitor = 0
    for epoch in range(max_epoch):
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = loss_object(y,preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                accuracy_object.update_state(y,preds)
        print('epoch: ',epoch+1,'loss: {:.4f}'.format(loss.numpy().item()),'accuracy: {:.4f}'.format(accuracy_object.result().numpy().item()))

        if pretraining:
            model.save_weights('classification_weights_low_fidelity/self_{}_{}.h5'.format(task, epoch+1))
        elif not pretraining:
            model.save_weights('classification_weights_low_fidelity/{}_{}.h5'.format(task, epoch+1))

        y_true = []
        y_preds = []

        for x, adjoin_matrix, y in test_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)

        test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
        print('test auc:{:.4f}'.format(auc_new), 'test accuracy:{:.4f}'.format(test_accuracy))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
        else:
            stopping_monitor += 1
        print('best test auc: {:.4f}'.format(auc))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor>10:
            break

    return auc

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Two-Step-Pretraining')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed (default: 7)')
    parser.add_argument('--pretraining', action='store_true', help="whether to load the weights of a trained model")
    parser.add_argument('--trained_epoch', type=int, default=100,
                        help='which epoch of the pretrained model to use (default: 100)')
    parser.add_argument('--task', type=str, default='x_bbbp',
                        help='the pseudo-label used in graph-level pretraining (default: x_bbbp)')
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='the maximum training epoch of the graph-level pretraining (default: 20)')

    args = parser.parse_args()

    seed = args.seed
    pretraining = args.pretraining # [True, False]
    trained_epoch = args.trained_epoch
    task = args.task  # tasks = ['a_bbbp', 'a_f20%', 'a_f30%', 'a_herg', 'x_bbbp', 'x_f20%', 'x_f30%', 'x_herg']

    max_epoch = args.max_epoch

    arch = medium3

    auc = main(seed=seed, arch = arch, pretraining = pretraining, trained_epoch = trained_epoch, task = task, max_epoch = max_epoch)
    print('best test auc:', auc)



