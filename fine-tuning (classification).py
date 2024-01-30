import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from dataset import Graph_Classification_Tune_Parameter_Dataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score, f1_score

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


def main(seed=7, arch = medium3, pretraining = 'NO', trained_epoch = 8, low_fidelity_task = 'x_bbbp', task = 'BBBP', max_epoch = 100, dropout_rate = 0.1, batch_size = 64, learning_rate=10e-5, dense_dropout=0.15):

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    # dropout_rate = 0.1

    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset , val_dataset = Graph_Classification_Tune_Parameter_Dataset('data/clf/{}.csv'.format(task), smiles_field='smiles', label_field='Label',addH=addH).get_data(batch_size)

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dropout_rate = dropout_rate, dense_dropout=dense_dropout)

    if pretraining == "self_supervised_pretraining":
        print("task: {}, self-supervise_pre-trained".format(task))
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_self_supervised_pretraining_weights')

    elif pretraining == "low_fidelity_pretraining":
        print("task: {}, low_fidelity_task: {}".format(task, low_fidelity_task))
        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        model.load_weights('classification_weights_low_fidelity/{}_{}.h5'.format(low_fidelity_task, trained_epoch))
        print('load_low_fidelity_pretraining_weights')

    elif pretraining == "self-low":
        print("task: {}, low_fidelity_task: {}, self-supervise_pre-trained".format(task, low_fidelity_task))
        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        model.load_weights('classification_weights_low_fidelity/self_{}_{}.h5'.format(low_fidelity_task, trained_epoch))
        print('load_self+low_pretraining_weights')

    elif pretraining == "NO":
        print("No pre-training")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'accuracy: {:.4f}'.format(accuracy_object.result().numpy().item()))

        y_true = []
        y_preds = []

        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)

        val_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_accuracy))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0

            if pretraining == "self_supervised_pretraining":
                model.save_weights('classification_weights/self_{}.h5'.format(task))
            elif pretraining == "low_fidelity_pretraining":
                model.save_weights('classification_weights/low_{}_{}.h5'.format(low_fidelity_task, task))
            elif pretraining == "self-low":
                model.save_weights('classification_weights/self-low_{}_{}.h5'.format(low_fidelity_task, task))
            elif pretraining == "NO":
                model.save_weights('classification_weights/{}.h5'.format(task))
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor==20:
            break

    y_true = []
    y_preds = []

    if pretraining == "self_supervised_pretraining":
        model.load_weights('classification_weights/self_{}.h5'.format(task))
    elif pretraining == "low_fidelity_pretraining":
        model.load_weights('classification_weights/low_{}_{}.h5'.format(low_fidelity_task, task))
    elif pretraining == "self-low":
        model.load_weights('classification_weights/self-low_{}_{}.h5'.format(low_fidelity_task, task))
    elif pretraining == "NO":
        model.load_weights('classification_weights/{}.h5'.format(task))

    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    y_preds = tf.sigmoid(y_preds).numpy()
    y_preds_label = y_preds.reshape(-1) > 0.5
    y_preds_label = y_preds_label.astype(int)

    test_auc = roc_auc_score(y_true, y_preds)
    test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    test_mcc = matthews_corrcoef(y_true.reshape(-1), y_preds_label)
    test_ba = balanced_accuracy_score(y_true.reshape(-1), y_preds_label)
    test_f1 = f1_score(y_true.reshape(-1), y_preds_label)

    print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_accuracy), 'test mcc:{:.4f}'.format(test_mcc))

    return test_auc, test_accuracy, test_mcc, test_ba, test_f1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Two-Step-Pretraining')

    parser.add_argument('--pretraining', type= str, default='NO', help='Whether to load the weights of a trained model and which model to load. Choose one from ["self_supervised_pretraining", "low_fidelity_pretraining", "self-low", "NO"], corresponding to the node-level, graph-level, and dual-level pretrained models, as well as no pretraining.')
    parser.add_argument('--trained_epoch', type=int, default=100,
                        help='which epoch of the pretrained model to use (default: 100)')
    parser.add_argument('--low_fidelity_task', type=str, default='x_bbbp',
                        help='the pseudo-label used in graph-level pretraining (default: x_bbbp)')
    parser.add_argument('--task', type=str, default='BBBP',
                        help='the fine-tuning task (default: BBBP)')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='the maximum training epoch of the fine-tuning (default: 100)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size of the training set (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=10e-5,
                        help='learning rate (default: 10e-5)')
    parser.add_argument('--dense_dropout', type=float, default=0.15,
                        help='dense dropout (default: 0.15)')

    args = parser.parse_args()

    results = pd.DataFrame()
    test_auc_list = []
    test_acc_list = []
    test_mcc_list = []
    test_ba_list = []
    test_f1_list = []

    pretraining = args.pretraining # pretraining：["self_supervised_pretraining", "low_fidelity_pretraining", 'self-low', 'NO']
    trained_epoch = args.trained_epoch
    low_fidelity_task = args.low_fidelity_task # ['a_bbbp', 'a_f20%', 'a_f30%', 'a_herg', 'x_bbbp', 'x_f20%', 'x_f30%', 'x_herg']
    task = args.task # task : ['BBBP', 'F20%', 'F30%', 'hERG']
    max_epoch = args.max_epoch
    dropout_rate = args.dropout_rate # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    batch_size = args.batch_size # [16,32,64]
    learning_rate = args.learning_rate # [1e-5, 5e-5, 1e-4]
    dense_dropout = args.dense_dropout # default 0.15；[0.0,0.1,0.2]

    arch = medium3
    for seed in [7,17,27,37,47,57,67,77,87,97]:
        print("seed:",seed)
        test_auc, test_accuracy, test_mcc, test_ba, test_f1 = main(seed=seed, arch=arch, pretraining=pretraining, trained_epoch=trained_epoch, low_fidelity_task=low_fidelity_task, task=task, max_epoch=max_epoch, dropout_rate=dropout_rate, batch_size=batch_size, learning_rate=learning_rate, dense_dropout=dense_dropout)

        test_auc_list.append(test_auc)
        test_acc_list.append(test_accuracy)
        test_mcc_list.append(test_mcc)
        test_ba_list.append(test_ba)
        test_f1_list.append(test_f1)

    results["test_auc"] = test_auc_list
    results["test_acc"] = test_acc_list
    results["test_mcc"] = test_mcc_list
    results["test_ba"] = test_ba_list
    results["test_f1"] = test_f1_list

    print("task:", task)
    print("dropout_rate:", dropout_rate)
    print("batch_size:", batch_size)
    print("learning_rate:", learning_rate)
    print("dense_dropout:", dense_dropout)
    print(results)

    if pretraining == "self_supervised_pretraining":
        results.to_csv('classification_results/self_{}.csv'.format(task), index=False)
    elif pretraining == "low_fidelity_pretraining":
        results.to_csv('classification_results/low_{}_{}.csv'.format(low_fidelity_task, task), index=False)
    elif pretraining == "self-low":
        results.to_csv('classification_results/self-low_{}_{}.csv'.format(low_fidelity_task, task), index=False)
    elif pretraining == "NO":
        results.to_csv('classification_results/{}.csv'.format(task), index=False)