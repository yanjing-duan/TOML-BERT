import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np

from dataset import Graph_Regression_Tune_Parameter_Dataset
from sklearn.metrics import r2_score,roc_auc_score

import os
from model import PredictModel,BertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights', 'addH': True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
medium2 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2',
           'addH': True}
medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3',
           'addH': True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights', 'addH': True}
medium_without_H = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H',
                    'addH': False}
medium_without_pretrain = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256,
                           'path': 'medium_without_pretraining_weights', 'addH': True}

def main(seed=7, arch = medium3, pretraining = 'NO', trained_epoch = 8, low_fidelity_task = 'cx_logD', task = 'LogD', max_epoch = 100, dropout_rate = 0.1, batch_size = 64, learning_rate=10e-5, dense_dropout=0.15):

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    # dropout_rate = 0.1 

    tf.random.set_seed(seed=seed)
    graph_dataset = Graph_Regression_Tune_Parameter_Dataset('data/reg/{}.csv'.format(task), smiles_field='smiles', label_field=task, addH=addH)
    train_dataset, test_dataset,val_dataset = graph_dataset.get_data(batch_size)
    value_range = graph_dataset.value_range

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size, dropout_rate = dropout_rate, dense_dropout=dense_dropout)

    if pretraining == "self_supervised_pretraining":
        print("task: {}, self-supervise_pre-trained".format(task))
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_self_supervised_pretraining_weights')

    elif pretraining == "low_fidelity_pretraining":
        print("task: {}, low_fidelity_task: {}".format(task, low_fidelity_task))
        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        model.load_weights('regression_weights_low_fidelity/{}_{}.h5'.format(low_fidelity_task, trained_epoch))
        print('load_low_fidelity_pretraining_weights')

    elif pretraining == "self-low":
        print("task: {}, low_fidelity_task: {}, self-supervise_pre-trained".format(task, low_fidelity_task))
        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        model.load_weights('regression_weights_low_fidelity/self_{}_{}.h5'.format(low_fidelity_task, trained_epoch))
        print('load_self+low_pretraining_weights')

    elif pretraining == "NO":
        print("No pre-training")

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, total_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.total_step = total_steps
            self.warmup_steps = total_steps*0.10

        def __call__(self, step):
            arg1 = step/self.warmup_steps
            arg2 = 1-(step-self.warmup_steps)/(self.total_step-self.warmup_steps)

            return 10e-5* tf.math.minimum(arg1, arg2)

    # steps_per_epoch = len([_ for _ in iter(train_dataset)])
    # learning_rate = CustomSchedule(128,100*steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    r2 = -10
    stopping_monitor = 0
    for epoch in range(max_epoch):
        mse_object = tf.keras.metrics.MeanSquaredError()
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = tf.reduce_mean(tf.square(y-preds))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mse_object.update_state(y,preds)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'mse: {:.4f}'.format(mse_object.result().numpy().item() * (value_range**2)))

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
        r2_new = r2_score(y_true,y_preds)

        val_mse = keras.metrics.MSE(y_true, y_preds).numpy() * (value_range**2)
        print('val r2: {:.4f}'.format(r2_new), 'val mse:{:.4f}'.format(val_mse))
        if r2_new > r2:
            r2 = r2_new
            stopping_monitor = 0

            if pretraining == "self_supervised_pretraining":
                model.save_weights('regression_weights/self_{}.h5'.format(task))
            elif pretraining == "low_fidelity_pretraining":
                model.save_weights('regression_weights/low_{}_{}.h5'.format(low_fidelity_task, task))
            elif pretraining == "self-low":
                model.save_weights('regression_weights/self-low_{}_{}.h5'.format(low_fidelity_task, task))
            elif pretraining == "NO":
                model.save_weights('regression_weights/{}.h5'.format(task))
        else:
            stopping_monitor +=1
        print('best r2: {:.4f}'.format(r2))

        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor==20:
            break

    y_true = []
    y_preds = []

    if pretraining == "self_supervised_pretraining":
        model.load_weights('regression_weights/self_{}.h5'.format(task))
    elif pretraining == "low_fidelity_pretraining":
        model.load_weights('regression_weights/low_{}_{}.h5'.format(low_fidelity_task, task))
    elif pretraining == "self-low":
        model.load_weights('regression_weights/self-low_{}_{}.h5'.format(low_fidelity_task, task))
    elif pretraining == "NO":
        model.load_weights('regression_weights/{}.h5'.format(task))

    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)

    test_r2 = r2_score(y_true, y_preds)
    test_mse = keras.metrics.MSE(y_true.reshape(-1), y_preds.reshape(-1)).numpy() * (value_range**2)
    test_mae = keras.metrics.MAE(y_true.reshape(-1), y_preds.reshape(-1)).numpy() * (value_range)
    print('test r2:{:.4f}'.format(test_r2), 'test mse:{:.4f}'.format(test_mse), 'test mae:{:.4f}'.format(test_mae))

    return test_r2, test_mse, test_mae

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Two-Step-Pretraining')

    parser.add_argument('--pretraining', type= str, default='NO', help='Whether to load the weights of a trained model and which model to load. Choose one from ["self_supervised_pretraining", "low_fidelity_pretraining", "self-low", "NO"], corresponding to the node-level, graph-level, and dual-level pretrained models, as well as no pretraining.')
    parser.add_argument('--trained_epoch', type=int, default=100,
                        help='which epoch of the pretrained model to use (default: 100)')
    parser.add_argument('--low_fidelity_task', type=str, default='cx_logD',
                        help='the pseudo-label used in graph-level pretraining (default: cx_logD)')
    parser.add_argument('--task', type=str, default='LogD',
                        help='the fine-tuning task (default: LogD)')
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
    test_r2_list = []
    test_mse_list = []
    test_mae_list = []

    pretraining = args.pretraining # pretraining：["self_supervised_pretraining", "low_fidelity_pretraining", 'self-low', 'NO']
    trained_epoch = args.trained_epoch
    low_fidelity_task = args.low_fidelity_task # ['cx_logD', 'cx_logP', 'm_hlogD', 'm_hlogP', 'm_hlogS', 'm_logP', 'm_logS', 'm_slogP', 'x_caco2', 'x_cilogS', 'x_logP', 'x_logS', 'x_mdck', 'a_caco2', 'a_logD', 'a_logP', 'a_logS', 'a_mdck']
    task = args.task # task : ['Caco-2', 'LogD', 'logD_19155', 'LogP', 'LogS', 'MDCK']
    max_epoch = args.max_epoch
    dropout_rate = args.dropout_rate # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    batch_size = args.batch_size # [16,32,64]
    learning_rate = args.learning_rate # [1e-5, 5e-5, 1e-4]
    dense_dropout = args.dense_dropout # default 0.15；[0.0,0.1,0.2]

    arch = medium3
    for seed in [7,17,27,37,47,57,67,77,87,97]:
        print("seed:",seed)
        test_r2, test_mse, test_mae = main(seed=seed, arch=arch, pretraining=pretraining, trained_epoch=trained_epoch, low_fidelity_task=low_fidelity_task, task=task, max_epoch=max_epoch, dropout_rate=dropout_rate, batch_size=batch_size, learning_rate=learning_rate, dense_dropout=dense_dropout)

        test_r2_list.append(test_r2)
        test_mse_list.append(test_mse)
        test_mae_list.append(test_mae)

    results["test_r2"] = test_r2_list
    results["test_mse"] = test_mse_list
    results["test_mae"] = test_mae_list

    print("task:", task)
    print("dropout_rate:", dropout_rate)
    print("batch_size:", batch_size)
    print("learning_rate:", learning_rate)
    print("dense_dropout:", dense_dropout)
    print(results)

    if pretraining == "self_supervised_pretraining":
        results.to_csv('results/self_{}.csv'.format(task), index=False)
    elif pretraining == "low_fidelity_pretraining":
        results.to_csv('results/low_{}_{}.csv'.format(low_fidelity_task, task), index=False)
    elif pretraining == "self-low":
        results.to_csv('results/self-low_{}_{}.csv'.format(low_fidelity_task, task), index=False)
    elif pretraining == "NO":
        results.to_csv('results/{}.csv'.format(task), index=False)

