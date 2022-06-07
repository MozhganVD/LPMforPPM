import os
import pickle
import argparse
import numpy as np
import time
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Concatenate, Input
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from tensorflow import keras
from data_loader import DataManager
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
import pandas as pd


def lstm_layers_creating(params, previous_layer):
    if params['layers'] == 1:
        return LSTM(params['units'], dropout=params['rate'])(previous_layer)
    elif params['layers'] == 2:
        l1 = LSTM(params['units'], dropout=params['rate'], return_sequences=True)(previous_layer)
        return LSTM(params['units'], dropout=params['rate'])(l1)
    elif params['layers'] == 3:
        l1 = LSTM(params['units'], dropout=params['rate'], return_sequences=True)(previous_layer)
        l2 = LSTM(params['units'], dropout=params['rate'], return_sequences=True)(l1)
        return LSTM(params['units'], dropout=params['rate'])(l2)
    elif params['layers'] == 4:
        l1 = LSTM(params['units'], dropout=params['rate'], return_sequences=True)(previous_layer)
        l2 = LSTM(params['units'], dropout=params['rate'], return_sequences=True)(l1)
        l3 = LSTM(params['units'], dropout=params['rate'], return_sequences=True)(l2)
        return LSTM(params['units'], dropout=params['rate'])(l3)


def embedding_model(params):
    print("Initialising default model")
    if LPMs:
        if Only_LPMs:
            Input_lpm = Input(shape=max_length, name='inputlpm')
            embedding_lpm = Embedding(num_lpms, params['embedding_lpms'], input_length=max_length, name='lpms')(
                Input_lpm)
            lstm = lstm_layers_creating(params, embedding_lpm)
            flat = Flatten()(lstm)
            dense = Dense(1, activation='sigmoid')(flat)
            model = Model(inputs=Input_lpm,
                          outputs=dense)
        else:
            Input_act = Input(shape=max_length, name='inputact')
            Input_lpm = Input(shape=max_length, name='inputlpm')
            embedding_act = Embedding(num_acts, params['embedding_act'], input_length=max_length, name='act')(Input_act)
            embedding_lpm = Embedding(num_lpms, params['embedding_lpms'], input_length=max_length, name='lpms')(
                Input_lpm)
            concatenating = Concatenate(name='concatenated', axis=2)([embedding_act, embedding_lpm])
            lstm = lstm_layers_creating(params, concatenating)
            flat = Flatten()(lstm)
            dense = Dense(1, activation='sigmoid')(flat)
            model = Model(inputs=[Input_act, Input_lpm],
                          outputs=dense)
    else:
        Input_act = Input(shape=max_length, name='inputact')
        embedding_act = Embedding(num_acts, params['embedding_act'], input_length=max_length, name='act')(
            Input_act)
        lstm = lstm_layers_creating(params, embedding_act)
        flat = Flatten()(lstm)
        dense = Dense(1, activation='sigmoid')(flat)
        model = Model(inputs=Input_act,
                      outputs=dense)

    if params['opt'] == 'adam':
        opt = keras.optimizers.Adam(lr=params['learning_rate'])
    else:
        opt = keras.optimizers.RMSprop(lr=params['learning_rate'])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(),
                                                                      tf.keras.metrics.Recall(), 'acc'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='auto')
    if LPMs:
        if Only_LPMs:
            start_time = time.time()
            model.fit(train_lpms, train_y,
                      batch_size=params['batch_size'],
                      epochs=1000,
                      validation_split=0.2,
                      callbacks=early_stopping)
        else:
            start_time = time.time()
            model.fit([train_x, train_lpms], train_y,
                      batch_size=params['batch_size'],
                      epochs=1000,
                      validation_split=0.2,
                      callbacks=early_stopping)
    else:
        start_time = time.time()
        model.fit(train_x, train_y,
                  batch_size=params['batch_size'],
                  epochs=1000,
                  validation_split=0.2,
                  callbacks=early_stopping)

    training_time = time.time() - start_time

    return model, training_time


def accuracy_(TP, FP, FN, TN):
    return (TP + TN) / (TP + FP + FN + TN)


def Fscore_(TP, FP, FN):
    return TP / (TP + 0.5 * (FP + FN))


def Evaluation(Y, yhat_probs):
    yhat_classes = [int(np.round(a)) for a in yhat_probs]
    yhat_classes = np.array(yhat_classes)
    # all_kappa.append(cohen_kappa_score(Y, yhat_classes))
    cm = confusion_matrix(Y, yhat_classes)
    if len(cm) < 2:
        cm_0 = np.array([[0, 0], [0, 0]])
        if Y[0] == 1:
            cm_0[0][0] = cm[0][0]
        else:
            cm_0[1][1] = cm[0][0]

        cm = cm_0
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    W_F1score = f1_score(Y, yhat_classes, average='weighted')
    W_precision = precision_score(Y, yhat_classes, average='weighted')
    W_recall = recall_score(Y, yhat_classes, average='weighted')

    Acc = accuracy_(TP, FP, FN, TN)
    F1score = Fscore_(TP, FP, FN)

    return W_F1score, W_precision, W_recall, Acc, F1score


def final_evaluation(model, x_test, y_test, prefixes, lpm_test=None, only_lpm=False):
    # evaluate per prefix length
    all_acc, all_fscore, all_auc = [], [], []
    all_W_fscore, all_W_recall, all_W_precision = [], [], []
    K = []
    size_pre = []

    for pre in np.unique(prefixes):
        indices = [i for i, x in enumerate(prefixes) if x <= pre]
        Y = np.array([y_test[y] for y in indices])
        if len(indices) < 2:
            continue
        if lpm_test is not None:
            if only_lpm:
                LPM = np.array([lpm_test[b] for b in indices])
                yhat_probs = model.predict(LPM, verbose=0)
            else:
                X = np.array([x_test[b] for b in indices])
                LPM = np.array([lpm_test[b] for b in indices])
                yhat_probs = model.predict([X, LPM], verbose=0)
        else:
            X = np.array([x_test[b] for b in indices])
            yhat_probs = model.predict(X, verbose=0)

        W_F1score, W_precision, W_recall, Acc, F1score = Evaluation(Y, yhat_probs)
        all_acc.append(Acc)
        all_fscore.append(F1score)
        yhat_probs = yhat_probs[:, 0]
        if len(np.unique(Y)) < 2:
            all_auc.append(-1)
        else:
            all_auc.append(roc_auc_score(Y, yhat_probs))
        K.append(pre)
        size_pre.append(len(indices))
        all_W_fscore.append(W_F1score)
        all_W_recall.append(W_recall)
        all_W_precision.append(W_precision)

    results_frame = pd.DataFrame({'prefix': K, 'size': size_pre, 'acc': all_acc, 'fscore': all_fscore, 'AUC': all_auc,
                                  'w_fscore': all_W_fscore, 'recall': all_W_recall, 'precision': all_W_precision})

    # evaluate on all test set
    if lpm_test is not None:
        if only_lpm:
            yhat_probs = model.predict(lpm_test, verbose=0)
        else:
            yhat_probs = model.predict([x_test, lpm_test], verbose=0)
    else:
        yhat_probs = model.predict(x_test, verbose=0)

    yhat_classes = [int(np.round(a)) for a in yhat_probs]
    yhat_classes = np.array(yhat_classes)
    yhat_probs = yhat_probs[:, 0]
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    cm = confusion_matrix(y_test, yhat_classes)
    report = classification_report(y_test, yhat_classes, digits=5)
    print(report)
    print(cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    Acc = accuracy_(TP, FP, FN, TN)
    Fscore = Fscore_(TP, FP, FN)

    W_F1score = f1_score(y_test, yhat_classes, average='weighted')
    W_precision = precision_score(y_test, yhat_classes, average='weighted')
    W_recall = recall_score(y_test, yhat_classes, average='weighted')

    print('Accuracy: %f' % Acc)
    print('Fscore: %f' % Fscore)

    return Acc, Fscore, auc, kappa, cm, results_frame, W_recall, W_precision, W_F1score, report


parser = argparse.ArgumentParser(description="Embedding Model")
parser.add_argument("--data_dir",
                    type=str,
                    default="./datasets",
                    help="dataset directory")

parser.add_argument("--raw_data",
                    type=str,
                    default="./datasets/BPIC11_f4_trunc40_completeLPMs_Agg.csv",
                    help="address to raw dataset")

parser.add_argument("--out_name",
                    type=str,
                    default="BPIC12_cancelled_N",
                    help="to store all data and results")

parser.add_argument("--LPMs",
                    type=bool,
                    default=False)

parser.add_argument("--Only_LPMs",
                    type=bool,
                    default=False)

parser.add_argument("--sample",
                    type=bool,
                    default=False)

parser.add_argument("--max_length",
                    type=int,
                    default=40)

parser.add_argument("--batch_size",
                    type=int,
                    default=8)

parser.add_argument("--embedding_act",
                    type=int,
                    default=29)

parser.add_argument("--embedding_lpms",
                    type=int,
                    default=3)

parser.add_argument("--learning_rate",
                    type=float,
                    default=8.366706242039303e-05)

parser.add_argument("--opt",
                    type=str,
                    default='RMSprop',
                    help="adam or RMSprop")

parser.add_argument("--rate",
                    type=float,
                    default=0.2442770683880605)

parser.add_argument("--units",
                    type=int,
                    default=95)

parser.add_argument("--layers",
                    type=int,
                    default=1)

args = parser.parse_args()
LPMs = args.LPMs
Only_LPMs = args.Only_LPMs

dataset_name = args.out_name
datasets_dir = args.data_dir
output_datasets_address = f"{datasets_dir}/{dataset_name}"
if not os.path.exists(f"{datasets_dir}/{dataset_name}"):
    os.makedirs(f"{datasets_dir}/{dataset_name}")
    dataset_address = args.raw_data
    data_manager = DataManager(dataset_address, 2, args.max_length)
    train_x, train_y, train_lpms, test_x, test_y, test_lpms = data_manager.encode_prefixes(LPMs, 0.8,
                                                                                           sample=args.sample)
    max_length = data_manager.max_length
    num_acts = data_manager.num_acts
    num_lpms = data_manager.num_lpms

    with open(output_datasets_address + '/num_lpms.pkl', 'wb') as handle:
        pickle.dump(num_lpms, handle)

    with open(output_datasets_address + '/train_lpms.pkl', 'wb') as handle:
        pickle.dump(train_lpms, handle)

    with open(output_datasets_address + '/test_lpms.pkl', 'wb') as handle:
        pickle.dump(test_lpms, handle)

    with open(output_datasets_address + '/train_x.pkl', 'wb') as handle:
        pickle.dump(train_x, handle)

    with open(output_datasets_address + '/train_y.pkl', 'wb') as handle:
        pickle.dump(train_y, handle)

    with open(output_datasets_address + '/test_x.pkl', 'wb') as handle:
        pickle.dump(test_x, handle)

    with open(output_datasets_address + '/test_y.pkl', 'wb') as handle:
        pickle.dump(test_y, handle)

    with open(output_datasets_address + '/max_length.pkl', 'wb') as handle:
        pickle.dump(max_length, handle)

    with open(output_datasets_address + '/num_acts.pkl', 'wb') as handle:
        pickle.dump(num_acts, handle)

else:
    with open(output_datasets_address + '/num_lpms.pkl', 'rb') as handle:
        num_lpms = pickle.load(handle)

    with open(output_datasets_address + '/train_lpms.pkl', 'rb') as handle:
        train_lpms = pickle.load(handle)

    with open(output_datasets_address + '/test_lpms.pkl', 'rb') as handle:
        test_lpms = pickle.load(handle)

    with open(output_datasets_address + '/train_x.pkl', 'rb') as handle:
        train_x = pickle.load(handle)

    with open(output_datasets_address + '/train_y.pkl', 'rb') as handle:
        train_y = pickle.load(handle)

    with open(output_datasets_address + '/test_x.pkl', 'rb') as handle:
        test_x = pickle.load(handle)

    with open(output_datasets_address + '/test_y.pkl', 'rb') as handle:
        test_y = pickle.load(handle)

    with open(output_datasets_address + '/max_length.pkl', 'rb') as handle:
        max_length = pickle.load(handle)

    with open(output_datasets_address + '/num_acts.pkl', 'rb') as handle:
        num_acts = pickle.load(handle)

output_file = output_datasets_address + "/Running_Results_LPMs%s_Only%s.txt" % (LPMs, Only_LPMs)
if LPMs:
    if Only_LPMs:
        space = {'batch_size': args.batch_size, 'embedding_lpms': args.embedding_lpms,
                 'learning_rate': args.learning_rate, 'opt': args.opt, 'rate': args.rate,
                 'units': args.units, 'layers': args.layers}
    else:
        space = {'batch_size': args.batch_size, 'embedding_act': args.embedding_act,
                 'embedding_lpms': args.embedding_lpms, 'learning_rate': args.learning_rate,
                 'opt': args.opt, 'rate': args.rate, 'units': args.units, 'layers': args.layers}
else:
    space = {'batch_size': args.batch_size, 'embedding_act': args.embedding_act,
             'learning_rate': args.learning_rate, 'opt': args.opt, 'rate': args.rate,
             'units': args.units, 'layers': args.layers}

# model training
prefixes = []
for test in test_x:
    prefixes.append(np.count_nonzero(test))
training_numbers = 1
outfile = open(output_file, 'w')
Accuracy_l = []
Fscore_l = []
AUC_l = []
Kappa_l = []
# for i in range(training_numbers):
# print('Starting model training % s...' % i)

best_model, training_time = embedding_model(space)
print(training_time)

outfile.write('\nTraining time: %.4f' % training_time)

if LPMs:
    if Only_LPMs:
        Acc, Fscore, auc, kappa, cm, results_frame, \
        W_recall, W_precision, W_F1score, report = final_evaluation(best_model, test_x, test_y,
                                                                    prefixes, lpm_test=test_lpms, only_lpm=Only_LPMs)
    else:
        Acc, Fscore, auc, kappa, cm, results_frame, \
        W_recall, W_precision, W_F1score, report = final_evaluation(best_model, test_x, test_y,
                                                                    prefixes, lpm_test=test_lpms)
else:
    Acc, Fscore, auc, kappa, cm, results_frame, \
    W_recall, W_precision, W_F1score, report = final_evaluation(best_model, test_x, test_y, prefixes)

outfile.write("\n")
outfile.write(str(report))
outfile.write("\n")
outfile.write('\n*******************************')
outfile.write('\nTesting Accuracy: %.4f' % Acc)
outfile.write('\nTesting F-Score: %.4f' % Fscore)
outfile.write('\nTesting AUC: %.4f' % auc)
outfile.write('\nTesting Kappa: %.4f' % kappa)
outfile.write('\nTesting Confusion matrix:\n %s' % str(cm))
outfile.write("\nWeighted f1 score: %.4f" % W_F1score)
outfile.write("\nRecall: %.4f" % W_recall)
outfile.write("\nPrecision: %.4f" % W_precision)
outfile.write('\n*******************************')

results_frame.to_csv(output_datasets_address + "/PrefixesResults_lpm%s_only%s.csv" % (LPMs, Only_LPMs), index=False)

outfile.close()

best_model.save(output_datasets_address + "/model_lpm%s_only%s" % (LPMs, Only_LPMs))

# creating dataset for explain ability
# train data
if LPMs and not Only_LPMs:
    yhat_probs_train = best_model.predict([train_x, train_lpms], verbose=0)
    yhat_classes_train = [int(np.round(a)) for a in yhat_probs_train]
    yhat_classes_train = np.array(yhat_classes_train)

    LPM_train_set = pd.DataFrame(train_lpms)
    LPM_train_set.columns = ["loc_patrn_%s" % i for i in range(len(LPM_train_set.columns))]
    LPM_train_set['predicted_label'] = yhat_classes_train
    LPM_train_set['init_label'] = train_y
    LPM_train_set.to_csv(output_datasets_address +"/%s_patterns_train.csv" % args.out_name, index=False)

    # test data
    yhat_probs_test = best_model.predict([test_x, test_lpms], verbose=0)
    yhat_classes_test = [int(np.round(a)) for a in yhat_probs_test]
    yhat_classes_test = np.array(yhat_classes_test)

    LPM_test_set = pd.DataFrame(test_lpms)
    LPM_test_set.columns = ["loc_patrn_%s" % i for i in range(len(LPM_test_set.columns))]
    LPM_test_set['predicted_label'] = yhat_classes_test
    LPM_test_set['init_label'] = test_y
    LPM_test_set.to_csv(output_datasets_address +"/%s_patterns_test.csv" % args.out_name, index=False)

elif not LPMs:
    # train set
    yhat_probs_train = best_model.predict(train_x, verbose=0)
    yhat_classes_train = [int(np.round(a)) for a in yhat_probs_train]
    yhat_classes_train = np.array(yhat_classes_train)
    new_data_frame = pd.DataFrame(train_x)

    new_data_frame.columns = ["loc_act_%s" % i for i in range(len(new_data_frame.columns))]
    new_data_frame['predicted_label'] = yhat_classes_train
    new_data_frame['init_label'] = train_y
    new_data_frame.to_csv(output_datasets_address +"/%s_acts_train.csv" % args.out_name, index=False)

    # test set
    yhat_probs_test = best_model.predict(test_x, verbose=0)
    yhat_classes_test = [int(np.round(a)) for a in yhat_probs_test]
    yhat_classes_test = np.array(yhat_classes_test)

    test_frame = pd.DataFrame(test_x)
    test_frame.columns = ["loc_act_%s" % i for i in range(len(test_frame.columns))]
    test_frame['predicted_label'] = yhat_classes_test
    test_frame['init_label'] = test_y
    test_frame.to_csv(output_datasets_address +"/%s_acts_test.csv" % args.out_name, index=False)

print("done")