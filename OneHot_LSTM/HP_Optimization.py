import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from tqdm.keras import TqdmCallback
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope  # quniform returns float, some parameters require int; use this to force int
import os
import config
from feature_encoder import FeatureEncoder
import time
from utils.Writing_methods import write_results_to_text, evaluate_each_prefix, evaluate
import pandas as pd


def f_lstm_cv(params):
    # Keras LSTM model
    model = Sequential()

    if params['layers'] == 1:
        model.add(LSTM(units=params['units'], input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(rate=params['rate']))
    else:
        # First layer specifies input_shape and returns sequences
        model.add(LSTM(units=params['units'], return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(rate=params['rate']))
        # Middle layers return sequences
        for i in range(params['layers'] - 2):
            model.add(LSTM(units=params['units'], return_sequences=True))
            model.add(Dropout(rate=params['rate']))
        # Last layer doesn't return anything
        model.add(LSTM(units=params['units']))
        model.add(Dropout(rate=params['rate']))

    model.add(Dense(1, activation='sigmoid'))
    if params['opt'] == 'adam':
        opt = keras.optimizers.Adam(lr=params['learning_rate'])
    else:
        opt = keras.optimizers.RMSprop(lr=params['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=[tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(), 'acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=0,
                                   mode='min', min_delta=0.0001, cooldown=0, min_lr=0)

    start_time = time.time()
    result = model.fit(train_X, train_y, verbose=0, validation_split=0.2,
                       batch_size=params['batch_size'],
                       epochs=200,
                       callbacks=[es, TqdmCallback(verbose=1), lr_reducer]
                       )
    training_time = time.time() - start_time
    # get the lowest validation loss of the training epochs
    validation_loss = np.amin(result.history['val_loss'])
    # print('Best validation loss of epoch:', validation_loss)
    global best_score, best_time, best_numparameters

    if best_score > validation_loss:
        best_score = validation_loss
        best_numparameters = model.count_params()
        best_time = training_time

    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model, 'params': params, 'time': training_time}


seed = 123
random_state = np.random.seed(seed)
args = config.load()

if not os.path.exists(f"{args.checkpoint_dir}/{args.dataset}"):
    os.makedirs(f"{args.checkpoint_dir}/{args.dataset}")

output_address = f"{args.checkpoint_dir}/{args.dataset}/"

if not os.path.exists(f"{args.data_dir}/{args.dataset}"):
    os.makedirs(f"{args.data_dir}/{args.dataset}")
output_datasets_address = f"{args.data_dir}/{args.dataset}"

model_name = '%s-%s_%s' % (
    args.dataset, args.num_epochs, args.batch_size)
output_file = output_address + args.dataset + "%s.log" % args.LPMs_type

with open(output_datasets_address + '/coded_activity.pkl', 'rb') as handle:
    coded_activity = pickle.load(handle)
with open(output_datasets_address + '/coded_labels.pkl', 'rb') as handle:
    coded_labels = pickle.load(handle)

with open(output_datasets_address + '/coded_lpms.pkl', 'rb') as handle:
    coded_lpms = pickle.load(handle)

with open(output_datasets_address + '/Max_prefix_length.pkl', 'rb') as handle:
    Max_prefix_length = pickle.load(handle)

Train_path = output_datasets_address + "/outcome_train.csv"
train_df = pd.read_csv(filepath_or_buffer=Train_path, header=0, sep=',')
print("Encoding training features ...")
fe = FeatureEncoder()
if args.encoding_type == 'W':
    train_X, train_y = fe.one_hot_encoding_Wrapped(train_df, coded_activity, coded_labels, Max_prefix_length,
                                                   Lpms_type=args.LPMs_type)
elif args.encoding_type == 'C':
    train_X, train_y = fe.one_hot_encoding_Classic(train_df, coded_activity, coded_labels, coded_lpms,
                                                   Max_prefix_length, LPMs=args.LPMs)
else:
    print("encoding type: %s is not supported! choose between C and W " % args.encoding_type)

print("done")

space = {'rate': hp.uniform('rate', 0.01, 0.3),
         'units': scope.int(hp.quniform('units', 10, 100, 5)),
         'batch_size': hp.choice('batch_size', [8, 16, 32, 64, 132]),
         'layers': scope.int(hp.quniform('layers', 1, 4, 1)),
         'opt': hp.choice('opt', ['adam', 'RMSprop']),
         'learning_rate': hp.uniform('learning_rate', 0.00001, 0.0001)
         }
# model selection
print('Starting model selection...')
start_selection_time = time.time()
best_score = np.inf
best_time = 0
best_numparameters = 0

trials = Trials()
best = fmin(f_lstm_cv, space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=np.random.RandomState(seed))
best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
selection_time = time.time() - start_selection_time
print(selection_time)
print(best_score)
print(best_params)
print(best_time)
outfile = open(output_file, 'w')
outfile.write("\n\nBest parameters:\n")
outfile.write(str(best_params))
outfile.write("\nBest Model parameters number: %d" % best_numparameters)
outfile.write('\nBest Time taken: %.4f' % best_time)
outfile.write('\nBest validation loss: %.4f' % best_score)
outfile.write('\nHyperparameter Optimization time taken: %.4f' % selection_time)
outfile.close()

print("Encoding test features...")
Test_path = output_datasets_address + "/outcome_test.csv"
test_df = pd.read_csv(filepath_or_buffer=Test_path, header=0, sep=',')
if args.encoding_type == 'W':
    test_X, test_y = fe.one_hot_encoding_Wrapped(test_df, coded_activity, coded_labels, Max_prefix_length,
                                                      Lpms_type=args.LPMs_type)
elif args.encoding_type == 'C':
    test_X, test_y = fe.one_hot_encoding_Classic(test_df, coded_activity, coded_labels, coded_lpms,
                                                 Max_prefix_length, LPMs=args.LPMs)
else:
    print("encoding type: %s is not supported! choose between C and W " % args.encoding_type)

print("done")
# model.load(args.checkpoint_dir, model_name=model_name)
print("Evaluating ...")
CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Cohen_kappa, AUC = evaluate(best_model, test_X, test_y)
results_file = output_address + "Results_ConfMat_LSTM_Encoding%s_LPMs%s_%s.txt" % (args.encoding_type,
                                                                                   args.LPMs,
                                                                                   args.LPMs_type)
test_prefixes = test_df["k"]
write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall,
                      best_time, Cohen_kappa, AUC, results_file)

evaluate_each_prefix(best_model, test_X, test_y, test_prefixes, output_address,
                     "results_prefixes_LSTM_Encoding%s_LPMs%s_%s" % (args.encoding_type,
                                                                     args.LPMs,
                                                                     args.LPMs_type))

best_model.save(output_datasets_address + "/BestModel_Encoding%s_LPMs%s_%s" % (args.encoding_type,
                                                                                   args.LPMs,
                                                                                   args.LPMs_type))
print("done")
