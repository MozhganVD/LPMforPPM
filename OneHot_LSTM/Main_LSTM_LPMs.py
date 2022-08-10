import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import config
from feature_encoder import FeatureEncoder
import time
from utils.Writing_methods import write_results_to_text, evaluate_each_prefix, evaluate
import pandas as pd


def LSTM_model(params):
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
    # model.compile(optimizer='adam', loss='mean_squared_error')
    if params['opt'] == 'adam':
        opt = keras.optimizers.Adam(lr=params['learning_rate'])
    else:
        opt = keras.optimizers.RMSprop(lr=params['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=[tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(), 'acc'])

    return model


args = config.load()

if not os.path.exists(f"{args.checkpoint_dir}/{args.dataset}"):
    os.makedirs(f"{args.checkpoint_dir}/{args.dataset}")

output_address = f"{args.checkpoint_dir}/{args.dataset}/"

if not os.path.exists(f"{args.data_dir}/{args.dataset}"):
    os.makedirs(f"{args.data_dir}/{args.dataset}")
output_datasets_address = f"{args.data_dir}/{args.dataset}"

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

print("Training model ...")

params = {'batch_size': args.batch_size, 'layers': args.layers, 'learning_rate': args.learning_rate,
          'opt': args.opt, 'rate': args.rate, 'units': args.units}

model = LSTM_model(params)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                               mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
# We iterate the learning process
start_time = time.time()
model.fit(train_X, train_y, batch_size=params['batch_size'], epochs=args.num_epochs, verbose=1,
          validation_split=0.2, callbacks=[early_stopping, lr_reducer])

Running_Time = time.time() - start_time

model.save(output_datasets_address + "/model_Encoding%s_LPMs%s_%s" % (args.encoding_type,
                                                                      args.LPMs,
                                                                      args.LPMs_type))
print('LSTM training time is %.3f S' % Running_Time)

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
CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Cohen_kappa, auc = evaluate(model, test_X, test_y)
results_file = output_address + "Results_ConfMat_LSTM_Encoding%s_LPMs%s_%s.txt" % (args.encoding_type,
                                                                                   args.LPMs,
                                                                                   args.LPMs_type)
test_prefixes = test_df["k"]
write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Running_Time, Cohen_kappa, auc,
                      results_file)
evaluate_each_prefix(model, test_X, test_y, test_prefixes, output_address,
                     "results_prefixes_LSTM_Encoding%s_LPMs%s_%s" % (args.encoding_type,
                                                                     args.LPMs,
                                                                     args.LPMs_type))
print("done")
