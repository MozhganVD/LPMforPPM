import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Concatenate, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle

plt.style.use('fivethirtyeight')
import math


class DataManager:
    def __init__(self, address, min_length, max_length):
        self.case_id = 'case:concept:name'
        self.activity = 'concept:name'
        self.outcome = 'label'
        self.lpms = 'LPMs_list'
        self.address = address
        self.data = pd.read_csv(self.address, sep=';')
        self.min_length = min_length
        self.max_length = max_length

    def get_act_dict(self):
        dictionary_acts = dict()
        activities = self.data[self.activity].unique()
        self.num_acts = len(activities)
        for idx, act in enumerate(activities):
            dictionary_acts[act] = idx

        return dictionary_acts

    def get_label_dict(self):
        dictionary_labels = dict()
        labels = self.data[self.outcome].unique()
        for idx, l in enumerate(labels):
            dictionary_labels[l] = idx

        return dictionary_labels

    def get_lpms_dict_str(self):
        dictionary_lpms = dict()
        lpms = self.data[self.lpms].unique()
        self.num_lpms = len(lpms)
        for idx, l in enumerate(lpms):
            dictionary_lpms[l] = "pattern_" + str(idx)

        return dictionary_lpms

    def get_lpms_dict(self):
        dictionary_lpms = dict()
        lpms = self.data[self.lpms].unique()
        self.num_lpms = len(lpms)
        for idx, l in enumerate(lpms):
            dictionary_lpms[l] = idx

        return dictionary_lpms

    def split_test_train(self, train_ratio, sample):
        Cases = []
        Prefixes = []
        Labels = []
        df = self.data
        case_id_col = self.case_id
        for case in df[case_id_col].unique():
            Cases.append(case)
            Prefixes.append(min(self.max_length, len(df[df[case_id_col] == case])))
            Labels.append(list(df[df[case_id_col] == case][self.outcome].unique())[0])

        Case_indexes = pd.DataFrame({case_id_col: Cases, "prefix length": Prefixes, self.outcome: Labels})
        median_prefix = np.median(list(Case_indexes["prefix length"]))
        first_half = Case_indexes[Case_indexes["prefix length"] < median_prefix]
        second_half = Case_indexes[Case_indexes["prefix length"] > median_prefix]
        quartile_1 = np.median(list(first_half["prefix length"]))
        quartile_2 = np.median(list(second_half["prefix length"]))
        # Case_indexes["prefix group"] = pd.cut(Case_indexes["prefix length"],
        #                                       bins=[0, quartile_1, median_prefix, quartile_2],
        #                                       include_lowest=True, labels=[quartile_1, median_prefix, quartile_2])
        #
        # train, test = train_test_split(Case_indexes, train_size=train_ratio, shuffle= True,
        #                                stratify=pd.concat([Case_indexes["prefix group"],
        #                                                    Case_indexes[self.label_col]], axis=1))
        #
        # train, test = train_test_split(Case_indexes, train_size=train_ratio,
        #                                stratify=Case_indexes[self.outcome], random_state=142)
        if sample:
            Case_indexes, _ = train_test_split(Case_indexes, train_size=0.1, random_state=142,
                                               stratify=Case_indexes[self.outcome])

        train, test = train_test_split(Case_indexes, train_size=train_ratio, random_state=142,
                                       stratify=Case_indexes[self.outcome])

        train_list = train[case_id_col]
        test_list = test[case_id_col]

        return train_list, test_list

    def generate_prefix_data(self, data):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id)[self.activity].transform(len)
        data['base_case_id'] = data[self.case_id]
        dt_prefixes = data[data['case_length'] >= self.min_length].groupby(self.case_id).head(self.min_length)
        for nr_events in range(self.min_length + 1, self.max_length + 1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id).head(nr_events)
            tmp[self.case_id] = tmp[self.case_id].apply(lambda x: "%s_%s" % (x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id)[self.activity].transform(len)
        dt_prefixes = dt_prefixes.reset_index(drop=True)

        return dt_prefixes

    def encode_all_dataset(self, LPMs=False):
        dictionary_acts = self.get_act_dict()
        dictionary_labels = self.get_label_dict()
        if LPMs:
            dictionary_lpms = self.get_lpms_dict()
        # train_list, test_list = self.split_test_train(train_ratio, sample)
        # dt_prefixes_trian = self.generate_prefix_data(self.data[self.data[self.case_id].isin(train_list)])
        # dt_prefixes_trian = self.data[self.data[self.case_id].isin(train_list)]
        dt_prefixes_trian = self.data
        token_x = list()
        token_y = list()
        train_lpms = list()
        for case in dt_prefixes_trian[self.case_id].unique():
            trace = dt_prefixes_trian[dt_prefixes_trian[self.case_id] == case][self.activity].tolist()
            encoded_trace = [dictionary_acts[s] for s in trace]
            token_x.append(encoded_trace)
            outcome = dictionary_labels[
                dt_prefixes_trian[dt_prefixes_trian[self.case_id] == case][self.outcome].values[0]]
            token_y.append(outcome)
            if LPMs:
                lpms_sequences = dt_prefixes_trian[dt_prefixes_trian[self.case_id] == case][self.lpms].tolist()
                encoded_lpms = [dictionary_lpms[s] for s in lpms_sequences]
                train_lpms.append(encoded_lpms)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=self.max_length)
        train_x = np.array(token_x, dtype=np.float32)
        train_y = np.array(token_y, dtype=np.float32)
        train_lpms = tf.keras.preprocessing.sequence.pad_sequences(
            train_lpms, maxlen=self.max_length)
        train_lpms = np.array(train_lpms, dtype=np.float32)

        return train_x, train_y, train_lpms

    def encode_prefixes(self, LPMs=False, train_ratio=0.8, sample=False):
        dictionary_acts = self.get_act_dict()
        dictionary_labels = self.get_label_dict()
        if LPMs:
            dictionary_lpms = self.get_lpms_dict()
        train_list, test_list = self.split_test_train(train_ratio, sample)
        # dt_prefixes_trian = self.generate_prefix_data(self.data[self.data[self.case_id].isin(train_list)])
        dt_prefixes_trian = self.data[self.data[self.case_id].isin(train_list)]
        token_x = list()
        token_y = list()
        train_lpms = list()
        for case in dt_prefixes_trian[self.case_id].unique():
            trace = dt_prefixes_trian[dt_prefixes_trian[self.case_id] == case][self.activity].tolist()
            encoded_trace = [dictionary_acts[s] for s in trace]
            token_x.append(encoded_trace)
            outcome = dictionary_labels[
                dt_prefixes_trian[dt_prefixes_trian[self.case_id] == case][self.outcome].values[0]]
            token_y.append(outcome)
            if LPMs:
                lpms_sequences = dt_prefixes_trian[dt_prefixes_trian[self.case_id] == case][self.lpms].tolist()
                encoded_lpms = [dictionary_lpms[s] for s in lpms_sequences]
                train_lpms.append(encoded_lpms)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=self.max_length)
        train_x = np.array(token_x, dtype=np.float32)
        train_y = np.array(token_y, dtype=np.float32)

        if LPMs:
            train_lpms = tf.keras.preprocessing.sequence.pad_sequences(
                train_lpms, maxlen=self.max_length)
            train_lpms = np.array(train_lpms, dtype=np.float32)

        # dt_prefixes_test = self.generate_prefix_data(self.data[self.data[self.case_id].isin(test_list)])
        dt_prefixes_test = self.data[self.data[self.case_id].isin(test_list)]
        test_x = list()
        test_y = list()
        test_lpms = list()
        for case in dt_prefixes_test[self.case_id].unique():
            trace = dt_prefixes_test[dt_prefixes_test[self.case_id] == case][self.activity].tolist()
            encoded_trace = [dictionary_acts[s] for s in trace]
            test_x.append(encoded_trace)
            outcome = dictionary_labels[
                dt_prefixes_test[dt_prefixes_test[self.case_id] == case][self.outcome].values[0]]
            test_y.append(outcome)
            if LPMs:
                lpms_sequences_test = dt_prefixes_test[dt_prefixes_test[self.case_id] == case][self.lpms].tolist()
                encoded_lpms_test = [dictionary_lpms[s] for s in lpms_sequences_test]
                test_lpms.append(encoded_lpms_test)

        test_x = tf.keras.preprocessing.sequence.pad_sequences(
            test_x, maxlen=self.max_length)
        test_x = np.array(test_x, dtype=np.float32)
        test_y = np.array(test_y, dtype=np.float32)

        if LPMs:
            test_lpms = tf.keras.preprocessing.sequence.pad_sequences(
                test_lpms, maxlen=self.max_length)
            test_lpms = np.array(test_lpms, dtype=np.float32)

        return train_x, train_y, train_lpms, test_x, test_y, test_lpms

    def init_model(self, LPMs=True):
        print("Initialising default model")
        # model = Sequential()
        if LPMs:
            Input_act = Input(shape=self.max_length, name='inputact')
            Input_lpm = Input(shape=self.max_length, name='inputlpm')
            embedding_act = Embedding(self.num_acts, 13, input_length=self.max_length, name='act')(Input_act)
            embedding_lpm = Embedding(self.num_lpms, 8, input_length=self.max_length, name='lpms')(Input_lpm)
            concatenating = Concatenate(name='concatenated', axis=2)([embedding_act, embedding_lpm])
            lstm_1 = LSTM(50, dropout=0.099)(concatenating)
            flat = Flatten()(lstm_1)
            dense = Dense(1, activation='sigmoid')(flat)
            self.model = Model(inputs=[Input_act, Input_lpm],
                               outputs=[dense])
        else:
            Input_act = Input(shape=self.max_length, name='inputact')
            embedding_act = Embedding(self.num_acts, 10, input_length=self.max_length, name='act')(Input_act)
            lstm_1 = LSTM(10, dropout=0.1)(embedding_act)
            flat = Flatten()(lstm_1)
            dense = Dense(1, activation='sigmoid')(flat)
            self.model = Model(inputs=Input_act,
                               outputs=dense)

        # model.add(Flatten())
        # model.add(Dense(1, activation='sigmoid'))
        # self.model = model

        return self.model.summary()

    def run_the_model(self, LPMs=True, optimizer='rmsprop', epochs=2, validation_split=0.2):
        optimizer = keras.optimizers.Adam(lr=0.0000687)
        train_x, train_y, train_lpms, test_x, test_y, test_lpms = self.encode_prefixes()
        model_summary = self.init_model(LPMs=LPMs)
        print(model_summary)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(),
                                                                                     tf.keras.metrics.Recall(), 'acc'])
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        self.epochs = epochs
        if LPMs:
            self.history = self.model.fit([train_x, train_lpms], train_y,
                                          batch_size=8,
                                          epochs=epochs,
                                          validation_split=validation_split,
                                          callbacks=early_stopping)

            # self.plot_model_output()
            self.final_evaluation(test_x, test_y, test_lpms)
        else:
            self.history = self.model.fit(train_x, train_y, epochs=epochs,
                                          validation_split=validation_split,
                                          callbacks=early_stopping)

            # self.plot_model_output()
            self.final_evaluation(test_x, test_y)

        return train_x, train_y, train_lpms, test_x, test_y, test_lpms

    def plot_model_output(self):
        history = self.history
        epochs = self.epochs
        plt.figure()
        plt.plot(range(epochs, ), history.history['loss'], label='training_loss')
        plt.plot(range(epochs, ), history.history['val_loss'], label='validation_loss')
        plt.legend()
        plt.figure()
        plt.plot(range(epochs, ), history.history['acc'], label='training_accuracy')
        plt.plot(range(epochs, ), history.history['val_acc'], label='validation_accuracy')
        plt.legend()
        plt.show()

    def final_evaluation(self, x_test, y_test, lpm_test=None):
        # predict crisp classes for test set
        if lpm_test is not None:
            yhat_probs = self.model.predict([x_test, lpm_test], verbose=0)
        else:
            yhat_probs = self.model.predict(x_test, verbose=0)

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
        print(cm)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        print('Testing Accuracy =', (TP + TN) / (TP + FP + FN + TN))
        print('Testing F-score = ', (TP / (TP + 0.5 * (FP + FN))))
