import warnings
import numpy as np
import tensorflow as tf
import keras
from keras import Input
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, \
    classification_report
import time

warnings.filterwarnings("ignore")


class net:
    def __init__(self):
        pass

    def evaluate(self, x_test, y_test):
        # calculate confusion matrix and weighted recall and precision 
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=[tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(), 'acc'])
        y_pred = self.model.predict(x_test)
        # predictions = np.argmax(y_pred, axis=1)
        # y_test_integer = np.argmax(y_test, axis=1)
        predictions = [int(np.round(a)) for a in y_pred]
        predictions = np.array(predictions)

        y_test_integer = [int(np.round(a)) for a in y_test]
        y_test_integer = np.array(y_test_integer)

        CF_matrix = confusion_matrix(y_test_integer, predictions)
        report = classification_report(y_test_integer, predictions, digits=5)
        # Print the precision and recall, among other metrics
        Accuracy = accuracy_score(y_test_integer, predictions)
        F1_Score = f1_score(y_test_integer, predictions)
        Precision = precision_score(y_test_integer, predictions)
        Recall = recall_score(y_test_integer, predictions)

        return CF_matrix, report, Accuracy, F1_Score, Precision, Recall

    def train(self, X_train, y_train, loss, n_epochs=100,
              y_normalize=False, tau=1.0, dropout=0.1, batch_size=128,
              model_name='predictor', checkpoint_dir='./checkpoints/'):

        if y_normalize:
            self.mean_y_train = np.mean(y_train)
            self.std_y_train = np.std(y_train)

            y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
            y_train_normalized = np.array(y_train_normalized, ndmin=2).T
        else:
            if len(y_train.shape) == 1:
                y_train_normalized = np.array(y_train, ndmin=2).T
            else:
                y_train_normalized = y_train

        # We construct the network
        N = X_train.shape[0]
        batch_size = batch_size
        val_split = 0.2

        print(X_train.shape[1], X_train.shape[2], y_train_normalized.shape[1])
        print("**************************")
        inputs = Input(shape=(X_train.shape[1], X_train.shape[2]), name='main_input')
        # inter = Dropout(dropout)(inputs, training=True)
        inter = LSTM(30, recurrent_dropout=dropout, return_sequences=True)(inputs, training=True)
        # inter = BatchNormalization()(inter)
        inter = Dropout(dropout)(inter, training=True)
        inter = LSTM(30, )(inputs, training=True)
        # inter = BatchNormalization()(inter)
        inter = Dropout(dropout)(inter, training=True)
        # outputs = Dense(y_train_normalized.shape[1], activation='softmax')(inter)
        outputs = Dense(1, activation='sigmoid')(inter)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=loss, optimizer='adam',
                      metrics=[tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), 'acc'])
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s_.h5' % (checkpoint_dir, model_name),
                                                           monitor='val_loss', verbose=0, save_best_only=True,
                                                           save_weights_only=False, mode='auto')
        lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                       mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=1,
                  validation_split=val_split, callbacks=[early_stopping, model_checkpoint, lr_reducer])

        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

    def load(self, checkpoint_dir, model_name):
        model = load_model('%smodel_%s_.h5' % (checkpoint_dir, model_name),
                           custom_objects={'categorical_precision': tf.keras.metrics.Precision(),
                                           'categorical_recall': tf.keras.metrics.Recall()})
        self.model = model

    def predict(self, X_test):
        X_test = np.array(X_test, ndmin=3)
        model = self.model
        T = 10
        Yt_hat = np.array([model.predict(X_test, batch_size=1, verbose=0) for _ in range(T)])
        MC_pred = np.mean(Yt_hat, 0)
        MC_uncertainty = list()
        for i in range(Yt_hat.shape[2]):
            MC_uncertainty.append(np.std(Yt_hat[:, :, i].squeeze(), 0))

        return MC_pred, MC_uncertainty
