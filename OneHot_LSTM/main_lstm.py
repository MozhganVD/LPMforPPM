import os

import numpy as np
import pandas as pd
import config
from feature_encoder import FeatureEncoder
from feature_generator import FeatureGenerator
from LSTM_model import net
from xgboost import XGBClassifier
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


def evaluate_each_prefix(test_X, test_y, prefixes, checkpoint_dir, filename, model_type):
    K_all, accuracies, fscores, precisions, recalls = [], [], [], [], []
    test_size = test_X.shape[0]

    for k in prefixes:
        indices = [i for i, x in enumerate(test_prefixes) if x == k]
        Num_prefix_k = len(indices)
        # Weight_for_k = Num_prefix_k / test_size
        if model_type == "LSTM":
            _, _, Accuracy, F1_Score, Precision, Recall = model.evaluate(test_X[np.array(indices)],
                                                                                      test_y[np.array(indices)])
        elif model_type == "XGB" or model_type == "RF":
            y_prediction = model.predict(test_X[np.array(indices)])
            predictions = [round(value) for value in y_prediction]
            Accuracy = accuracy_score(test_y[np.array(indices)], predictions)
            F1_Score = f1_score(test_y[np.array(indices)], predictions, average="weighted")
            Precision = precision_score(test_y[np.array(indices)], predictions, average="weighted")
            Recall = recall_score(test_y[np.array(indices)], predictions, average="weighted")
        else:
            print("model type is not defined!")

        K_all.append(k)
        accuracies.append(Accuracy)
        fscores.append(F1_Score)
        precisions.append( Precision)
        recalls.append(Recall)

    results_df = pd.DataFrame({"k": K_all, "accuracy": accuracies, "fscore": fscores,
                               "precision": precisions, "recall": recalls})
    results_df.to_csv(checkpoint_dir + filename + ".csv", index=False)


def write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Running_Time, results_file):
    with open(results_file, 'w') as f:
        f.write("Running time is %.5f \n" % Running_Time)
        f.write(str(CF_matrix))
        f.write("\n")
        f.write(str(report))
        f.write("\n")
        f.write("Accuracy = %.5f \n" % Accuracy)
        f.write("F1_Score = %.5f \n" % F1_Score)
        f.write("Precision = %.5f \n" % Precision)
        f.write("Recall = %.5f \n" % Recall)
        f.close()


if __name__ == '__main__':
    args = config.load()
    Lpms_type = args.LPMs_type
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    filename = args.data_dir + args.data_set
    filename_test = args.data_dir + args.data_set_test
    model_name = '%s-%s_%s' % (
        args.data_set, num_epochs, batch_size)
    feature_name = '%s' % args.data_set
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # load data
    print("flag: loading training data")
    fg = FeatureGenerator()
    train_df = fg.create_initial_log(filename)
    print("done")
    print(len(set(train_df["Case ID"])))

    loss = 'binary_crossentropy'
    feature_type_list = ["activity_history"]

    print("flag: loading testing data")
    fg_test = FeatureGenerator()
    test_df = fg_test.create_initial_log(filename_test)
    print("done")
    print(len(set(test_df["Case ID"])))
    prefixes = np.unique(test_df["k"])
    test_prefixes = test_df["k"]

    # Encoding training set
    print("flag: encoding training features")
    fe = FeatureEncoder()
    if args.model == "LSTM":
        print("flag: training encoding")
        train_X, train_y = fe.one_hot_encoding(train_df, feature_name, "train",
                                               features=args.features, LPMs=args.LPMs,
                                               Lpms_type=Lpms_type, Normalize=args.LPMs_Normal)
    elif args.model == "XGB" or args.model == "RF":
        print("flag: training encoding")
        train_X, train_y = fe.one_hot_encoding_xgb(train_df, feature_name, "train", features=args.features,
                                                   LPMs=args.LPMs, Lpms_type=Lpms_type)
    print("done")

    print("flag: training model")
    if args.model == "LSTM":
        model = net()
        model.train(train_X, train_y, loss, n_epochs=num_epochs, batch_size=batch_size,
                    model_name=model_name, checkpoint_dir=args.checkpoint_dir)
        Running_Time = model.running_time
        print('LSTM training time is %.3f S' % model.running_time)
        print("flag: encoding features")
        fe_test = FeatureEncoder()
        print("flag: testing encoding")
        test_X, test_y = fe_test.one_hot_encoding(test_df, feature_name, "test",
                                                  features=args.features, LPMs=args.LPMs,
                                                  Lpms_type=Lpms_type, Normalize=args.LPMs_Normal)
        print("done")
        model.load(args.checkpoint_dir, model_name=model_name)
        CF_matrix, report, Accuracy, F1_Score, Precision, Recall = model.evaluate(test_X, test_y)
        results_file = args.checkpoint_dir + "Results_LSTM.txt"
        write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Running_Time, results_file)
        evaluate_each_prefix(test_X, test_y, prefixes, args.checkpoint_dir, "results_prefixes_LSTM", "LSTM")
        print("done")

    elif args.model == "XGB":
        model = XGBClassifier()
        Start_time = time.time()
        model.fit(train_X, train_y, )
        Running_Time = time.time() - Start_time
        fe_test = FeatureEncoder()
        print("flag: testing encoding")
        # test_X, test_y = fe_test.Testing_one_hot_encode_xgb(test_df, feature_type_list, feature_name)
        test_X, test_y = fe_test.one_hot_encoding_xgb(test_df, feature_name, "test", features=args.features,
                                                      LPMs=args.LPMs, Lpms_type=Lpms_type)
        print("done")
        y_prediction = model.predict(test_X)
        predictions = [round(value) for value in y_prediction]
        # evaluate
        CF_matrix = confusion_matrix(test_y, predictions)
        print(CF_matrix)
        # Print the precision and recall, among other metrics
        report = classification_report(test_y, predictions, digits=3)
        print(report)
        Accuracy = accuracy_score(test_y, predictions)
        F1_Score = f1_score(test_y, predictions, average="weighted")
        Precision = precision_score(test_y, predictions, average="weighted")
        Recall = recall_score(test_y, predictions, average="weighted")
        print("Accuracy: %.5f" % Accuracy)
        print("F1-Score: %.5f" % F1_Score)
        print("precision: %.5f" % Precision)
        print("recall: %.5f" % Recall)
        print('XGB training time is %.3f S' % Running_Time)
        results_file = args.checkpoint_dir + "Results_XGBoost.txt"
        write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Running_Time, results_file)
        evaluate_each_prefix(test_X, test_y, prefixes, args.checkpoint_dir, "results_prefixes_XGBoost", "XGB")

        print("done")

    elif args.model == "RF":
        model = RandomForestClassifier(n_estimators=100)
        Start_time = time.time()
        model.fit(train_X, train_y, )
        Running_Time = time.time() - Start_time
        fe_test = FeatureEncoder()
        print("flag: testing encoding")
        # test_X, test_y = fe_test.Testing_one_hot_encode_xgb(test_df, feature_type_list, feature_name)
        test_X, test_y = fe_test.one_hot_encoding_xgb(test_df, feature_name, "test", features=args.features,
                                                      LPMs=args.LPMs, Lpms_type=Lpms_type)
        print("done")
        y_prediction = model.predict(test_X)
        predictions = [round(value) for value in y_prediction]
        # evaluate
        CF_matrix = confusion_matrix(test_y, predictions)
        print(CF_matrix)
        # Print the precision and recall, among other metrics
        report = classification_report(test_y, predictions, digits=3)
        print(report)
        Accuracy = accuracy_score(test_y, predictions)
        F1_Score = f1_score(test_y, predictions, average="weighted")
        Precision = precision_score(test_y, predictions, average="weighted")
        Recall = recall_score(test_y, predictions, average="weighted")
        print("Accuracy: %.5f" % Accuracy)
        print("F1-Score: %.5f" % F1_Score)
        print("precision: %.5f" % Precision)
        print("recall: %.5f" % Recall)
        print('XGB training time is %.3f S' % Running_Time)
        results_file = args.checkpoint_dir + "Results_RF.txt"
        write_results_to_text(CF_matrix, report, Accuracy, F1_Score, Precision, Recall, Running_Time, results_file)
        evaluate_each_prefix(test_X, test_y, prefixes, args.checkpoint_dir, "results_prefixes_RF", "RF")
        print("done")

