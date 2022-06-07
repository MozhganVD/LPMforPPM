import pickle
import numpy as np
from sys import getsizeof
import time


class FeatureEncoder(object):
    def one_hot_encoding_xgb(self, df, feature_name, purpose, features=False, LPMs=False, Lpms_type="LPMs_binary"):
        df = df.sample(frac=1)
        num_features_dict = dict()
        feature_len = 0
        dict_feature_char_to_int = dict()
        Outcomes_list = ["regular", "deviant"]
        feature_type = "activity_history"
        if purpose == "test":
            with open("%s/%s_%s.pkl" % ("./", feature_type, feature_name), 'rb') as f:
                feature_set = pickle.load(f)
        if purpose == "train":
            activities_list = []
            for seq in df["prefix"]:
                activities = np.unique(seq.split(" "))
                activities_list.extend(activities)
            feature_set = sorted(list(set(activities_list)))
            with open("%s/%s_%s.pkl" % ("./", feature_type, feature_name), 'wb') as f:
                pickle.dump(feature_set, f)
        num_feature = len(feature_set)
        feature_len += num_feature
        num_features_dict[feature_type] = num_feature
        dict_feature_char_to_int[feature_type] = dict((str(c), i) for i, c in enumerate(feature_set))
        outcome_char_to_int = {j: i for i, j in enumerate(Outcomes_list)}
        feature_int_to_char = dict((i, c) for i, c in enumerate(feature_set))

        X_train = list()
        y_train = list()
        if purpose == "train":
            maxlen = max([len(str(x).split(' ')) for x in df["prefix"]])
            with open("%s/%s_%s.pkl" % ("./", "maxlen", feature_name), 'wb') as f:
                pickle.dump(maxlen, f)
        if purpose == "test":
            with open("%s/%s_%s.pkl" % ("./", "maxlen", feature_name), 'rb') as f:
                maxlen = pickle.load(f)
        for i in range(0, len(df)):
            # if purpose == "test":
            #     if i > 20000:
            #         break
            # else:
            #     if i > 100000:
            #         break
            print("{}th among {}".format(i, len(df)))
            # prepare X
            onehot_encoded_X = list()
            hist_len = len(str(df.at[i, "prefix"]).split(" "))
            if LPMs:
                seq_lpm = [int(l.split(".")[0].replace(" ", "")) for l in
                           df.at[i, "LPMs_frequency"].strip('[]').split(',')]
            merged_encoding = list()
            for j in range(hist_len):
                parsed_hist = str(df.at[i, "prefix"]).split(" ")
                feature = parsed_hist[j]
                feature_int = dict_feature_char_to_int[feature_type][feature]
                onehot_encoded_feature = [0 for _ in range(num_features_dict[feature_type])]
                onehot_encoded_feature[feature_int] = 1
                if LPMs:
                    onehot_encoded_feature[feature_int] += seq_lpm[j]
                merged_encoding += onehot_encoded_feature
            if features:
                merged_encoding.append(df.at[i, "CreditScore"] / 100)
                Threshold_zeroes = (maxlen * num_feature) + 1
            else:
                Threshold_zeroes = maxlen * num_feature

            while len(merged_encoding) != Threshold_zeroes:
                merged_encoding.insert(0, 0)
            X_train.append(merged_encoding)

            # next_act = str(df.at[i, 'next_activity'])
            # int_encoded_next_act = dict_feature_char_to_int["activity_history"][next_act]
            # activities = sorted(list(set(df['activity'])))
            # activities.append("!")
            # y_train.append(int_encoded_next_act)
            outcome = str(df.at[i, 'outcome'])
            y_train.append(outcome_char_to_int[outcome])

        # print(getsizeof(X_train))
        X_train = np.asarray(X_train)
        # print(X_train.nbytes())
        y_train = np.asarray(y_train)
        return X_train, y_train

    def one_hot_encoding(self, df, coded_activity, coded_labels, maxlen, features=False, LPMs=False,
                         Lpms_type="LPMs_binary", Normalize=False):
        # dict_feature_char_to_int = dict()
        # feature_type = "activity_history"
        Outcomes_list = list(coded_labels.keys())
        feature_set = list(coded_activity.keys())
        feature_len = len(feature_set)
        dict_feature_char_to_int = dict((str(c), i) for i, c in enumerate(feature_set))
        outcome_char_to_int = {j: i for i, j in enumerate(Outcomes_list)}
        # feature_int_to_char = dict((i, c) for i, c in enumerate(feature_set))

        X_train = list()
        y_train = list()

        for ii in range(0, len(df)):
            # print("{}th among {}".format(ii, len(df)))
            # prepare X
            onehot_encoded_X = list()
            hist_len = int(df.at[ii, "k"])
            parsed_hist = str(df.at[ii, "prefix"]).split(" ")
            if LPMs:
                seq_lpm = [int(ll.split(".")[0].replace(" ", "")) for ll in
                           str(df.at[ii, Lpms_type]).strip('[]').split(',')]
            for jj in range(hist_len):
                merged_encoding = list()
                feature = parsed_hist[jj]
                feature_int = dict_feature_char_to_int[feature]
                onehot_encoded_feature = [0 for _ in range(feature_len)]
                onehot_encoded_feature[feature_int] = 1
                if LPMs:
                    if Normalize and Lpms_type == "LPMs_frequency":
                        onehot_encoded_feature[feature_int] += (seq_lpm[jj] / 15)
                    else:
                        onehot_encoded_feature[feature_int] += seq_lpm[jj]

                merged_encoding += onehot_encoded_feature
                onehot_encoded_X.append(merged_encoding)
            while len(onehot_encoded_X) != maxlen:
                onehot_encoded_X.insert(0, [0] * feature_len)
            X_train.append(onehot_encoded_X)

            # prepare y
            outcome = str(df.at[ii, 'label'])
            # int_encoded_outcome = outcome_char_to_int[outcome]
            # activities = sorted(list(set(df['activity'])))
            # activities.append("!")
            # onehot_encoded_next_act = [0 for _ in range(feature_len)]
            # onehot_encoded_next_act[int_encoded_outcome] = 1
            y_train.append(outcome_char_to_int[outcome])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        return X_train, y_train
