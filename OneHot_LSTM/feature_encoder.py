import pickle
import numpy as np
import time


class FeatureEncoder(object):
    def one_hot_encoding_Wrapped(self, df, coded_activity, coded_labels, maxlen, Lpms_type="LPMs_binary"):

        Outcomes_list = list(coded_labels.keys())
        feature_set = list(coded_activity.keys())
        feature_len = len(feature_set)
        dict_feature_char_to_int = dict((str(c), i) for i, c in enumerate(feature_set))
        outcome_char_to_int = {j: i for i, j in enumerate(Outcomes_list)}

        X_train = list()
        y_train = list()

        for ii in range(0, len(df)):
            # print("{}th among {}".format(ii, len(df)))
            # prepare X
            onehot_encoded_X = list()
            hist_len = int(df.at[ii, "k"])
            parsed_hist = str(df.at[ii, "prefix"]).split(" ")
            seq_lpm = [int(ll.split(".")[0].replace(" ", "")) for ll in
                       str(df.at[ii, Lpms_type]).strip('[]').split(',')]

            for jj in range(hist_len):
                merged_encoding = list()
                feature = parsed_hist[jj]
                feature_int = dict_feature_char_to_int[feature]
                onehot_encoded_feature = [0 for _ in range(feature_len)]
                onehot_encoded_feature[feature_int] = 1
                onehot_encoded_feature[feature_int] += seq_lpm[jj]

                merged_encoding += onehot_encoded_feature
                onehot_encoded_X.append(merged_encoding)
            while len(onehot_encoded_X) != maxlen:
                onehot_encoded_X.insert(0, [0] * feature_len)
            X_train.append(onehot_encoded_X)

            # prepare y
            outcome = str(df.at[ii, 'label'])
            y_train.append(outcome_char_to_int[outcome])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        return X_train, y_train

    def one_hot_encoding_Classic(self, df, coded_activity, coded_labels, coded_lpms, maxlen, LPMs=False):

        Outcomes_list = list(coded_labels.keys())
        feature_set = list(coded_activity.keys())

        if LPMs:
            All_lpms_combined = list(coded_lpms.keys())
            Unique_lpms = []
            for l in All_lpms_combined:
                Unique_lpms.extend(l.split("+"))

            Unique_lpms = list(np.unique(Unique_lpms))
            Unique_lpms.remove(str())
            coded_unique_lpms = dict(zip(Unique_lpms, range(len(Unique_lpms))))

            feature_set.extend(coded_unique_lpms.keys())

        feature_len = len(feature_set)

        dict_feature_char_to_int = dict((str(c), i) for i, c in enumerate(feature_set))
        outcome_char_to_int = {j: i for i, j in enumerate(Outcomes_list)}
        if LPMs:
            dict_all_lpm_char_to_int = dict((i, str(c)) for i, c in enumerate(coded_lpms))

        X_train = list()
        y_train = list()

        for ii in range(0, len(df)):
            # print("{}th among {}".format(ii, len(df)))
            onehot_encoded_X = list()
            parsed_hist = str(df.at[ii, "prefix"]).split(" ")
            parsed_hist = parsed_hist[:maxlen]
            hist_len = len(parsed_hist)
            if LPMs:
                seq_lpm = [int(ll.split(".")[0].replace(" ", "")) for ll in
                           str(df.at[ii, 'LPMs_list']).strip('[]').split(',')]

            for jj in range(hist_len):
                merged_encoding = list()
                feature = parsed_hist[jj]
                feature_int = dict_feature_char_to_int[feature]
                onehot_encoded_feature = [0 for _ in range(feature_len)]
                onehot_encoded_feature[feature_int] = 1

                if LPMs:
                    if seq_lpm[jj] != 0:
                        lpmS = dict_all_lpm_char_to_int[seq_lpm[jj]].split("+")
                        for l in lpmS:
                            if l != str():
                                lpm_index = dict_feature_char_to_int[l]
                                onehot_encoded_feature[lpm_index] = 1

                merged_encoding += onehot_encoded_feature
                onehot_encoded_X.append(merged_encoding)

            while len(onehot_encoded_X) < maxlen:
                onehot_encoded_X.insert(0, [0] * feature_len)

            X_train.append(onehot_encoded_X)

            # prepare y
            outcome = str(df.at[ii, 'label'])
            y_train.append(outcome_char_to_int[outcome])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        return X_train, y_train
