import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from .constants import Task
import random

random.seed(2021)

class LogsDataProcessor:
    def __init__(self, filepath, dire_path, pool=1,
                 features=False, LPMs=False):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._dir_path = dire_path
        self._features = features
        self._LPMs = LPMs
        self._filepath = filepath
        self._pool = pool
        self.case_id_col = "case:concept:name"
        self.activity_col = "concept:name"
        self.time_col = "time:timestamp"
        self.label_col = "label"
        self.pos_label = "deviant"
        self.neg_label = "regular"
        # different LPMs encoding type
        self.lpms_binary = "LPMs_binary"
        self.lpms_frequency = "LPMs_frequency"

    def _load_df(self):
        df = pd.read_csv(self._filepath, sep=";")
        df = df[[self.case_id_col, self.activity_col, self.label_col, self.lpms_frequency,
                 self.lpms_binary, self.time_col, 'base_case_id']]
        df.columns = [self.case_id_col, self.activity_col, self.label_col, self.lpms_frequency,
                      self.lpms_binary, self.time_col, 'base_case_id']

        df[self.activity_col] = df[self.activity_col].str.lower()
        df[self.activity_col] = df[self.activity_col].str.replace(" ", "-")
        # df[self.time_col] = df[self.time_col].str.replace("/", "-")
        # df[self.time_col] = pd.to_datetime(df[self.time_col],
        #                                    dayfirst=True).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f"))

        return df

    def _extract_logs_metadata(self, df):
        activities = list(df[self.activity_col].unique())
        outcomes = [self.pos_label, self.neg_label]
        coded_activity = dict(zip(activities, range(len(activities))))
        coded_labels = dict(zip(outcomes, range(len(outcomes))))
        return coded_activity, coded_labels

    def _next_activity_helper_func(self, df):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns=["case_id",
                                             "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                next_act = act[i + 1]
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "next_act"] = next_act
                idx = idx + 1
        return processed_df

    def _process_next_activity(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test.csv", index=False)

    def _outcome_helper_func(self, df, Max_prefix_length):
        case_id, case_name = self.case_id_col, self.activity_col
        lpm_measures = [self.lpms_binary, self.lpms_frequency]
        processed_df = pd.DataFrame(columns=[self.case_id_col, "prefix", "k", self.label_col, ])

        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            if self._LPMs:
                lpms = dict()
                for l in lpm_measures:
                    lpms[l] = df[df[case_id] == case][l].to_list()

            Outcome = df[df[case_id] == case][self.label_col].to_list()[0]
            if len(act) > Max_prefix_length:
                act = act[:Max_prefix_length]
                if self._LPMs:
                    for ll in lpm_measures:
                        lpms[ll] = lpms[ll][:Max_prefix_length]

            for i in range(1, len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))

                processed_df.at[idx, self.case_id_col] = case
                processed_df.at[idx, "prefix"] = prefix
                if self._LPMs:
                    for ll in lpm_measures:
                        LPM_seq = np.where(i == 0, str(lpms[ll][0]), " ".join(str(lpms[ll][:i + 1])))
                        processed_df.at[idx, ll] = LPM_seq

                processed_df.at[idx, "k"] = i+1
                processed_df.at[idx, self.label_col] = Outcome
                idx = idx + 1
                print(idx, Outcome)

        return processed_df

    def _outcome_helper_func_complete(self, df, Max_prefix_length):
        case_id, case_name = self.case_id_col, self.activity_col
        lpm_measures = [self.lpms_binary, self.lpms_frequency]
        processed_df = pd.DataFrame(columns=[self.case_id_col, "prefix", "k", self.label_col, 'base_case_id'])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            if self._LPMs:
                lpms = dict()
                for l in lpm_measures:
                    lpms[l] = df[df[case_id] == case][l].to_list()

            Outcome = df[df[case_id] == case][self.label_col].to_list()[0]
            processed_df.at[idx, self.case_id_col] = case
            processed_df.at[idx, "prefix"] = " ".join(act)
            if self._LPMs:
                for ll in lpm_measures:
                    processed_df.at[idx, ll] = " ".join(str(lpms[ll]))

            processed_df.at[idx, "k"] = len(act)
            processed_df.at[idx, self.label_col] = Outcome
            base_case = df[df[case_id] == case]['base_case_id'].to_list()[0]
            processed_df.at[idx, 'base_case_id'] = base_case
            idx = idx + 1
            print(idx, Outcome)

        return processed_df

    def _process_outcome(self, df, train_list, test_list, Max_prefix_length, complete_lpms):
        # df_split = np.array_split(df, self._pool)
        # with Pool(processes=self._pool) as pool:
        #     processed_df = pd.concat(pool.imap_unordered(self._outcome_helper_func, df_split))
        if complete_lpms:
            processed_df = self._outcome_helper_func_complete(df, Max_prefix_length)
            train_df = processed_df[processed_df['base_case_id'].isin(train_list)]
            test_df = processed_df[processed_df['base_case_id'].isin(test_list)]
        else:
            processed_df = self._outcome_helper_func(df, Max_prefix_length)
            train_df = processed_df[processed_df[self.case_id_col].isin(train_list)]
            test_df = processed_df[processed_df[self.case_id_col].isin(test_list)]

        # return train_df, test_df
        train_df.to_csv(f"{self._dir_path}/outcome_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/outcome_test.csv", index=False)

    def _next_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed",
                                             "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            next_time = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y.%m.%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 1], "%Y.%m.%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y.%m.%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 2], "%Y.%m.%d %H:%M:%S")
                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time
                if i + 1 < len(time):
                    next_time = datetime.datetime.strptime(time[i + 1], "%Y.%m.%d %H:%M:%S") - \
                                datetime.datetime.strptime(time[i], "%Y.%m.%d %H:%M:%S")
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "next_time"] = next_time_days
                idx = idx + 1
        processed_df_time = processed_df[["case_id", "prefix", "k", "time_passed",
                                          "recent_time", "latest_time", "next_time"]]
        return processed_df_time

    def _process_next_time(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_time_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_test.csv", index=False)

    def _remaining_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed",
                                             "recent_time", "latest_time", "next_act", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y.%m.%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 1], "%Y.%m.%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y.%m.%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 2], "%Y.%m.%d %H:%M:%S")

                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], "%Y.%m.%d %H:%M:%S") - \
                      datetime.datetime.strptime(time_stamp, "%Y.%m.%d %H:%M:%S")
                ttc = str(ttc.days)

                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1
        processed_df_remaining_time = processed_df[["case_id", "prefix", "k",
                                                    "time_passed", "recent_time", "latest_time", "remaining_time_days"]]
        return processed_df_remaining_time

    def _process_remaining_time(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._remaining_time_helper_func, df_split))
        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_train.csv", index=False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_test.csv", index=False)

    def test_train_spliting(self, df, train_ratio, Max_prefix_length, complete_lpms):
        Cases = []
        Prefixes = []
        Labels = []
        if complete_lpms:
            case_id_col = 'base_case_id'
        else:
            case_id_col = self.case_id_col

        for case in df[case_id_col].unique():
            Cases.append(case)
            Prefixes.append(min(Max_prefix_length, len(df[df[case_id_col] == case])))
            Labels.append(list(df[df[case_id_col] == case][self.label_col].unique())[0])

        Case_indexes = pd.DataFrame({case_id_col: Cases, "prefix length": Prefixes, self.label_col: Labels})
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
        train, test = train_test_split(Case_indexes, train_size=train_ratio,
                                       stratify=Case_indexes[self.label_col])

        train_list = train[case_id_col]
        test_list = test[case_id_col]

        return train_list, test_list, Max_prefix_length

    def process_logs(self,
                     train_ratio=0.80,
                     trunc_ratio=0.90,
                     complete_lpms=False):

        df = self._load_df()
        coded_activity, coded_labels = self._extract_logs_metadata(df)
        train_list, test_list, Max_prefix_length = self.test_train_spliting(df, train_ratio, trunc_ratio, complete_lpms)
        self._process_outcome(df, train_list, test_list, Max_prefix_length, complete_lpms)

        return Max_prefix_length, coded_activity, coded_labels

    def process_logs_light(self,
                     train_ratio=0.80,
                     trunc_ratio=0.90):

        df = self._load_df()
        coded_activity, coded_labels = self._extract_logs_metadata(df)
        train_list, test_list, Max_prefix_length = self.test_train_spliting(df, train_ratio, trunc_ratio)

        return Max_prefix_length, coded_activity, coded_labels
