import random
import pickle
import numpy as np
import pandas as pd


class FeatureGenerator(object):
    date_format = "%Y.%m.%d %H:%M"

    def train_test_split(self, df, test_ratio):
        caseid = list(set(df['id']))
        print("# cases: {}".format(len(caseid)))
        num_test = int(len(caseid) * test_ratio)
        test_caseid = list(random.sample(caseid, num_test))
        train_caseid = [x for x in caseid if x not in test_caseid]
        train = df.loc[df['id'].isin(train_caseid)]
        test = df.loc[df['id'].isin(test_caseid)]
        return train, test

    def create_initial_log(self, path):
        df = self.read_into_panda_from_csv(path)
        return df


    def read_into_panda_from_csv(self, path, sep=','):
        df_log = pd.read_csv(filepath_or_buffer=path, header=0, sep=sep)  # , index_col=0)
        # columns = list()
        # rename_columns = list()
        # columns.append("case_id")
        # rename_columns.append("id")
        # columns.append("prefix")
        # rename_columns.append("prefix")
        # columns.append("outcome")
        # rename_columns.append("outcome")
        # columns.append("Age")
        # rename_columns.append("age")
        # df_log = df_log[columns]
        # # rename columns
        # df_log.columns = rename_columns
        return df_log

    def add_next_activity(self, df):
        df['next_activity'] = ''
        # df['next_time'] = 0
        num_rows = len(df)
        for i in range(0, num_rows - 1):
            # print(str(i) + ' out of ' + str(num_rows))

            if df.at[i, 'id'] == df.at[i + 1, 'id']:
                df.at[i, 'next_activity'] = df.at[i + 1, 'activity']
                # df.at[i, 'next_time'] = df.at[i + 1, 'complete_timestamp']
            else:
                df.at[i, 'next_activity'] = '!'
                # df.at[i, 'next_time'] = df.at[i, 'complete_timestamp']
        df.at[num_rows - 1, 'next_activity'] = '!'
        # df.at[num_rows - 1, 'next_time'] = df.at[num_rows - 1, 'complete_timestamp']

        return df

    def add_outcome(self, df):
        df['outcome'] = ''
        # df['next_time'] = 0
        num_rows = len(df)
        counter = 0
        for i in range(0, num_rows-1):
            # print(str(i) + ' out of ' + str(num_rows))
            if df.at[i, 'id'] == df.at[i + 1, 'id']:
                counter += 1
                Outcome = df.at[i+1, 'activity']
                # df.at[i, 'next_time'] = df.at[i + 1, 'complete_timestamp']
            else:
                counter = 0
                for j in range(1, counter + 2):
                    df.at[i-j, 'outcome'] = Outcome
                df.drop(i)
                # df.at[i, 'next_time'] = df.at[i, 'complete_timestamp']
        df.at[num_rows - 1, 'outcome'] = Outcome
        # df.at[num_rows - 1, 'next_time'] = df.at[num_rows - 1, 'complete_timestamp']

        return df

    def add_activity_history(self, df):
        df['activity_history'] = ""
        ids = []
        num_rows = len(df)
        prefix = str(df.at[0, 'activity'])
        df.at[0, 'activity_history'] = prefix

        for i in range(1, num_rows):
            if df.at[i, 'id'] == df.at[i - 1, 'id']:
                prefix = prefix + '+' + str(df.at[i, 'activity'])
                df.at[i, 'activity_history'] = prefix
            else:
                ids.append(df.at[i - 1, 'id'])
                prefix = str(df.at[i, 'activity'])
                df.at[i, 'activity_history'] = prefix
        return df




