import pm4py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def test_train_spliting(df, train_ratio, trunc_ratio):
    case_id = "case:concept:name"
    if trunc_ratio > 1:
        Max_prefix_length = trunc_ratio
    else:
        Max_prefix_length = int(
            np.ceil(df[df['label'] == 'deviant'].groupby(case_id).size().quantile(trunc_ratio)))

    Cases = []
    Prefixes = []
    Labels = []
    for case in df[case_id].unique():
        Cases.append(case)
        Prefixes.append(min(Max_prefix_length, len(df[df[case_id] == case])))
        Labels.append(list(df[df[case_id] == case]['label'].unique())[0])

    Case_indexes = pd.DataFrame({case_id: Cases, "prefix length": Prefixes, 'label': Labels})
    median_prefix = np.median(list(Case_indexes["prefix length"]))
    first_half = Case_indexes[Case_indexes["prefix length"] < median_prefix]
    second_half = Case_indexes[Case_indexes["prefix length"] > median_prefix]
    quartile_1 = np.median(list(first_half["prefix length"]))
    quartile_2 = np.median(list(second_half["prefix length"]))
    Case_indexes["prefix group"] = pd.cut(Case_indexes["prefix length"],
                                          bins=[0, quartile_1, median_prefix, quartile_2],
                                          include_lowest=True, labels=[quartile_1, median_prefix, quartile_2])

    train, test = train_test_split(Case_indexes, train_size=train_ratio, shuffle=True,
                                   stratify=pd.concat([Case_indexes["prefix group"],
                                                       Case_indexes['label']], axis=1))
    #
    # train, test = train_test_split(Case_indexes, train_size=train_ratio,
    #                                stratify=Case_indexes[self.label_col])

    train_list = train[case_id]
    test_list = test[case_id]

    return train_list, test_list, Max_prefix_length


Main_log = pm4py.read_xes("../../../Datasets/BPIC2012/bpic2012_O_Accepted/bpic_2012_1_trim40.xes")
Main_df = pm4py.convert_to_dataframe(Main_log)

train_list, test_list, _ = test_train_spliting(Main_df, 0.8, 40)
train_df = Main_df[Main_df["case:concept:name"].isin(train_list)]
test_df = Main_df[Main_df["case:concept:name"].isin(test_list)]

regular_df = train_df[train_df["label"] == "regular"]
deviant_df = train_df[train_df["label"] == "deviant"]

regular_df.to_csv("../../../Datasets/BPIC2012/bpic2012_O_Accepted/Discriminatives/bpic12_1_trunc40_train_regular.csv", index=False)
deviant_df.to_csv("../../../Datasets/BPIC2012/bpic2012_O_Accepted/Discriminatives/bpic12_1_trunc40_train_deviant.csv", index=False)
test_df.to_csv("../../../Datasets/BPIC2012/bpic2012_O_Accepted/Discriminatives/bpic12_1_trunc40_test.csv", index=False)
train_df.to_csv("../../../Datasets/BPIC2012/bpic2012_O_Accepted/Discriminatives/bpic12_1_trunc40_train_all.csv", index=False)

