import pandas as pd
import glob
import os
import numpy as np

folder_address = "../../Datasets/000_Experimemts/Production/explainability/"
Main_df = pd.read_csv(folder_address + "Production_Trunc23_Complete_HighCorel_Explainability.csv", sep=',')
Main_df = Main_df.sort_values(by=["case:concept:name", "event_nr"])
LPMs_number_mention = False
keep_overlapped = False
num_events = len(Main_df)
All_LPMs = dict()

LPMs_file = glob.glob(folder_address + "LPMs/*.pnml")
list_of_lpms = []
for net_file in LPMs_file:
    lpm_number = os.path.basename(net_file).split(".")[0].split("_")[1]
    list_of_lpms.append("LPM_%s" % lpm_number)

# list_of_lpms = ["LPM_"+str(i) for i in range(0, 42)]
exist_lpms = []
for l in list_of_lpms:
    try:
        All_LPMs[l] = np.array(Main_df[l])
        exist_lpms.append(l)
    except:
        continue

# non_overlapped_list = []
# with open("./Relations_mat1/Cancelled_parents_M1_ver2.txt", "r") as f:
#     for line in f:
#         non_overlapped_list.append(line.rstrip().split('_')[1].split(".")[0])

# for file in glob.glob("../../Datasets/BPIC2017/Cancelled_label/*.csv"):
#     # if os.path.basename(file).split(".")[0] not in non_overlapped_list:
#     #     continue
#     df = pd.read_csv(file)
#     if len(df) < 1:
#         continue
#     df = df.sort_values(by=["Case ID", "event_nr"])
#     for l in list_of_lpms:
#     number_lpm = os.path.basename(file).split(".")[0]
#     print(number_lpm)
#     All_LPMs[number_lpm] = np.array(df["part_of_LPM_instance"])


lpms_list_name = []
for idx in range(num_events):
    existed_lpms = "+"
    for lpm in All_LPMs:
        if All_LPMs[lpm][idx]:
            existed_lpms += lpm + "+"
    lpms_list_name.append(existed_lpms)

lpms_list_frequency = np.zeros(num_events)
for idx in range(num_events):
    print(idx)
    overlapped_number = 0
    for lpm in All_LPMs:
        if All_LPMs[lpm][idx]:
            overlapped_number += 1

    lpms_list_frequency[idx] = overlapped_number

lpms_list_binary = np.zeros(num_events)
for idx in range(num_events):
    print(idx)
    for lpm in All_LPMs:
        if All_LPMs[lpm][idx]:
            lpms_list_binary[idx] = 1
            break

Main_df["LPMs_list"] = lpms_list_name
Main_df["LPMs_binary"] = lpms_list_binary
Main_df["LPMs_frequency"] = lpms_list_frequency
Main_df = Main_df.drop(columns=exist_lpms)

Main_df.to_csv(folder_address + "Production_Trunc23_Complete_HighCorel_Explainability_Agg.csv", index=False, sep=";")
