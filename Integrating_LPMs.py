import argparse
import pandas as pd
import glob
import os
import numpy as np


parser = argparse.ArgumentParser(description="LPM Integration")

parser.add_argument("--LPMs_dir",
                    type=str,
                    default="./lpm_folder",
                    help="path to lpms (.pnml format)")

parser.add_argument("--log_file",
                    type=str,
                    default="/Production_Trunc23_Complete_HighCorel_Explainability.csv",
                    help="path to event log with lpms feature in csv format")

parser.add_argument("--processed_log_file",
                    type=str,
                    default="./datasets/Production_Trunc23_completeLPMs_Aggregated.csv",
                    help="path to location store and name of the output file in csv format")

args = parser.parse_args()

if __name__ == '__main__':
    data_dire = args.log_file
    lpm_address = args.LPMs_dir
    output_file = args.processed_log_file

    Main_df = pd.read_csv(data_dire , sep=',')
    Main_df = Main_df.sort_values(by=["case:concept:name", "event_nr"])
    LPMs_number_mention = False
    keep_overlapped = False
    num_events = len(Main_df)
    All_LPMs = dict()

    LPMs_file = glob.glob(lpm_address + "/*.pnml")
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

    Main_df.to_csv(output_file, index=False, sep=";")
