import argparse
import os
import pickle
import config
import time
from utils.processor import LogsDataProcessor
import pandas as pd

parser = argparse.ArgumentParser(
    description="LSTM - LPMs encoding - Data Processing.")

parser.add_argument("--dataset",
                    type=str,
                    default="production_30LPMs_HPopt_CompleteLPMs_NoCorrel",
                    help="dataset name")

parser.add_argument("--dir_path",
                    type=str,
                    default="./datasets",
                    help="path to store processed data")

parser.add_argument("--raw_log_file",
                    type=str,
                    default="../datasets/Production_Trunc23_completeLPMs_NoCorrelatedLPMS_Agg.csv",
                    help="path to raw csv log file")

parser.add_argument("--task",
                    type=str,
                    default="outcome",
                    help="task name")

parser.add_argument("--sort_temporally",
                    type=bool,
                    default=False,
                    help="sort cases by timestamp")

parser.add_argument("--features",
                    type=bool,
                    default=False,
                    help="training with patient features")

parser.add_argument("--lpms",
                    type=bool,
                    default=True,
                    help="training with patient features")

parser.add_argument("--incl_label",
                    type=bool,
                    default=True,
                    help="True if dataset includes outcome labels, False otherwise")

parser.add_argument("--label_type",
                    type=int,
                    default=2,
                    help="1: predicting return er , 2: predicting non A release ")

parser.add_argument("--deleted_activities",
                    type=str,
                    default="[release-a,release-b,release-c,release-d,release-e,return-er]")

parser.add_argument("--complete_lpms",
                    type=bool,
                    default=False)

parser.add_argument('--trunc_ratio', default=23, type=float)  # if bigger than one, means that it is a fix number!
parser.add_argument('--train_ratio', default=0.80, type=float)

args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(f"{args.dir_path}/{args.dataset}"):
        os.makedirs(f"{args.dir_path}/{args.dataset}")

    output_datasets_address = f"{args.dir_path}/{args.dataset}"

    print("data preprocessing...")
    start = time.time()
    data_processor = LogsDataProcessor(filepath=args.raw_log_file, dire_path=output_datasets_address,
                                       pool=1,
                                       features=args.features, LPMs=True)

    Max_prefix_length, coded_activity, coded_labels = data_processor.process_logs(train_ratio=args.train_ratio,
                                                                                  trunc_ratio=args.trunc_ratio,
                                                                                  complete_lpms=args.complete_lpms)

    with open(output_datasets_address + '/coded_activity.pkl', 'wb') as handle:
        pickle.dump(coded_activity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_datasets_address + '/coded_labels.pkl', 'wb') as handle:
        pickle.dump(coded_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_datasets_address + '/Max_prefix_length.pkl', 'wb') as handle:
        pickle.dump(Max_prefix_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print(f"Total data preprocessing time: {end - start}")
