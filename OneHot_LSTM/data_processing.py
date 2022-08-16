import argparse
import os
import pickle
import time
from utils.processor import LogsDataProcessor

parser = argparse.ArgumentParser(
    description="LSTM - LPMs encoding - Data Processing.")

parser.add_argument("--dataset",
                    type=str,
                    default="production_LPMs_HPopt",
                    help="dataset name")

parser.add_argument("--dir_path",
                    type=str,
                    default="./datasets",
                    help="path to store processed data")

parser.add_argument("--raw_log_file",
                    type=str,
                    default="./datasets/Production_Trunc23_completeLPMs_Aggregated.csv",
                    help="path to raw csv log file")

parser.add_argument('--Max_length', default=23, type=float)
parser.add_argument('--train_ratio', default=0.80, type=float)

args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(f"{args.dir_path}/{args.dataset}"):
        os.makedirs(f"{args.dir_path}/{args.dataset}")

    output_datasets_address = f"{args.dir_path}/{args.dataset}"

    print("data preprocessing...")
    start = time.time()
    data_processor = LogsDataProcessor(filepath=args.raw_log_file, dire_path=output_datasets_address,
                                       pool=1, LPMs=True)

    Max_prefix_length, coded_activity, coded_labels, coded_lpms = data_processor.process_logs(
        train_ratio=args.train_ratio,
        Max_length=args.Max_length)

    with open(output_datasets_address + '/coded_activity.pkl', 'wb') as handle:
        pickle.dump(coded_activity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_datasets_address + '/coded_labels.pkl', 'wb') as handle:
        pickle.dump(coded_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_datasets_address + '/coded_lpms.pkl', 'wb') as handle:
        pickle.dump(coded_lpms, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_datasets_address + '/Max_prefix_length.pkl', 'wb') as handle:
        pickle.dump(Max_prefix_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print(f"Total data preprocessing time: {end - start}")
