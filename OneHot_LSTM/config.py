import os
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load():
    parser = argparse.ArgumentParser()

    # features
    parser.add_argument('--features', default=False, type=bool)
    parser.add_argument('--LPMs', default=False, type=bool)
    parser.add_argument('--LPMs_type', default="None", type=str, help="LPMs_binary, LPMs_frequency, None")
    parser.add_argument('--LPMs_Normal', default=False, type=bool)
    # dnn
    parser.add_argument('--num_epochs', default=100, type=int)

    # all models
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--opt', type=str, help="RMSprop or adam")
    parser.add_argument('--rate', type=float)
    parser.add_argument('--units', type=int)
    # evaluation
      # LSTM 256 #dnc 1
    parser.add_argument('--trunc_ratio', default=23, type=float)  # if bigger than one, means that it is a fix number!
    parser.add_argument('--train_ratio', default=0.80, type=float)

    # data
    parser.add_argument("--dataset", type=str, default="bpic11_1_30LPMs_HPopt",
                        help="dataset name")

    parser.add_argument("--raw_log_file",type=str,
                        default="../bpic2011_f1_aggregated_trunc36_All30LPMs.csv",
                        help="path to raw csv log file")
    # parser.add_argument('--data_set', default="outcome_train.csv")
    # parser.add_argument('--data_set_test', default="outcome_test.csv")
    parser.add_argument('--data_dir', default="./datasets")
    parser.add_argument('--checkpoint_dir', default="./checkpoints")

    #model
    parser.add_argument('--model', default="LSTM")

    args = parser.parse_args()

    return args
