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

    # LPMs and encoding types
    parser.add_argument('--LPMs', default=True, type=bool)
    parser.add_argument('--LPMs_type', default="LPMs_binary", type=str, help="LPMs_binary, LPMs_frequency")
    parser.add_argument('--encoding_type', default='C', type=str, help='W: wrapped,C: classic one-hot')

    parser.add_argument('--num_epochs', default=100, type=int)
    # models parameter if you run Main_LSTM_LPM.py
    # otherwise you do not need defining these parameters for HP_Optimization.py
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--opt', type=str, help="RMSprop or adam")
    parser.add_argument('--rate', type=float)
    parser.add_argument('--units', type=int)
    # evaluation
    parser.add_argument('--train_ratio', default=0.80, type=float)
    # data
    parser.add_argument("--dataset", type=str, default="BPIC12_ca_N",
                        help="dataset name")
    parser.add_argument('--data_dir', default="./datasets")
    parser.add_argument('--checkpoint_dir', default="./checkpoints")

    args = parser.parse_args()

    return args
