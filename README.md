# Encoding High-Level Control-Flow Construct Information for Process Outcome Prediction

This repository provides implementations for encoding high-level control-flow constructs information for outcome-oriented predictive process monitoring.
To use this repository, you need to discover Local Process Models using LPM miner plugin available in ProM 6.9 and store discovered patterns as petri-nets in .pnml format.


# Datasets
The labeled datasets to do experiments can be found at https://github.com/irhete/predictive-monitoring-benchmark 

# Usage

### LPM Feature Generation
1. Discover and save LPMs using ProM as .pnml format
2. Save event logs in xes format 
3. Run ```LPMDetection_Complete.py``` with following flags:
    -  *--LPMs_dir*: path/to/discovered/LPMs
    -  *--raw_log_file*: path/to/raw/eventlog/.xes
    -  *--processed_log_file*: path/to/save/processed/eventlog/.csv
    -  *--Min_prefix_size*
    -  *--Max_prefix_size*

- Example:

```LPMDetection_Complete.py --LPMs_dir "./LPMs" --raw_log_file "./datasets/eventlog.xes" --processed_log_file "./datasets/eventlog_processed.csv" --Min_prefix_size 2 --Max_prefix_size 36``` 

### One-hot encoding (Classic/ Wrapped)
1. Prepare dataset by running ```data_processing.py``` with following flags:
    -  *--dataset*: dataset name
    -  *--dir_path*: path/to/store/processed/data
    -  *--raw_log_file*: path/to/processed/eventlog/.csv
    -  *--Max_length*
    -  *--train_ratio*
