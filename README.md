# Encoding High-Level Control-Flow Construct Information for Process Outcome Prediction

Supplementary material for the article *"Encoding High-Level Control-Flow Construct Information for Process Outcome Prediction"* by Mozhgan Vazifehdoostirani, Laura Genga, Remco Dijkman. 

This repository provides implementations for encoding high-level control-flow constructs information for outcome-oriented predictive process monitoring.
To use this repository, you need to discover Local Process Models using LPM miner plugin available in ProM 6.9 and store discovered patterns as petri-nets in .pnml format.


# Datasets
The labeled datasets to do experiments can be found at https://github.com/irhete/predictive-monitoring-benchmark 

# Usage

- Install dependencies (Python 3.8.0) :

```pip install -r requirements.txt```

You first need to generate LPMs feature for each event log, then you can choose between one-hot encoding based methods or embedding layers to encode and train the LSTM model as described step by step below:

## LPMs Feature Generation
1. Discover and save LPMs using ProM as .pnml format
2. Save event logs in xes format 
3. Run ```LPMDetection_Complete.py``` with following flags:
    -  *--LPMs_dir*: path/to/discovered/LPMs
    -  *--raw_log_file*: path/to/raw/eventlog/.xes
    -  *--processed_log_file*: path/to/save/processed/eventlog/.csv
    -  *--Min_prefix_size*
    -  *--Max_prefix_size*

- Example:

```Python LPMDetection_Complete.py --LPMs_dir "./LPMs" --raw_log_file "./datasets/eventlog.xes" --processed_log_file "./datasets/eventlog_processed.csv" --Min_prefix_size 2 --Max_prefix_size 36``` 

## One-hot encoding (Classic/ Wrapped)
1. Prepare dataset by running ```data_processing.py``` with following flags:
    -  *--dataset*: dataset name
    -  *--dir_path*: path/to/store/processed/data
    -  *--raw_log_file*: path/to/processed/eventlog/.csv
    -  *--Max_length*
    -  *--train_ratio*

- Example:

```Python data_processing.py --dataset "Production" --dir_path "./datasets" --raw_log_file "./datasets/eventlog_processed.csv" --Max_length 36 --train_ratio 0.8``` 

2. Hyperparameter tuning by running ```HP_Optimization.py``` with following flags:
    -  *--dataset*: dataset name (same name as data processing name)
    -  *--data_dir*: path/to/store/processed/data
    -  *--checkpoint_dir*: path/to/sdave/results
    -  *--LPMs*: True/False
    -  *--encoding_type*: W: wrapped, C: classic one-hot
    -  *--LPMs_type*: LPMs_binary/LPMs_frequency (if you choose the wrapped for encoding type)
    
- Example:

```Python HP_Optimization.py --dataset "Production" --dir_path "./datasets" --checkpoint_dir "./checkpoints" --LPMs True --encoding_type "W" --LPMs_type "LPMs_binary"``` 

3. Run LSTM model with a predifined parameters by running ```Main_LSTM_LPMs.py``` with following flags:
    -  *--dataset*: dataset name (same name as data processing name)
    -  *--data_dir*: path/to/store/processed/data
    -  *--checkpoint_dir*: path/to/sdave/results
    -  *--LPMs*: True/False
    -  *--encoding_type*: W: wrapped, C: classic one-hot
    -  *--LPMs_type*: LPMs_binary/LPMs_frequency (if you choose the wrapped for encoding type)
    -  *--learning_rate*
    -  *--batch_size*
    -  *--layers*: number of LSTM layers
    -  *--opt*: RMSprop or adam
    -  *--rate*: dropout rate
    -  *--units*: number of neuron units per layer
    
## Embedding layers 
1. Hyperparameter tuning by running ```HPO_embedding_args.py``` with following flags:
    -  *--data_dir*: path/to/save/results
    -  *--raw_data*: path/to/processed/eventlog/.csv
    -  *--out_name*: dataset name
    -  *--LPMs*: True/False
    -  *--Only_LPMs*: True/False (if you want to train model only with LPMs not activities, in this case both LPMs and Only_LPMs flags must be True.)
    -  *--max_length*
    -  *--results*: path/to/save/results/in/.text
    
 - Example:
 
 ```Python HPO_embedding_args.py --data_dir "./datasets" --raw_data "./EventLog.csv" --out_name "production" --LPMs True --Only_LPMs False --max_length 36 --results "./results_lpms_act.txt"``` 
 
2. Run LSTM model with a predifined parameters by running ```Embedding_Run.py``` with following flags:
    -  *--data_dir*: path/to/save/results
    -  *--raw_data*: path/to/processed/eventlog/.csv
    -  *--out_name*: dataset name
    -  *--LPMs*: True/False
    -  *--Only_LPMs*: True/False (if you want to train model only with LPMs not activities, in this case both LPMs and Only_LPMs flags must be True.)
    -  *--max_length*
    -  *--results*: path/to/save/results/in/.text
    -  *--batch_size*
    -  *--embedding_act*: dimension of the embedding layers for activities 
    -  *--embedding_lpms*: dimension of the embedding layers for LPMs
    -  *--learning_rate*
    -  *--opt*: RMSprop or adam
    -  *--rate*: dropout rate
    -  *--units*: number of neuron units per layer
    -  *--layers*: number of LSTM layers
    

## Note
- We assume the input csv file contains the columns named after the xes elements, e.g., concept:name
- We assume the input event log contains a column named "event_nr" indicating the event orders for each case 

    
