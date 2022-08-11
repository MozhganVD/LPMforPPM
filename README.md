# Encoding High-Level Control-Flow Construct Information for Process Outcome Prediction

This repository provides implementations for encoding high-level control-flow constructs information for outcome-oriented predictive process monitoring.
To use this repository, you need to discover Local Process Models using LPM miner plugin available in ProM 6.9 and store discovered patterns as petri-nets in .pnml format.


# Datasets
The labeled datasets to do experiments can be found at https://github.com/irhete/predictive-monitoring-benchmark 

# Usage
You first need to the following steps in order to generate LPMs feature from discovered LPMs. Then you can choose between On-hot encoding and Embedding method to encode generated features and learn the LSTM model.

### LPM Feature Generation
