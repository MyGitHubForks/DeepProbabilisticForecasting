# Deep Probabilistic Forecasting

## Prerequisites
In your working directory, run "git clone git@github.com:Rose-ML-Lab/DeepProbabilisticForecasting.git"

## Installing
Install the packages necessary to run this code by running "pip install -r requirements.txt"
Note: requirements.txt contains more than the packages that are necessary to run this codebase

Verify a proper installation by running "python train.py" in the batchedRNN directory

## Codebase
Below are the relevant files and a brief explanation of their purpose:

- scripts/gen_adj_mx_human.py:  generates the adjascency matrix for the H3.6M human motion dataset for graphical algorithms
- scripts/gen_adj_mx.py: generates the adjascency matrix for the METR-LA traffic dataset for graphical algorithms
- scripts/generate_training_data.py: generates the data matrix for the traffic dataset
- scripts/GetPoseSequences2D.py: generate data matrix for human motion dataset
- batedRNN/utils.py: functions necessary to train the models
- batchedRNN/train.py: train script
- models/RoseSeq2Seq.py: batched sequence to sequence RNN
- models/seq2seq.py: sequence to sequence RNN with attention (uses attention.py, decoders.py, encoders.py)
- models/Data.py: objects to load and serve data as batches (see its use in batchedRNN/train.py
- models/SketchRNN.py: initial attempt at implementing SketchRNN
- dcrnn/: directory for dcrnn from previous work, intended to be a baseline, but was unused
- notebooks/GetLossObj.ipynb: helpful functions for analyzing results
- notebooks/ObservePredsRNN.ipynb: analyze RNN results

## Authors

* **Daniel Zeibergn** - zeiberg.d@husky.neu.edu
