# MIMIC-DL2

This project uses the [DL2 framework](https://github.com/eth-sri/dl2) to create logical constraints for the in-hospital-mortality benchmark dataset for the [MIMIC-III Database](https://physionet.org/content/mimiciii/1.4/). See following paper for the MIMIC-III benchmarks:
[Multitask learning and benchmarking with clinical time series data](https://arxiv.org/abs/1703.07771).

## Installation
The code has been tested on machines running python >= 3.8.
To install and run the code, it is assumed that one has access to the following:

- Our fork of the [DL2 framework](https://github.com/IvoAA/dl2) - OBS. it must be the master branch. 
- The [MIMIC-III Benchmarks Code](https://github.com/YerevaNN/mimic3-benchmarks). 
- The in-hospital-mortality dataset placed inside the 'data' folder.

The former two should simply be cloned locally, and then the locations must be added to the PYTHONPATH. Or they can be installed directly from github through your favourite package manager.

To recreate the in-hospital-mortality dataset, one must have access to the MIMIC-III Database. Then, the benchmark can be created by following the instructions outlined in the repository for the MIMIC-III benchmarks as given above.

The remaining dependencies are specified in the [requirements file](requirements.txt).

## Training the network
Run the 'main.py' file. The file accepts the following arguments:

|     Argument     | type   | Required | Default | Help                                                        |
| :--------------- | :----- | :------- | :------ | :---------------------------------------------------------- |
| --batch-size     |  int   |  False   |  64     |  Number of samples in a batch.                              |
| --constraint     |  str   |  True    |         |  The constraint to train with.                              |
| --delay          |  int   |  False   |  0      |  Number of epochs to wait before training with constraints. |
| --dl2-weight     |  float |  False   |  0.0    |  Weight of DL2 loss.                                        |
| --grid-search    |        |  False   |         |  Perform a grid-search for l2-weight and pos-weight.        |
| --l2             |  float |  False   |  0.01   |  L2 regularizxation weight.                                 |
| --num-epochs     |  int   |  False   |  200    |  Number of epochs to train for.                             |
| --num-iters      |  int   |  False   |  50     |  Number of oracle iterations.                               |
| --pos-weight     |  float |  False   |  3      |  Weight of positive examples in loss function.              |
| --print-freq     |  int   |  False   |  10     |  Print frequency for batches.                               |
| --report-dir     |  str   |  True    |         |  Directory where results should be stored.                  |
| --verbose        |        |  False   |         |  Print metrics and results.                                 |



### Example
  ```bash
  main.py --batch-size 64  --constraint Mimic3Constraint()  --dl2-weight 0.1  --num-epochs 200  --report-dir reports  --verbose
  ```
